#pragma once

#include <alpaka/alpaka.hpp>
// ForceMatrixKernel
#include <simulation/kernels/forceMatrixKernel.hpp>
//updatePositionKernel
#include <simulation/kernels/updatePositionsKernel.hpp>
// Vector
#include <simulation/types/vector.hpp> 

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLE)
    #define ACC_FORCEM alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<2u>,std::size_t>
    #define ACC_UPDATEP alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>,std::size_t>
    #define STREAM alpaka::stream::StreamCudaRtSync
#elif defined(ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
    #define ACC_FORCEM alpaka::acc::AccCpuOmp4<alpaka::dim::DimInt<2u>,std::size_t>
    #define ACC_UPDATEP alpaka::acc::AccCpuOmp4<alpaka::dim::DimInt<1u>,std::size_t>
    #define STREAM alpaka::stream::StreamCpuSync
#else
    #define ACC_FORCEM alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>,std::size_t>
    #define ACC_UPDATEP alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<1u>,std::size_t>
    #define STREAM alpaka::stream::StreamCpuSync
#endif
    
namespace nbody {

namespace simulation {
    
    /** Class Simulation
     *
     * This Class provides an esay interface to the N-body simulation
     */
template<
    std::size_t NDim,
    typename TElem,
    typename TTime,
    typename TSize
    >
class Simulation
{
private:
    //alpaka
    decltype( alpaka::dev::DevMan<ACC_FORCEM>::getDevByIdx(0) ) devAccForceM;
    decltype( alpaka::dev::DevMan<ACC_UPDATEP>::getDevByIdx(0) ) devAccUpdateP;
    STREAM streamForceM;
    STREAM streamUpdateP;
    alpaka::dev::DevCpu devHost;
    TSize pitchBytesForceMatrix;

    alpaka::Vec<
        alpaka::dim::DimInt<1u>,TSize>
        const extentBodies;

    alpaka::Vec<
        alpaka::dim::DimInt<2u>,TSize>
        const extentForceMatrix;
    
    //Data on Host
    alpaka::mem::view::ViewPlainPtr<
        //std::decay<decltype(devHost)>::type,
        alpaka::dev::DevCpu,
        types::Vector<NDim,TElem>,
        alpaka::dim::DimInt<1u>,
        TSize> hostBodiesPosition;
    alpaka::mem::view::ViewPlainPtr<
        //std::decay<decltype(devHost)>::type,
        alpaka::dev::DevCpu,
        types::Vector<NDim,TElem>,
        alpaka::dim::DimInt<1u>,
        TSize> hostBodiesVelocity;
    alpaka::mem::view::ViewPlainPtr<
        //std::decay<decltype(devHost)>::type,
        alpaka::dev::DevCpu,
        TElem,
        alpaka::dim::DimInt<1u>,
        TSize> hostBodiesMass;

    //Data on Acc
    decltype( alpaka::mem::buf::alloc
            <types::Vector<NDim,TElem> , TSize>(devAccForceM, extentForceMatrix) ) accForceMatrix;
    decltype( alpaka::mem::buf::alloc
            <types::Vector<NDim,TElem> , TSize>(devAccForceM, extentBodies) ) accBodiesPosition;
    decltype( alpaka::mem::buf::alloc
            <types::Vector<NDim,TElem> , TSize>(devAccForceM, extentBodies) ) accBodiesVelocity;
    decltype( alpaka::mem::buf::alloc
            <TElem, TSize>(devAccForceM, 1) ) accBodiesMass;

    TSize numBodies;
    float gravitationalConstant;// = 6.674e-11;
    float smoothnessFactor;
    //flag if a new step had been done
    bool stepFlag = true;
public:
    /**
     */
    Simulation(
            types::Vector<NDim,TElem> * bodiesPosition,
            types::Vector<NDim,TElem> * bodiesVelocity,
            TElem * bodiesMass,
            TSize numBodies,
            float smoothnessFactor,
            float gravitationalConstant) :
        devAccForceM(alpaka::dev::DevMan<ACC_FORCEM>::getDevByIdx(0)),
        devAccUpdateP(alpaka::dev::DevMan<ACC_FORCEM>::getDevByIdx(0)),
        streamForceM(devAccForceM),
        streamUpdateP(devAccUpdateP),
        devHost(alpaka::dev::DevManCpu::getDevByIdx(0)),
        extentBodies(numBodies),
        extentForceMatrix(numBodies,numBodies),
        hostBodiesPosition(bodiesPosition, devHost, extentBodies),
        hostBodiesVelocity(bodiesVelocity, devHost, extentBodies),
        hostBodiesMass(bodiesMass, devHost, extentBodies),
        accForceMatrix( alpaka::mem::buf::alloc<types::Vector<NDim,TElem> , TSize>
            ( devAccForceM, extentForceMatrix ) ),
        accBodiesPosition( alpaka::mem::buf::alloc<types::Vector<NDim,TElem> , TSize>
            ( devAccForceM, extentBodies ) ),
        accBodiesVelocity( alpaka::mem::buf::alloc<types::Vector<NDim,TElem> , TSize>
            ( devAccForceM, extentBodies ) ),
        accBodiesMass( alpaka::mem::buf::alloc<TElem , TSize>
            ( devAccForceM, extentBodies ) ),
        numBodies(numBodies),
        gravitationalConstant(gravitationalConstant),
        smoothnessFactor(smoothnessFactor)

    {

        /*** Memory copy ***/
        alpaka::mem::view::copy(
            streamForceM,
            accBodiesPosition,
            hostBodiesPosition,
            extentBodies );

        alpaka::mem::view::copy(
            streamForceM,
            accBodiesVelocity,
            hostBodiesVelocity,
            extentBodies );

        alpaka::mem::view::copy(
            streamForceM,
            accBodiesMass,
            hostBodiesMass,
            extentBodies );
        //Wait for data
        alpaka::wait::wait( streamForceM );

    }
    /*** Funtion to execute a simulation step ***/
    void step(TTime dt)
    {   
        this->stepFlag = true;

        //Executing the ForceMatrixKernel
        auto const workDivForceM(
                alpaka::workdiv::getValidWorkDiv< ACC_FORCEM >(
                    devAccForceM,
                    extentForceMatrix,
                    alpaka::Vec<
                        alpaka::dim::DimInt<2u>,
                        TSize
                    >::ones(),
                    false,
                    alpaka::workdiv::GridBlockExtentSubDivRestrictions::
                    EqualExtent
                )
        );

        kernels::ForceMatrixKernel forceMatrixKernel;

        auto const forceKernelExec(
                alpaka::exec::create<ACC_FORCEM>(
                    workDivForceM,
                    forceMatrixKernel,
                    alpaka::mem::view::getPtrNative( accBodiesPosition ),
                    alpaka::mem::view::getPtrNative( accBodiesMass ),
                    alpaka::mem::view::getPtrNative( accForceMatrix ),
                    static_cast<TSize>(
                        alpaka::mem::view::getPitchBytes<1u>
                            (accForceMatrix)
                    ),
                    numBodies,
                    smoothnessFactor
                )
        );
        
        alpaka::stream::enqueue( streamForceM, forceKernelExec);
        alpaka::wait::wait( streamForceM );

        /*** Execute updatePositionKernel ***/
        auto const workDivUpdatePositions(
                alpaka::workdiv::getValidWorkDiv< ACC_UPDATEP >(
                    devAccUpdateP,
                    extentBodies,
                    alpaka::Vec<
                        alpaka::dim::DimInt<1u>,
                        TSize
                    >::ones(),
                    false,
                    alpaka::workdiv::GridBlockExtentSubDivRestrictions::
                    Unrestricted
                )
        );
        kernels::UpdatePositionsKernel updatePositionsKernel;
        auto const updatePositionsExec(
                alpaka::exec::create<ACC_UPDATEP>(
                    workDivUpdatePositions,
                    updatePositionsKernel,
                    alpaka::mem::view::getPtrNative( accForceMatrix ),
                    alpaka::mem::view::getPtrNative( accBodiesPosition ),
                    alpaka::mem::view::getPtrNative( accBodiesVelocity ),
                    static_cast<TSize>(
                        alpaka::mem::view::getPitchBytes<1u>
                            ( accForceMatrix )
                    ),
                    numBodies,
                    gravitationalConstant,
                    dt
                )
        );

        alpaka::stream::enqueue( streamUpdateP, updatePositionsExec);
        alpaka::wait::wait( streamUpdateP );

    }

    types::Vector<NDim,TElem> * getPositions(){
        if(stepFlag)
        {
            alpaka::mem::view::copy(
                streamForceM,
                hostBodiesPosition,
                accBodiesPosition,
                extentBodies);

            alpaka::wait::wait( streamForceM );
        }
        stepFlag = false;
        return alpaka::mem::view::getPtrNative(hostBodiesPosition);
    }

};

} //end namespace simulation

} //end namespace nbody
