#pragma once

#include <alpaka/alpaka.hpp>
#include <simulation/kernels/nBodyKernel.hpp>
// Vector
#include <simulation/types/vector.hpp> 

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    #define ACC_UPDATEP alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>,std::size_t>
    #define STREAM alpaka::stream::StreamCudaRtSync
#elif defined(ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
    #define ACC_UPDATEP alpaka::acc::AccCpuOmp4<alpaka::dim::DimInt<1u>,std::size_t>
    #define STREAM alpaka::stream::StreamCpuSync
#elif defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    #define ACC_UPDATEP alpaka::acc::AccCpuOmp2Blocks<alpaka::dim::DimInt<1u>,std::size_t>
    #define STREAM alpaka::stream::StreamCpuSync
#else
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
    decltype( alpaka::pltf::getDevByIdx<alpaka::pltf::Pltf<alpaka::dev::Dev<ACC_UPDATEP>>>(0u) ) devAccUpdateP;
    STREAM streamUpdateP;
    alpaka::dev::DevCpu devHost;

    alpaka::Vec<
        alpaka::dim::DimInt<1u>,TSize>
        const extentBodies;

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
        types::Vector<NDim-1,TElem>,
        alpaka::dim::DimInt<1u>,
        TSize> hostBodiesVelocity;

    //Data on Acc
    decltype( alpaka::mem::buf::alloc
            <types::Vector<NDim,TElem> , TSize>(devAccUpdateP, extentBodies) ) accBodiesPosition;
    decltype( alpaka::mem::buf::alloc
            <types::Vector<NDim-1,TElem> , TSize>(devAccUpdateP, extentBodies) ) accBodiesVelocity;

    TSize numBodies;
    float gravitationalConstant;// = 6.674e-11;
    float smoothnessFactor;
    //flag if a new step had been done
    bool stepFlag = true;
public:
    std::size_t elements = 1; //Alpaka elements
    /**
     */
    Simulation(
            types::Vector<NDim,TElem> * bodiesPosition,
            types::Vector<NDim-1,TElem> * bodiesVelocity,
            TSize numBodies,
            float smoothnessFactor,
            float gravitationalConstant) :
        devAccUpdateP( alpaka::pltf::getDevByIdx<alpaka::pltf::Pltf<alpaka::dev::Dev<ACC_UPDATEP>>>(0u) ) ,
        streamUpdateP(devAccUpdateP),
        devHost(alpaka::pltf::getDevByIdx<alpaka::pltf::PltfCpu>(0u)),
        extentBodies(numBodies),
        hostBodiesPosition(bodiesPosition, devHost, extentBodies),
        hostBodiesVelocity(bodiesVelocity, devHost, extentBodies),
        accBodiesPosition( alpaka::mem::buf::alloc<types::Vector<NDim,TElem> , TSize>
            ( devAccUpdateP, extentBodies ) ),
        accBodiesVelocity( alpaka::mem::buf::alloc<types::Vector<NDim-1,TElem> , TSize>
            ( devAccUpdateP, extentBodies ) ),
        numBodies(numBodies),
        gravitationalConstant(gravitationalConstant),
        smoothnessFactor(smoothnessFactor)

    {

        /*** Memory copy ***/
        alpaka::mem::view::copy(
            streamUpdateP,
            accBodiesPosition,
            hostBodiesPosition,
            extentBodies );

        alpaka::mem::view::copy(
            streamUpdateP,
            accBodiesVelocity,
            hostBodiesVelocity,
            extentBodies );

        //Wait for data
        alpaka::wait::wait(streamUpdateP );

    }
    /*** Funtion to execute a simulation step ***/
    void step(TTime dt)
    {   
        this->stepFlag = true;

       
        /*** Execute NBodyKernel ***/
        auto const workDivNBody(
                alpaka::workdiv::getValidWorkDiv< ACC_UPDATEP >(
                    devAccUpdateP,
                    extentBodies,
                    alpaka::Vec<
                        alpaka::dim::DimInt<1u>,
                        TSize
                    >(this->elements),
                    false,
                    alpaka::workdiv::GridBlockExtentSubDivRestrictions::
                    Unrestricted
                )
        );
        kernels::NBodyKernel nBodyKernel;
        auto const nBodyExec(
                alpaka::exec::create<ACC_UPDATEP>(
                    workDivNBody,
                    nBodyKernel,
                    alpaka::mem::view::getPtrNative( accBodiesPosition ),
                    alpaka::mem::view::getPtrNative( accBodiesVelocity ),
                    numBodies,
                    gravitationalConstant,
                    smoothnessFactor,
                    dt
                )
        );

        alpaka::stream::enqueue( streamUpdateP, nBodyExec);
        alpaka::wait::wait( streamUpdateP );

    }

    types::Vector<NDim,TElem> * getPositions(){
        if(stepFlag)
        {
            alpaka::mem::view::copy(
                streamUpdateP,
                hostBodiesPosition,
                accBodiesPosition,
                extentBodies);

            alpaka::wait::wait( streamUpdateP );
        }
        stepFlag =false;
        return alpaka::mem::view::getPtrNative(hostBodiesPosition);
    }

};

} //end namespace simulation

} //end namespace nbody
