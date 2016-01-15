#pragma once

#include <alpaka/alpaka.hpp>
// ForceMatrixKernel
#include <simulation/kernels/forceMatrixKernel.hpp>
//updatePositionKernel
#include <simulation/kernels/updatePositionKernel.hpp>
// Vector
#include <simulation/kernels/types/vector.hpp> 

namespace nbody {

namespace simulation {
    
    /** Class Simulation
     *
     * This Class provides an esay interface to the N-body simulation
     */
template<
    typename TAcc,
    typename TStream,
    TSize NDim,
    typename TElem,
    typename TTime,
    typname TSize
    >
class Simulation
{
private:
    //alpaka
    TStream * stream;
    TAcc devAcc;
    alpaca::dev::DevCpu devHost;
    TSize pitchBytesForceMatrix;
    
    //Data on Host
    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        types::Vector<NDim,TElem>,
        alpaka::dim::DimInt<1u>,
        TSize> * hostBodiesPosition;
    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        types::Vector<NDim,TElem>,
        alpaka::dim::DimInt<1u>,
        TSize> * hostBodiesVelocity;
    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        TElem,
        alpaka::dim::DimInt<1u>,
        TSize> * hostBodiesMass;

    //Data on Acc
    types::Vector<NDim,TElem> * accForceMatrix;
    types::Vector<NDim,TElem> * accBodiesPosition;
    types::Vector<NDim,TElem> * accBodiesVelocity;
    TElem * accBodiesMass;

    TSize numBodies;
    float gravitationalConstant(6.674e-11);
    float smoothnessFactor;
    //flag if a new step had been done
    bool stepFlag(true);
public:
    /**
     */
    Simulation(
            types::Vector<NDim,TElem> * bodiesPosition,
            types::Vector<NDim,TElem> * bodiesVelocity,
            TElem * bodiesMass,
            TSize numBodies,
            float smoothnessFactor)
    {
        /*** Devices ***/
        this->devAcc = alpaka::dev::DevMan<TAcc>::getDevByIdx(0);
        this->stream = new TStream(this->devAcc);
        this->devHost = alpaka::dev::DevManCpu::getDevByIdx(0);
        /*** Work extent ***/
        alpaka::Vec<
            alpaka::dim::DimInt<1u>,TSize>
            const extentBodies(numBodies);

        alpaka::Vec<
            alpaka::dim::DimInt<2u>,TSize>
            const extentForceMatrix(numBodies,numBodies);

        /*** Memory Host ***/

        this->numBodies = numBodies;
        this->smoothnesFactor = smoothnessFactor;

        this->hostBodiesPosition = new alpaka::mem::view::ViewPlainPtr<
            std::decay<decltype(devHost)>::type,
            types::Vector<NDim,TElem>,
            alpaka::dim::DimInt<1u>,
            TSize>(bodiesPosition, this->devHost, extentBodies);

        this->hostBodiesVelocity = new alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
            types::Vector<NDim,TElem>,
            alpaka::dim::DimInt<1u>,
            TSize>(bodiesVelocity, this->devHost, extentBodies);

        this->hostBodiesMass = new alpaka::mem::view::ViewPlainPtr<
            std::decay<decltype(devHost)>::type,
            TElem,
            alpaka::dim::DimInt<1u>,
            TSize>(bodiesMass, this->devHost, extentBodies);

        /*** Memory Acc ***/

        this-> accForceMatrix =
            alpaka::mem::buf::alloc<Vector,Size>( devAcc, extentForceMatrix );

        this->accBodiesPosition =
            alpaka::mem::buf::alloc<Vector, Size>( devAcc, extentBodies );

        this->accBufBodiesVelocity =
            alpaka::mem::buf::alloc<Vector, Size>( devAcc, extentBodies );

        this->accBufBodiesMass =
            alpaka::mem::buf::alloc<float, Size>( devAcc, extentBodies );

        /*** Memory copy ***/
        alpaka::mem::view::copy(
            * stream,
            accBodiesPosition,
            * hostBodiesPosition,
            extentBodies );

        alpaka::mem::view::copy(
            * stream,
            accBodiesVelocity,
            * hostBodiesVelocity,
            extentBodies );

        alpaka::mem::view::copy(
            * stream,
            accBodiesMass,
            * hostBodiesMass,
            extentBodies );
        //Wait for data
        alpaka::wait::wait( * stream );

    }
    /*** Funtion to execute a simulation step ***/
    void step(TTime dt)
    {   
        this->stepFlag = true;

        //Executing the ForceMatrixKernel
        auto const workDivForceM(
                alpaka::workdiv::getValidWorkDiv< TAcc >(
                    devAcc,
                    extentForceMatrix,
                    alpaka::Vec<
                        alpaka::Dim::DimInt<2u>,
                        TSize
                    >::ones(),
                    false,
                    alpaka::workdiv::GridBlockExtendSubDivRestrictions::
                    EqualExtent
                )
        );
        kernels::ForceMatrixKernel forceMatrixKernel;
        auto const forceKernelExec(
                alpaka::exec::create<TAcc>(
                    workDivForce,
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
        
        alpaka::stream::enqueue( * stream, forceKernelExec);

        /*** Execute updatePositionKernel ***/
        auto const workDivUpdatePositions(
                alpaka::workdiv::getValidWorkDiv< TAcc >(
                    devAcc,
                    extentBodies,
                    alpaka::Vec<
                        alpaka::Dim::DimInt<1u>,
                        TSize
                    >::ones(),
                    false,
                    alpaka::workdiv::GridBlockExtendSubDivRestrictions::
                    Unrestricted
                )
        );
        kernels::UpdatePositionsKernel updatePositionsKernel;
        auto const forceKernelExec(
                alpaka::exec::create<TAcc>(
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

    }

    types::Vector<NDim,TElem> getPositions(){
        if(stepFlag)
        {
            alpaka::Vec<
                alpaka::dim::DimInt<1u>,TSize>
                const extentBodies(numBodies);

            alpaka::mem::view::copy(
                * stream,
                * hostBodiesPosition,
                accBodiesPosition,
                extentBodies);
        }
        stepFlag=0;
        return alpaka::mem::view::getPtrNative(hostBodiesPosition),
    }

};

} //end namespace simulation

} //end namespace nbody
