/** Kernel for the calculation of the Force Matrix
 *
 * This file implements an Alpaka Kernel
 * for the n body simulation. The kernel
 * utilizes position and mass to calculate
 * gravitational forces between bodies.
 *
 * @file forceMatrixKernel.hpp
 * @author Valentin Gehrke
 * @version 0.1
 * @date Thursday, 17. December 2015 21:31
 */

#pragma once

// alpaka, ALPAKA_FN_ACC, ALPAKA_NO_HOST_ACC_WARNING
#include <stdio.h> // printf
#include <alpaka/alpaka.hpp>
#include <simulation/types/vector.hpp> // Vector
#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

namespace nbody {

namespace simulation {

namespace kernels {

/** Class containing the Force Matrix Kernel
 *
 * This class contains the Force Matrix Kernel
 *
 */
class NBodyKernel
{
public:
    /** NBodyKernel
     *
     * This kernel
     * utilizes position and mass to calculate
     * gravitational forces between bodies.
     * And update their Positioons and Velocitys
     * It utilizes a smoothnessFactor which counteracts
     * potential singularities when distances become too
     * small and therefor forces too high
     *
     * The gravitationalConstant is
     * G = 6.674 * 10^−11 N*m^2/kg^2
     *
     * @tparam TAcc Accelerator type
     * @tparam NDim Dimension of the vectors
     * @tparam TElem datatype of mass and position
     * @param acc the accelerator
     * @param bodiesPosition array of the bodies' position and mass
     * @param bodiesVelocity  array of velocitys
     * @param gravitationalConstant constant G
     * @param smoothnessFactor Smoothness Factor
     * @param dt timestep
     */
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        std::size_t NDim,
        typename TElem,
        typename TSize,
        typename TFactor,
        typename TTime>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        types::Vector<NDim,TElem> * const bodiesPosition,
        types::Vector<NDim-1,TElem> * const bodiesVelocitys,
        TSize const & numBodies,
        TElem const & gravitationalConstant,
        TFactor const & smoothnessFactor
        TTime &dt) const
    -> void
    {
        static_assert(
                alpaka::dim::Dim<TAcc>::value == 1,
                "This kernel required 1-dimensional indices");

        auto const blockSize(
                alpaka::workdiv::getWorkDiv<
                    alpaka::Block,
                    alpaka::Threads>
                    (acc));

        char * const sharedMem(
                alpaka::block::shared::dyn::getMem< char >(acc) );

        types::Vector<NDim,TElem> * const sharedPositions(
                ( types::Vector<NDim,TElem> * )sharedMem);


        auto const blockThreadIdx(
                alpaka::idx::getIdx< alpaka::Block, alpaka::Threads >
                    ( acc ));

        auto const gridThreadIdx(
                alpaka::idx::getIdx< alpaka::Grid,alpaka::Threads >
                    ( acc ));
        
        auto const myPosition(bodiesPosition[gridThreadIdx]);
        auto const myVelocity(bodiesVelocity[gridThreadIdx]);
        types::Vector<NDim-1,TElem> myAcceleration=types::Vector<NDim-1,TElem>(static_cast<TElem>(0));
        
        // Each thread in blocks load one Position for per tile in shared memory 
        for(TSize tile=0; tile<numBodies/blockSize; tile++)
        {
            sharedPositions[blockThreadIdx]=bodiesPosition[tile*blockSize+blockThreadIdx];
            //sync threads
            alpaka::block::sync::syncBlockThreads(acc);
            //calc acceleration per Tile
            for(TSize i(0); i<blockSize; i++)
            {
                types::Vector<NDim,TElem> b(sharedPositions[i]); 
                types::Vector<NDim-1,TElem> r;
                #pragma unroll
                for(TSize j(0); j<NDim-1;j++)
                {
                    r[j]=b[j]-myPosition[j];
                }
                TElem bMass =b[NDim];
                TElem distSqr =r.absSq()+smoothnessFactor;
                TElem distSixth = distSquare*distSquar*distSquare;
                TElem invDistCube = alpaka::math::rsqrt(acc,distSixth);
                myAcceleration+=invDistCube*bMass*r;
            }
            alpaka::block::sync::syncBlockThreads(acc);
        }
        
        //Update Position and Velocity
        
        myVelocity+=dt*myAcceleration;
        #pragma unroll
        for( TSize i(0); i<NDim-1;i++)
        {
             myPosition[i]=myVelocity[i]*dt;
        }

        //writeback to global
        bodiesPosition[gridThreadId]=myPosition;
        bodiesVelocity[gridThreadId]=myVelocity;
    }
};
}// end namespace kernels
}//end namespace simulation
}//end namespace nbody

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            template<
                typename TAcc
            > struct BlockSharedMemSizeBytes<
                nbody::simulation::kernels::NBodyKernel,
                TAcc>
            {
                template<
                    std::size_t NDim,
                    typename TElem,
                    typename... TArgs>
                ALPAKA_FN_HOST static auto getBlockSharedMemDynSizeBytes(

                    alpaka::Vec<
                        alpaka::dim::Dim<TAcc>,
                        size::Size<Tacc>>
                        const & vblockThreadsExtents,

                    alpaka::Vec<
                        alpaka::dim::Dim<TAcc>,
                        size::Size<TAcc> >
                        const & threadElemExtent,
                    
                    nbody::simulation::types::Vector<NDim,TElem>
                        *const bodiesPosition,
                    TArgs && ...)
                -> std::uint32_t
                {
                    return vblockThreadsExtend[0u]*threadElemExtend[0u]
                        *sizeof(types::Vector<NDim,TElem>);
                }
            };
        }
    }
}
