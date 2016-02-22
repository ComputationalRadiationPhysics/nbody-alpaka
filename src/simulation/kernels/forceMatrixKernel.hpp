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
class ForceMatrixKernel
{
public:
    /** Force Matrix Kernel
     *
     * This kernel
     * utilizes position and mass to calculate
     * gravitational forces between bodies.
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
     * @param bodiesPosition array of the bodies' position
     * @param bodiesMass array of the bodies' mass
     * @param forceMatrix Force Matrix as one dimensional array
     * @param gravitationalConstant constant G
     * @param smoothnessFactor Smoothness Factor
     *
     */
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        std::size_t NDim,
        typename TElem,
        typename TSize,
        typename TFactor>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        types::Vector<NDim,TElem> const * const bodiesPosition,
        TElem const * const bodiesMass,
        types::Vector<NDim,TElem> * const forceMatrix,
        TSize const & pitchBytesForceMatrix,
        TSize const & numBodies,
        // Wird von UpdatePositionsKernel genutzt
        // TElem const & gravitationalConstant,
        TFactor const & smoothnessFactor ) const
    -> void
    {
        static_assert(
                alpaka::dim::Dim<TAcc>::value == 2,
                "This kernel required 2-dimensional indices");

        auto const blockSize(
                alpaka::workdiv::getWorkDiv<
                    alpaka::Block,
                    alpaka::Threads>
                    (acc));

        auto const threadElemExtent(
                alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>
                    (acc));

        char * const sharedMem(
                alpaka::block::shared::dyn::getMem< char >(acc) );

        types::Vector<NDim,TElem> * const sharedPositionInfluencing(
                ( types::Vector<NDim,TElem> * )sharedMem);

        types::Vector<NDim,TElem> * const sharedPositionInfluenced(
                sharedPositionInfluencing +
                blockSize[0u] * threadElemExtent[0u] );

        TElem * const sharedMassInfluencing(
                ( TElem * ) (
                sharedMem +
                (blockSize[0u] * threadElemExtent[0u] +
                 blockSize[1u] * threadElemExtent[1u]) *
                sizeof(types::Vector<NDim,TElem>)));

        auto const blockThreadIdx(
                alpaka::idx::getIdx< alpaka::Block, alpaka::Threads >
                    ( acc ));

        // The influencing body
        auto const gridThreadIdx(
                alpaka::idx::getIdx< alpaka::Grid,alpaka::Threads >
                    ( acc ));

        auto const gridFirstInfluenced(
                gridThreadIdx[ 0u ] * threadElemExtent[ 0u ]);

        auto const gridFirstInfluencing(
                gridThreadIdx[ 1u ] * threadElemExtent[ 1u ]);

        if(gridFirstInfluenced > numBodies || gridFirstInfluencing > numBodies)
            return;

        auto const threadExtentInfluenced(
            (gridFirstInfluenced + threadElemExtent[ 0u ] > numBodies) ?
                numBodies - gridFirstInfluenced : threadElemExtent[ 0u ] );

        auto const threadExtentInfluencing(
            (gridFirstInfluencing + threadElemExtent[ 1u ] > numBodies) ?
                numBodies - gridFirstInfluencing : threadElemExtent[ 1u ] );

        auto const blockFirstInfluenced(
                blockThreadIdx[ 0u ] * threadElemExtent[ 0u ]);

        auto const blockFirstInfluencing(
                blockThreadIdx[ 1u ] * threadElemExtent[ 0u ]);

        // first Row fills shared mem for all rows
        for( TSize threadBodyInfluencing = blockThreadIdx[ 0u ];
             threadBodyInfluencing < threadExtentInfluencing;
             threadBodyInfluencing+=blockSize[ 0u ])
        {
            sharedPositionInfluencing[ blockFirstInfluencing +
                threadBodyInfluencing ] =
                    bodiesPosition[ gridFirstInfluencing +
                        threadBodyInfluencing ];
            sharedMassInfluencing[ blockFirstInfluencing +
                threadBodyInfluencing ] =
                    bodiesMass[ gridFirstInfluencing +
                        threadBodyInfluencing ];
        }

        // first Col fills shared mem for all cols
        for( TSize threadBodyInfluenced = blockThreadIdx[ 1u ];
            threadBodyInfluenced < threadExtentInfluenced;
            threadBodyInfluenced+=blockSize[ 1u ])
        {
            sharedPositionInfluenced[ blockFirstInfluenced +
                threadBodyInfluenced ] =
                    bodiesPosition[ gridFirstInfluenced +
                        threadBodyInfluenced ];
        }

        //Sync here to ensure that shared memory has been filled
        alpaka::block::sync::syncBlockThreads(acc);

        for( TSize threadBodyInfluenced = 0;
            threadBodyInfluenced < threadExtentInfluenced;
            threadBodyInfluenced++)
        {
            // Warning: The following is necessary as the pitchBytes may
            // not be divisable by the size of Vector<NDim,TElem>.
            // e.g. Vector<3,float>'s size is 12 bytes
            types::Vector<NDim,TElem> * const matrixRow(
                (types::Vector<NDim,TElem>*)(
                    (char*)forceMatrix +
                    (gridFirstInfluenced + threadBodyInfluenced) *
                        pitchBytesForceMatrix));

            for( TSize threadBodyInfluencing = 0;
                 threadBodyInfluencing < threadExtentInfluencing;
                 threadBodyInfluencing++)
            {
                types::Vector<NDim,TElem> result;

                // position of influencing relative to influenced body
                // ( direction of force )
                types::Vector<NDim,TElem> const positionRelative(
                        sharedPositionInfluencing[ blockFirstInfluencing +
                            threadBodyInfluencing ] -
                        sharedPositionInfluenced[ blockFirstInfluenced +
                            threadBodyInfluenced ] );

                // Distance squared + smoothnessFactor
                auto const dist(
                        positionRelative.absSq() +
                        smoothnessFactor);

                auto const distCb(dist*dist*dist);

                auto const rdistCb(alpaka::math::rsqrt(acc,distCb));
                // force scalar and normalizing factor
                // force scalar * 1/(distance)
                TElem const forceFactor(
                        sharedMassInfluencing[threadBodyInfluencing] *
                        rdistCb);

                result = forceFactor * positionRelative;

                if(gridFirstInfluenced + threadBodyInfluenced ==
                        gridFirstInfluencing + threadBodyInfluencing) {
                    result = types::Vector<NDim,TElem>(
                            static_cast<TElem>(0));
                }

                // Save value
                matrixRow[ gridFirstInfluencing + threadBodyInfluencing ] =
                    result;
            }

        }
    }
};

} // namespace kernels

} // namespace simulation

} // namespace nbody

/* Alpaka Shared Memory definition */

namespace alpaka {

namespace kernel {

namespace traits {

template<
    typename TAcc>
struct BlockSharedMemDynSizeBytes<
    nbody::simulation::kernels::ForceMatrixKernel,
    TAcc>
{

    template<
        std::size_t NDim,
        typename TElem,
        typename TSize,
        typename TFactor>
    ALPAKA_FN_HOST static auto getBlockSharedMemDynSizeBytes(
        alpaka::Vec<
            alpaka::dim::Dim< TAcc >,
            size::Size< TAcc > >
            const & vblockThreadsExtents,
        alpaka::Vec<
            alpaka::dim::Dim< TAcc >,
            size::Size< TAcc > >
            const & threadElemExtent,
        nbody::simulation::types::Vector<NDim,TElem>
            const * const bodiesPosition,
        TElem const * const bodiesMass,
        nbody::simulation::types::Vector<NDim,TElem> * const forceMatrix,
        TSize const & pitchBytesForceMatrix,
        TSize const & numBodies,
        // Wird von UpdatePositionsKernel genutzt
        // TElem const & gravitationalConstant,
        TFactor const & smoothnessFactor )
    -> size::Size<TAcc>
    {
        // Ignore unused
        boost::ignore_unused(threadElemExtent);
        boost::ignore_unused(bodiesPosition);
        boost::ignore_unused(bodiesMass);
        boost::ignore_unused(forceMatrix);
        boost::ignore_unused(pitchBytesForceMatrix);
        boost::ignore_unused(numBodies);
        boost::ignore_unused(smoothnessFactor);

        return (vblockThreadsExtents[0u] * threadElemExtent[0u] +
                vblockThreadsExtents[1u] * threadElemExtent[1u]) *
            sizeof(nbody::simulation::types::Vector<NDim,TElem>) +
            vblockThreadsExtents[1u] * threadElemExtent[1u] * sizeof(TElem);
    }
};

} // namespace traits

} // namespace kernel

} // namespace alpaka
