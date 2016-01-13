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
        typename TSize >
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        types::Vector<NDim,TElem> const * const bodiesPosition,
        TElem const * const bodiesMass,
        types::Vector<NDim,TElem> * const forceMatrix,
        TSize const & pitchBytesForceMatrix,
        TSize const & numBodies,
        // Wird von UpdatePositionsKernel genutzt
        // TElem const & gravitationalConstant,
        TElem const & smoothnessFactor ) const
    -> void
    {
        static_assert(
                alpaka::dim::Dim<TAcc>::value == 2,
                "This kernel required 2-dimensional indices");

        auto const blockSizeX(
                alpaka::workdiv::getWorkDiv<
                    alpaka::Block,
                    alpaka::Threads>
                    (acc)[0u]);

        auto const blockSizeY(
                alpaka::workdiv::getWorkDiv<
                    alpaka::Block,
                    alpaka::Threads>
                    (acc)[1u]);

        char * const sharedMem(
                alpaka::block::shared::dyn::getMem< char >(acc) );

        types::Vector<NDim,TElem> * const sharedPositionInfluencing(
                ( types::Vector<NDim,TElem> * )sharedMem);

        types::Vector<NDim,TElem> * const sharedPositionInfluenced(
                sharedPositionInfluencing + blockSizeX );

        TElem * const sharedMassInfluencing(
                ( TElem * ) (
                sharedMem + (blockSizeX + blockSizeY) *
                sizeof(types::Vector<NDim,TElem>)));

        auto const blockBodyInfluencing(
                alpaka::idx::getIdx< alpaka::Block, alpaka::Threads >
                    ( acc )[ 0u ]);

        auto const blockBodyInfluenced(
                alpaka::idx::getIdx< alpaka::Block, alpaka::Threads >
                    ( acc )[ 1u ]);

        // The influencing body
        auto const indexBodyInfluencing(
                alpaka::idx::getIdx< alpaka::Grid,alpaka::Threads >
                    ( acc )[ 0u ]);
        // The influenced body
        auto const indexBodyInfluenced(
                alpaka::idx::getIdx< alpaka::Grid,alpaka::Threads >
                    ( acc )[ 1u ]);

        // first Row fills shared mem for all rows
        if(blockBodyInfluenced == 0) {
            sharedPositionInfluencing[ blockBodyInfluencing ] =
                bodiesPosition[ indexBodyInfluencing ];
            sharedMassInfluencing[ blockBodyInfluencing ] =
                bodiesMass[ indexBodyInfluencing ];
        }

        // first Col fills shared mem for all cols
        if(blockBodyInfluencing == 0) {
            sharedPositionInfluenced[ blockBodyInfluenced ] =
                bodiesPosition[ indexBodyInfluenced ];
        }

        //Sync here to ensure that shared memory has been filled
        alpaka::block::sync::syncBlockThreads(acc);

        // Warning: The following is necessary as the pitchBytes may not be
        // divisable by the size of Vector<NDim,TElem>. e.g. Vector<3,float>'s
        // size is 12 bytes
        types::Vector<NDim,TElem> * const matrixRow(
            (types::Vector<NDim,TElem>*)(
                (char*)forceMatrix +
                indexBodyInfluenced * pitchBytesForceMatrix));

        // Both exist?
        if( indexBodyInfluencing >= numBodies ||
                indexBodyInfluenced >= numBodies )
            return;
        // A body does not influence itself
        else if( indexBodyInfluenced == indexBodyInfluencing )
        {
            // forceMatrix[ matrixIdx ] =
            matrixRow[indexBodyInfluencing] =
                types::Vector<NDim,TElem>(0.0f);
        }
        // One body influences a different body
        else
        {
            // position of influencing relative to influenced body
            // ( direction of force )
            types::Vector<NDim,TElem> const positionRelative(
                    sharedPositionInfluencing[ blockBodyInfluencing ] -
                    sharedPositionInfluenced[ blockBodyInfluenced ] );

            // Distance squared + smoothnessFactor
            auto const dist(
                    positionRelative.absSq() +
                    smoothnessFactor);

            auto const distCb(dist*dist*dist);

            auto const rdistCb(alpaka::math::rsqrt(acc,distCb));
            // force scalar and normalizing factor
            // force scalar * 1/(distance)
            TElem const forceFactor(
                    //This is handled by the UpdatePositionsKernel
                    //gravitationalConstant *
                    //bodiesMass[indexBodyInfluenced] *
                    sharedMassInfluencing[blockBodyInfluencing] *
                    rdistCb);

            auto const result = forceFactor * positionRelative;

            // Save value
            matrixRow[indexBodyInfluencing] = result;
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
        typename TSize>
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
        TElem const & smoothnessFactor )
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

        return (vblockThreadsExtents[0u] + vblockThreadsExtents[1u]) *
            sizeof(nbody::simulation::types::Vector<NDim,TElem>) +
            vblockThreadsExtents[0u] * sizeof(TElem);
    }
};

} // namespace traits

} // namespace kernel

} // namespace alpaka
