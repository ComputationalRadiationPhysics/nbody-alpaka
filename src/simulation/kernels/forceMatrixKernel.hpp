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

        auto const threadElemExtent(
                alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));

        // Thread id
        auto const gridThreadIdx(
                alpaka::idx::getIdx< alpaka::Grid,alpaka::Threads >
                    ( acc ));

        for( TSize threadBodyInfluenced = 0,
            indexBodyInfluenced = gridThreadIdx[1u] *
            threadElemExtent[1u];
            threadBodyInfluenced < threadElemExtent[ 1u ] && 
            indexBodyInfluenced < numBodies;
            threadBodyInfluenced++,
            indexBodyInfluenced++)
        {
            // Warning: The following is necessary as the pitchBytes may
            // not be divisable by the size of Vector<NDim,TElem>.
            // e.g. Vector<3,float>'s size is 12 bytes
            types::Vector<NDim,TElem> * const matrixRow(
                (types::Vector<NDim,TElem>*)(
                    (char*)forceMatrix +
                    indexBodyInfluenced * pitchBytesForceMatrix));

            for( TSize threadBodyInfluencing = 0,
                 indexBodyInfluencing = gridThreadIdx[0u] *
                 threadElemExtent[0u];
                 threadBodyInfluencing < threadElemExtent[0u] &&
                 indexBodyInfluencing < numBodies;
                 threadBodyInfluencing++,
                 indexBodyInfluencing++)
            {

                // position of influencing relative to influenced body
                // ( direction of force )
                types::Vector<NDim,TElem> const positionRelative(
                        bodiesPosition[ indexBodyInfluencing ] -
                        bodiesPosition[ indexBodyInfluenced ] );

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
                        bodiesMass[indexBodyInfluencing] *
                        rdistCb);

                auto const result = forceFactor * positionRelative;

                // Save value
                matrixRow[indexBodyInfluencing] = result;
            }

            matrixRow[indexBodyInfluenced] =
                types::Vector<NDim, TElem>(static_cast<TElem>(0));
        }
    }
};

} // namespace kernels

} // namespace simulation

} // namespace nbody

