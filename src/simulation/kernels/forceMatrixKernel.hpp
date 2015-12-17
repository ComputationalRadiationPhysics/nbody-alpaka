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
#include <alpaka/alpaka.hpp>
#include <simulation/types/vector.hpp> // Vector
#include "forceMatrixKernel.hpp" // Class definition

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
        typename TVector = nbody::simulation::types::Vector<NDim, TElem>
    >
    ALPAKA_FN_ACC
    auto operator()(
        TAcc const acc,
        TVector const * const bodiesPosition,
        TElem const * const bodiesMass,
        TVector * const forceMatrix,
        std::size_t numBodies,
        TElem const gravitationalConstant,
        TElem const smoothnessFactor
    ) const
    -> void
    {
        static_assert(
                alpaka::dim::Dim<TAcc>::value == 2,
                "This kernel required 2-dimensional indices");

        auto const indexBodyInfluence(
                alpaka::idx::getIdx<alpaka::Grid,alpaka::Threads>(acc)[0u]);
        auto const indexBodyForce(
                alpaka::idx::getIdx<alpaka::Grid,alpaka::Threads>(acc)[1u]);

        if( indexBodyInfluence >= numBodies || indexBodyForce >= numBodies )
            return;
        else if( indexBodyForce == indexBodyInfluence )
        {
            forceMatrix[ indexBodyForce * numBodies + indexBodyInfluence ] = 
                TVector(0.0f);
        }
        else
        {
            TVector const positionRelative(
                    bodiesPosition[indexBodyInfluence] -
                    bodiesPosition[indexBodyForce]);
            
            TElem const forceFactor(
                    gravitationalConstant *
                    bodiesMass[indexBodyForce] *
                    bodiesMass[indexBodyInfluence] /
                    (
                        alpaka::math::pow(
                            acc,
                            positionRelative.absSq() +
                            alpaka::math::pow(
                                acc,
                                smoothnessFactor,
                                2.0f),
                            1.5f)
                    ));
            
            forceMatrix[ indexBodyForce * numBodies + indexBodyInfluence ] = 
                forceFactor * positionRelative;
        }
    }
};

// TODO: Remove
class TestKernel
{
    template<
        typename TAcc
    >
    void operator()(TAcc const acc, types::Vector<2,float>* const bodiesPositions) {
        // Do nothing
    }
};

} // namespace kernels

} // namespace simulation

} // namespace nbody
