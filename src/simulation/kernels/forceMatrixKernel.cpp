/** Kernel for the calculation of the Force Matrix
 * 
 * This file implements an Alpaka Kernel
 * for the n body simulation. The kernel
 * utilizes position and mass to calculate
 * gravitational forces between bodies.
 *
 * @file forceMatrixKernel.cpp
 * @author Valentin Gehrke
 * @version 0.1
 * @date Tuesday, 15. December 2015 11:00 
 */

#include <alpaka/alpaka.hpp>
#include "../types/vector.hpp"

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
     * 
     * @tparam TAcc Accelerator type
     * @tparam NDim Dimension of the vectors
     * @tparam TElem datatype of mass and position
     * @param acc the accelerator
     * @param bodiesPosition array of the bodies' position
     * @param bodiesMass array of the bodies' mass
     * @param forceMatrix Force Matrix as one dimensional array
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
        TVector * const forceMatrix
    )
    -> void
    {

    }
};

} // namespace kernels

} // namespace simulation

} // namespace nbody
