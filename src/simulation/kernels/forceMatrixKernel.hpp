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
 * @date Tuesday, 15. December 2015 10:59
 */

#pragma once

namespace nbody {

namespace simulation {

namespace kernels {

class ForceMatrixKernel;

} // namespace kernels

} // namespace simulation

} // namespace nbody
