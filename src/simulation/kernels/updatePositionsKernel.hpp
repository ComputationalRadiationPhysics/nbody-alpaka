/**
*
*This file implements an Alpaka Kernel
*for the n body simulation.
*The kernel utlize the force between bodies, their velocity 
*and current positions to calculate their new position and velocity
*
*@file updatePositionsKernel.hpp
*@autor Vincent Ridder
*@version 0.1
*@date 19.Dezember 2015 
*/
#pragma once
// alpaka, ALPAKA_FN_ACC, ALPAKA_NO_HOST_ACC_WARNING
#include <alpaka/alpaka.hpp>
#include <simulation/types/vector.hpp> //vector

namespace nbody {

namespace simulation {

namespace kernels {

/** Cass containing the Update Position Kernel
*
* This class contains the Update Positions Kernel
*/
class UpdatePositionsKernel
{
public:
    
	ALPAKA_NO_HOST_ACC_WARNING
	template<
        typename TAcc,
        std::size_t NDim,
        typename TElem,
        typename TSize,
        typename TTime
    >
    ALPAKA_FN_ACC auto operator()(
		TAcc const & acc,
		types::Vector<NDim,TElem> const * const forceMatrix,
        TSize const & pitchSizeForceMatrix,
		types::Vector<NDim, TElem> * const bodiesPosition,
		types::Vector<NDim,TElem> * const bodiesVelocity,
		TElem const * const bodiesMass,
		TTime const & dt,
		TSize const & numBodies) const
	->void
	{
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "This kernel required 1-dimensional indices");
		
        auto const threadIdx(
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
		
        //Check if out of range
        if(threadIdx>=numBodies)
            return;

		//Sum up the forces
        types::Vector<NDim,TElem> force(static_cast<TElem>(0));
		for(std::size_t i(0); i< numBodies;i++)
        {
			force+=forceMatrix[threadIdx*pitchSizeForceMatrix+i];
		}
        //calculate new position p=a/2*dt² +v*dt + p_0
        bodiesPosition[threadIdx]+= (force/(2*bodiesMass[threadIdx])*dt+
            bodiesVelocity[threadIdx])*dt;
		//calculate velocity v=a*dt
		bodiesVelocity[threadIdx]+=force*dt/bodiesMass[threadIdx];
		
	}
};

} // namespace kernels

} //namespace simulation

} //namespace nbody
