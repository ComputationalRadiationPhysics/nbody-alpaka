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
        typename TGrav,
        typename TTime
    >
    ALPAKA_FN_ACC auto operator()(
		TAcc const & acc,
		types::Vector<NDim,TElem> const * const forceMatrix,
		types::Vector<NDim, TElem> * const bodiesPosition,
		types::Vector<NDim,TElem> * const bodiesVelocity,
        TSize const & pitchSizeForceMatrix,
        TSize const & numBodies,
        TGrav const & gravitationalConstant,
		TTime const & dt
		) const
	->void
	{
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "This kernel required 1-dimensional indices");
		
        auto const gridThreadIdx(
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
		auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::
                Thread, alpaka::Elems>(acc)[0u]);
        auto const threadFirstElemIdx(gridThreadIdx *threadElemExtent);

    //Check if out of range
        if(threadFirstElemIdx>=numBodies)
            return;

    //Sum up the "forces"(accelerations/gravitationalConstant)
        //Calculate last element+1 
        auto const threadLastElemIdxHelp(
                threadElemExtent+threadFirstElemIdx);
        auto const threadLastElemIdx(
                (threadLastElemIdxHelp > numBodies) ? 
                numBodies : threadLastElemIdxHelp);
        for(TSize p(threadFirstElemIdx); p< threadLastElemIdx;p++)
        { 
        //calculate begin of line
            types::Vector<NDim,TElem> * beginOfLine(
                    (types::Vector<NDim,TElem>*)(
                        (char*)forceMatrix +
                        p * pitchSizeForceMatrix)
                    );

            types::Vector<NDim,TElem> acceleration(static_cast<TElem>(0));
            for(std::size_t i(0); i< numBodies;i++)
            {
                acceleration+=beginOfLine[i];
            }
            acceleration=acceleration*gravitationalConstant;
            //calculate new position p=a/2*dt² +v*dt + p_0
            bodiesPosition[p]+= (acceleration/(2)*dt+
            bodiesVelocity[p])*dt;
            //calculate velocity v=a*dt
            bodiesVelocity[p]+=acceleration*dt;
		
	    }
    }
};

} // namespace kernels

} //namespace simulation

} //namespace nbody
