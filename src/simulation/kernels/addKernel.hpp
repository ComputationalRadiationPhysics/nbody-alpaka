#pragma once
#include <alpaka/alpaka.hpp>
#include <simulation/types/vector.hpp>

namespace nbody {
namespace simulation {
namespace kernels{
class AddKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        std::size_t NDim,
        typename TElem,
        typename TSize>
    ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            types::Vector<NDim,TElem> const* const forceMatrix,
            TSize const & pitchSizeForceMatrix,
            TSize const & numBodies
            ) const
    ->void
    {
        static_assert(
                alpaka::dim::Dim<TAcc>::value == 2,
                "This Kernel requierd 2-dimensional indeces");

        auto const threadIdY =alpaka::idx::getIdx<alpaka::Grid,
             alpaka::Threads>(acc)[0u];
        auto const threadIdX =alpaka::idx::getIdx<alpaka::Grid,
             alpaka::Threads>(acc)[1u];
        //num of threads in x
        auto const threadsInX= alpaka::workdiv::getWorkDiv<alpaka::Grid,alpaka::Threads>(acc)[1u];
        //num of Elmes per Thread in Y
        auto const linesInThread= alpaka::workdiv::getWorkDiv<alpaka::Thread,alpaka::Elems>(acc)[0u];
        //first line
        auto const firstLine(threadIdY*linesInThread);
        auto const lastLineHelp(firstLine+linesInThread);
        auto const lastLine((lastLineHelp > numBodies) ?
            numBodies : lastLineHelp);
        if((threadIdX+threadsInX)>=numBodies)
            return;

        for(TSize line(firstLine); line< lastLine; line++){
            types::Vector<NDim,TElem>* beginOfLine(
                    (types::Vector<NDim,TElem>*)(
                        (char*)forceMatrix +
                        line * pitchSizeForceMatrix)
                    );
            beginOfLine[threadIdX]+=beginOfLine[threadIdX+threadsInX];
        
        }
    }
};
} // kernels
} // simulation
} // nbody
