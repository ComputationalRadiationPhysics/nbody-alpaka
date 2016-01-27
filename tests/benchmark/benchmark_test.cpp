#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SimulationClassTest
#include <iostream> // std::cout, std::endl;
#include <simulation/types/vector.hpp> //Vector
#include <simulation/simulation.hpp> // Simulation
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <boost/type_index.hpp>
#include <ctime>

#define GET_CLOCK() (clock()/(float)CLOCKS_PER_SEC)

using namespace nbody::simulation;

//BOOST_AUTO_TEST_CASE_TEMPLATE( simulationBenchmark, T, test_cases )
template<
    std::size_t NDim,
    typename TElem>
void runTest(std::size_t const NSize, std::size_t const NSteps)
{
    std::cout << NSize << " bodies with "<< NDim << "-dimensional " <<  
        boost::typeindex::type_id<TElem>().pretty_name() << 
        " vectors for " << NSteps << " steps" << std::endl;

    types::Vector<NDim, TElem> * bodiesPosition =
        new types::Vector<NDim, TElem>[NSize];
    types::Vector<NDim, TElem> * bodiesVelocity =
        new types::Vector<NDim, TElem>[NSize];
    TElem * bodiesMass = new TElem[NSize];

    for(std::size_t i = 0; i < NSize; i++) {
        bodiesPosition[i] = types::Vector<NDim,TElem>( static_cast<TElem>(0) );
        bodiesVelocity[i] = types::Vector<NDim,TElem>( static_cast<TElem>(0) );
        bodiesMass[i] = static_cast<TElem>(0);
    }

    float const smoothnessFactor = 1e-8;
    float const gravitationalConstant = 1.0f;

    Simulation<
        NDim,
        TElem,
        float,
        std::size_t> sim(
                bodiesPosition,
                bodiesVelocity,
                bodiesMass,
                NSize,
                smoothnessFactor,
                gravitationalConstant);

    float tStart, tEnd;
    tStart = GET_CLOCK();
    for(std::size_t i = 0; i < NSteps; i++) {
        sim.step(0.1f);
    }
    tEnd = GET_CLOCK();

    std::cout << "Time: " << (tEnd - tStart) << " secs" << std::endl;

    delete[] bodiesPosition;
    delete[] bodiesVelocity;
    delete[] bodiesMass;
}

BOOST_AUTO_TEST_CASE( benchmark )
{
    for(int i=1 ; i <=32 ;i*=2)
    {
        runTest< 2,   float>(  i*128, 100 );
    }
    for(int i=1 ; i <=32 ;i*=2)
    {
        runTest< 3,   float>(  i*128, 100 );
    }
}
