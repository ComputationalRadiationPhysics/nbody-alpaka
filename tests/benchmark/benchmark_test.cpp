//#define BOOST_TEST_DYN_LINK
//#define BOOST_TEST_MODULE SimulationClassTest
#include <iostream> // std::cout, std::endl;
#include <simulation/types/vector.hpp> //Vector
#include <simulation/simulation.hpp> // Simulation
//#include <boost/test/unit_test.hpp>
#include <boost/type_index.hpp>
#include <boost/timer.hpp>
#include <chrono>

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


    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    for(std::size_t i = 0; i < NSteps; i++) {
        sim.step(0.1f);
    }
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << 
        std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count() / 1000.0f << " secs" << std::endl;

    delete[] bodiesPosition;
    delete[] bodiesVelocity;
    delete[] bodiesMass;
}

int main(void) {
    for(std::size_t i = 128; i <= 2048; i*=2) {
        runTest<2,float>(i,1000);
    }

    for(std::size_t i = 128; i <= 2048; i*=2) {
        runTest<3,float>(i,1000);
    }
}
