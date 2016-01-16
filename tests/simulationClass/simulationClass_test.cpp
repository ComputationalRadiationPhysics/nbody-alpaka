#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SimulationClassTest
#include <iostream> // std::cout, std::endl;
#include <simulation/types/vector.hpp> //Vector
#include <simulation/simulation.hpp> // Simulation
#include <boost/test/unit_test.hpp>

using namespace nbody::simulation;

BOOST_AUTO_TEST_CASE( simulationClass )
{
    types::Vector<2,float> bodiesPosition[2] = {
        {1.0f,1.0f}, {0.0f,0.0f}
    };
    types::Vector<2,float> bodiesVelocity[2] = {
        {0.0f,0.0f}, {0.0f,0.0f}
    };

    float bodiesMass[2] = {
        1.0f, 1.0f
    };

    std::size_t numBodies = 2;

    float smoothnessFactor = 1e-8;

    Simulation<
        2,
        float,
        float,
        std::size_t> sim(
                bodiesPosition,
                bodiesVelocity,
                bodiesMass,
                numBodies,
                smoothnessFactor );

    for(unsigned int i(0); i < 10; i++) {
        sim.step(0.1f);
    }
    
}
