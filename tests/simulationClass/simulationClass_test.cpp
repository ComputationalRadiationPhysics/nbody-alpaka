#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SimulationClassTest
#include <iostream> // std::cout, std::endl;
#include <simulation/types/vector.hpp> //Vector
#include <simulation/simulation.hpp> // Simulation
#include <boost/test/unit_test.hpp>

using namespace nbody::simulation;

template<
    std::size_t NDim,
    typename TElem
>
std::ostream& operator<<(std::ostream & s, types::Vector<NDim, TElem> vec) {
    s << "(" << vec[0];
    for(unsigned int i = 1; i < NDim; i++) {
        s << ", " << vec[i];
    }
    s << ")";
    return s;
}

template<
    std::size_t NDim,
    typename TElem
>
bool operator==(types::Vector<NDim,TElem> const a, types::Vector<NDim,TElem> const b) {
    for(std::size_t i = 0; i < 2; i++) {
        if(a[i] != b[i])
            return false;
    }
    return true;
}

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

    float gravitationalConstant = 0.2f;

    Simulation<
        2,
        float,
        float,
        std::size_t> sim(
                bodiesPosition,
                bodiesVelocity,
                bodiesMass,
                numBodies,
                smoothnessFactor,
                gravitationalConstant);

    for(unsigned int i(0); i < 10; i++) {
        sim.step(0.1f);
    }

    types::Vector<2, float> * result = sim.getPositions();

    for(unsigned int i(0); i < 2; i++) {
        std::cout << result[i] << std::endl;
    }

    bool b = ( result[0] + result[1] == types::Vector<2,float>{1.0f,1.0f} );
    BOOST_CHECK( b );

    
}
