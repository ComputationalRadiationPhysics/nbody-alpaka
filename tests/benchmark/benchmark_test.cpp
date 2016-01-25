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

template<
    typename TElem,
    std::size_t NDim,
    std::size_t NSize>
struct test_case {
    typedef TElem elem_type;
    static std::size_t const Dim = NDim;
    static std::size_t const Size = NSize;
};

typedef boost::mpl::list<
    test_case<float,2,128>,
    test_case<float,2,256>,
    test_case<float,2,512>,
    test_case<float,2,1024>,
    test_case<float,3,128>,
    test_case<float,3,256>,
    test_case<float,3,512>,
    test_case<float,3,1024>,
    test_case<double,2,128>,
    test_case<double,2,256>,
    test_case<double,2,512>,
    test_case<double,2,1024>,
    test_case<double,3,128>,
    test_case<double,3,256>,
    test_case<double,3,512>,
    test_case<double,3,1024>
> test_cases;

BOOST_AUTO_TEST_CASE_TEMPLATE( simulationBenchmark, T, test_cases )
{
    std::cout << T::Size << " bodies with "<< T::Dim << "-dimensional " <<  
        boost::typeindex::type_id<typename T::elem_type>().pretty_name() << 
        " vectors" << std::endl;

    std::size_t const NDim = T::Dim;
    std::size_t const NSize = T::Size;
    using TElem = typename T::elem_type;

    types::Vector<NDim, TElem> bodiesPosition[NSize];
    types::Vector<NDim, TElem> bodiesVelocity[NSize];
    TElem bodiesMass[NSize];

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
    for(std::size_t i = 0; i < 100; i++) {
        sim.step(0.1f);
    }
    tEnd = GET_CLOCK();

    std::cout << "Time: " << (tEnd - tStart) << " secs" << std::endl;
}

/* BOOST_AUTO_TEST_CASE( simulationClass )
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

    
} */
