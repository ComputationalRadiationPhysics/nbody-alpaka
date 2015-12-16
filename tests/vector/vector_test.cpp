#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE VectorTest
#include <simulation/types/vector.hpp> // Vector
#include <boost/test/unit_test.hpp>

template<
    std::size_t NDim,
    typename TElem>
using Vector = nbody::simulation::types::Vector<NDim,TElem>;

// equal operator for Vectors. Just for testing purposes.
template<
    std::size_t NDim,
    typename TElem>
bool operator==(Vector<NDim,TElem> a, Vector<NDim,TElem> b) {
    for(std::size_t i = 0; i < NDim; i++) {
        if(a[i] != b[i])
            return false;
    }
    return true;
}

BOOST_AUTO_TEST_SUITE( Math )

BOOST_AUTO_TEST_CASE( multiplicationWithScalar )
{
    Vector<3,float> a{1.0f,2.0f,3.0f},
                    aTimes2{2.0f,4.0f,6.0f};
    BOOST_CHECK( a*2 == aTimes2 );
    BOOST_CHECK( 2*a == aTimes2 );
}

BOOST_AUTO_TEST_CASE( additionAndSubtractionWithVector )
{
    Vector<3,float> a{1.0f,2.0f,3.0f},
                    b{2.0f,2.0f,2.0f},
                    c{3.0f,4.0f,5.0f};

    BOOST_CHECK( a + b == c );
    BOOST_CHECK( c - a == b );
}

BOOST_AUTO_TEST_CASE( divisionWithScalar )
{
    Vector<3,float> a{1.0f,2.0f,3.0f},
                    result{0.5f,1.0f,1.5f};

    BOOST_CHECK( a / 2 == result );
}

BOOST_AUTO_TEST_CASE( negation ) {
    Vector<3,float> a{1.0f,2.0f,3.0f},
                    result{-1.0f,-2.0f,-3.0f};

    BOOST_CHECK( -a == result );
}

BOOST_AUTO_TEST_CASE( absoluteSquared )
{
    Vector<2,float> a{3.0f,4.0f};

    BOOST_CHECK( a.absSq() == 25.0f );
}

BOOST_AUTO_TEST_SUITE_END()
