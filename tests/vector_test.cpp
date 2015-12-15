#include <iostream>
#include "../src/simulation/types/vector.hpp"

using namespace nbody::simulation::types;

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

int main( int argc, char *argv[] ) {
    Vector<3,float> a{1.0f,1.0f,1.0f};
    Vector<3,float> minusA{-1.0f,-1.0f,-1.0f};
    Vector<3,float> aTimes2{2.0f,2.0f,2.0f};
    Vector<3,float> null(0.0f);
    Vector<2,float> b{3.0f,4.0f};

    if( 2*a == aTimes2 ) {
        std::cout << "Multiplication with scalar works." << std::endl;
    } else {
        std::cout << "Multiplication with scalar doesn't work." << std::endl;
    }

    if( a+a == aTimes2 ) {
        std::cout << "Addition works." << std::endl;
    } else {
        std::cout << "Addition doesn't work." << std::endl;
    }

    if( aTimes2 / 2 == a ) {
        std::cout << "Division with scalar works." << std::endl;
    } else {
        std::cout << "Division with scalar doesn't work." << std::endl;
    }

    if( a - a == null ) {
        std::cout << "Subtraction works." << std::endl;
    } else {
        std::cout << "Subtraction doesn't work." << std::endl;
    }

    if( -a == minusA ) {
        std::cout << "Negation works." << std::endl;
    } else {
        std::cout << "Negation doesn't work." << std::endl;
    }

    if( b.length() == 5.0f ) {
        std::cout << "Length works." << std::endl;
    } else {
        std::cout << "Length doesn't work." << std::endl;
    }

    if( b.normalize().length() == 1.0f ) {
        std::cout << "Normalization works." << std::endl;
    } else {
        std::cout << "Normalization doesn't work." << std::endl;
    }

}
