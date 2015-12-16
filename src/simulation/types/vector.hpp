/** Vector type and functionality
 *
 * This file defines the Vector type which defines some
 * basic operations on a vector
 *
 * @file vector.cpp
 * @author Valentin Gehrke
 * @version 0.1
 * @date Wednesday, 16. December 2015 19:38 
 */

#pragma once

#include <cassert> // assert
#include <cmath> // sqrt
#include <initializer_list> // std::initializer_list
#include <algorithm> // std::copy

namespace nbody {

namespace simulation {

namespace types {

/** Vector type for N dimensions
 *
 * This class defines the Vector type
 * with basic operations for vectors
 *
 * TElem must have add, subtract and multiplication defined.
 *
 * @tparam NDim dimension of the Vector
 * @tparam TElem type of the coordinates
 */
template<
    std::size_t NDim,
    typename TElem
>
class Vector
{
    static_assert( NDim > 0 , "The Vector must atleast have one dimension." );
private:
    TElem coord[ NDim ];
public:
    /** Empty constructor
     *
     * This constructor just creates a Vector
     * that doesn't have to be initialized
     *
     */
    Vector() {}

    /** Constructor for one value
     *
     * This constructor takes one value and
     * initializes all coordinates with it
     *
     * @param initialElement Value for initialization
     */
    Vector(const TElem initialElement)
    {
        for( std::size_t i = 0; i < NDim; i++ )
        {
            this->coord[ i ] = initialElement;
        }
    }

    /** constructor for initialization with array
     *
     * This constructor initializes the coordinates with the
     * contents of an array
     *
     * @tparam template parameter
     * @param parameter
     * @return return value
     */
    Vector( const TElem initialData[ NDim ] )
    {
        for( std::size_t i = 0; i < NDim; i++ )
        {
            this->coord[ i ] = initialData[ i ];
        }
    }

    /** constructor for brace enclosed initializer list
     *
     * this constructor initializes the coordinates with a
     * brace enclosed initializer list which enables the object
     * to be initialized like the following code example
     *
     * @code
     * Vector<3,int> v{1,2,3};
     * @end-code
     *
     * @tparam template parameter
     * @param parameter
     * @return return value
     */
    Vector( std::initializer_list<TElem> initialData )
    {
        assert( initialData.size() == NDim );
        std::copy( initialData.begin(), initialData.end(), this->coord );
    }

    /** absolute of Vector squared
     *
     * This method calculates the squared absolute scalar of 
     * the vector using the Pythagorean Theorem
     *
     * @return absolute of Vector squared
     */
    auto
    absSq() const
    -> decltype( sqrt( coord[0] ) )
    {
        TElem len = 0;
        for( std::size_t index = 0; index < NDim; index++ )
        {
            len += this->coord[ index ] * this->coord[ index ];
        }
        return len;
    }

    template<
        typename TElemOther
    >
    auto
    operator=( Vector<NDim,TElemOther> const & other )
    -> Vector< NDim, TElem >&
    {
        for(std::size_t index = 0; index < NDim; index++) {
            this->coord[ index ] = (TElem) other[ index ];
        }
        return *this;
    }

    /** division by scalar
     *
     * This method divides the coordinates of the vector
     * by a scalar elementwise
     *
     * @param other divisor
     * @return vector divided by divisor
     */
    template<
        typename TFactor
    >
    auto
    operator/( TFactor const other ) const
    -> Vector<NDim, decltype( coord[0]/other ) >
    {
        Vector< NDim, decltype( coord[0]/other ) > result;
        for( std::size_t index = 0; index < NDim; index++ )
        {
            result[ index ] = this->coord[ index ] / other;
        }
        return result;
    }

    /** + operator for Vector
     *
     * This operator adds 2 vectors together
     * by adding their coordinates elementwise
     *
     * @param other Vector which is added to this vector
     * @return A new Vector containing the result
     */
    template<
        typename TElemOther
    >
    auto
    operator+ ( Vector<NDim, TElemOther> const other ) const
    -> Vector<NDim, decltype( coord[0] + other[0] ) >
    {
        Vector< NDim, decltype( coord[0] + other[0] ) > result;
        for( std::size_t i = 0 ; i < NDim ; i++ ) {
            result[ i ] = this->coord[ i ] + other[ i ];
        }
        return result;
    }

    /** += operator for Vector
     *
     * This operator adds another vector to the current
     * vector without creating a new vector
     *
     * @param other The vector to be added
     * @return The current vector after the addition
     */
    template<
        typename TElemOther
    >
    auto
    operator+= ( Vector< NDim, TElemOther > const other )
    -> Vector<NDim, TElem>&
    {
        for( std::size_t i = 0; i < NDim; i++ ) {
            this->coord[ i ] = ( TElem )( this->coord[ i ] + other[ i ] );
        }
        return *this;
    }

    /** - operator for Vector
     *
     * This operator subtract another vector from the
     * current vector resulting in a new Vector object
     *
     * @param other The vector to subtract
     * @return A new vector containing the result
     */
    template<
        typename TElemOther
    >
    auto
    operator- ( Vector<NDim,TElemOther> const other ) const
    -> Vector<
        NDim,
        decltype( coord[0] - other[0] )
    >
    {
        Vector< NDim, decltype( coord[0] - other[0] ) > result;
        for(std::size_t i = 0; i < NDim; i++)
        {
            result[ i ] = this->coord[ i ] - other[ i ];
        }
        return result;
    }

    /** negation operator for Vector
     *
     * This operator negates the current vector and
     * returns the result as a new Vector
     *
     * @return A new vector containing the result
     */
    auto
    operator- ( ) const
    -> Vector<
        NDim,
        TElem
    >
    {
        Vector< NDim , TElem > result;
        for( std::size_t i = 0; i < NDim; i++ )
        {
            result[i] = - this->coord[ i ];
        }
        return result;
    }

    /** multiplication operator for multiplication with a scalar
     *
     * This operator multiplys every coordinate with a factor
     *
     * @param factor factor used for the multiplication
     * @return A new vector containing the result
     */
    template<
        typename TFactor
    >
    auto
    operator*( TFactor const factor ) const
    -> Vector<
        NDim,
        decltype( factor * coord[0] )
    >
    {
        Vector< NDim , decltype( factor * coord[0] ) > result;
        for( std::size_t i = 0; i < NDim; i++ ) {
            result[i] = this->coord[ i ] * factor;
        }
        return result;
    }

    /** subscript operator
     *
     * This operator is used to read or write single coordinates
     * of the vector
     *
     * @param index The index of the coordinate to be read or written
     * @return A reference to the coordinate specified by index
     */
    TElem&
    operator[]( std::size_t const index )
    {
        assert( index >= 0 && index <= NDim );
        return this->coord[ index ];
    }

    /** subscript operator for constant use
     *
     * This operator is used to read a single coordinate as a constant value
     *
     * @param index The index of the coordinate to be read
     * @return The value of the coordinate specified by index
     */
    const TElem
    operator[]( std::size_t const index ) const
    {
        assert( index >= 0 && index <= NDim );
        return this->coord[ index ];
    }
};

/** reverse definition for multiplikation with a scalar
 *
 * This function defines the multiplication operator
 * for TElem * Vector<NDim, TElem> as the same as
 * Vector<NDim, TElem> * TElem
 *
 * @tparam template parameter
 * @param parameter
 * @return return value
 */
template<
    std::size_t NDim,
    typename TElem,
    typename TFactor
>
Vector<
    NDim,
    TElem
>
operator*( TFactor const factor, Vector<NDim, TElem> const vector )
{
    return vector.operator*( factor );
}

} // namespace types

} // namespace simulation

} // namespace nbody
