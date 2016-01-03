#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ForceMatrixTest
#include <iostream> // std::cout, std::endl;
#include <alpaka/alpaka.hpp>
#include <simulation/kernels/forceMatrixKernel.hpp> // ForceMatrixKernel
#include <simulation/types/vector.hpp> //Vector
#include <boost/test/unit_test.hpp>

template<
    std::size_t NDim,
    typename TElem
>
using Vector = nbody::simulation::types::Vector<NDim,TElem>;

// equal operator for Vectors. Just for testing purposes.
template<
    std::size_t NDim,
    typename TElem
>
bool operator==(Vector<NDim,TElem> const a, Vector<NDim,TElem> const b) {
    for(std::size_t i = 0; i < 2; i++) {
        if(a[i] != b[i])
            return false;
    }
    return true;
}

template<
    std::size_t NDim,
    typename TElem
>
std::ostream& operator<<(std::ostream & s, Vector<NDim, TElem> vec) {
    s << "(" << vec[0];
    for(unsigned int i = 1; i < NDim; i++) {
        s << ", " << vec[i];
    }
    s << ")";
    return s;
}

// Run kernel easily in tests
template<
    typename TAcc,
    typename TStream,
    typename TVector
>
auto
createForceMatrix(
        TVector * bodiesPosition,
        float * bodiesMass,
        std::size_t numBodies,
        float const gravitationalConstant,
        float const smoothnessFactor)
-> TVector*
{
    using Kernel = nbody::simulation::kernels::ForceMatrixKernel;
    // using Acc = alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>, std::size_t>;
    // using Acc = alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<2u>, std::size_t>;
    using Size = std::size_t;
    // using Stream = alpaka::stream::StreamCpuSync;
    // using Stream = alpaka::stream::StreamCudaRtSync;

    /*** Kernel ***/
    Kernel kernel;

    /*** Devices ***/
    auto devHost( alpaka::dev::DevManCpu::getDevByIdx( 0 ) );

    alpaka::dev::Dev<TAcc> devAcc( alpaka::dev::DevMan<TAcc>::getDevByIdx( 0 ) );

    TStream stream(devAcc);

    /*** Work extent ***/
    alpaka::Vec<
        alpaka::dim::DimInt<1u>,
        Size
    > const extentBodies( numBodies );

    alpaka::Vec<
        alpaka::dim::DimInt<2u>,
        Size
    > const extentForceMatrix( numBodies, numBodies );

    alpaka::workdiv::WorkDivMembers<
        alpaka::dim::DimInt<2u>,
        Size
    > const workDiv(
        alpaka::workdiv::getValidWorkDiv< TAcc >(
            devAcc,
            extentForceMatrix,
            alpaka::Vec<
                alpaka::dim::DimInt<2u>,
                Size
            >::ones(),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted
        )
    );

    std::cout << workDiv << std::endl;

    /*** Memory Host ***/

    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        TVector,
        alpaka::dim::DimInt<1u>,
        Size>
    hostBufBodiesPosition(
            bodiesPosition,
            devHost,
            extentBodies);

    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        float,
        alpaka::dim::DimInt<1u>,
        Size>
    hostBufBodiesMass(
            bodiesMass,
            devHost,
            extentBodies);

    TVector* forceMatrix = new TVector[ numBodies * numBodies ];

    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        TVector,
        alpaka::dim::DimInt<2u>,
        Size>
    hostBufForceMatrix(
            forceMatrix,
            devHost,
            extentForceMatrix);

    /*** Memory Acc ***/
    auto accBufBodiesPosition(
            alpaka::mem::buf::alloc<TVector, Size>(devAcc, extentBodies));

    auto accBufBodiesMass(
            alpaka::mem::buf::alloc<float, Size>(devAcc, extentBodies));

    auto accBufForceMatrix(
            alpaka::mem::buf::alloc<TVector, Size>(devAcc, extentForceMatrix));


    /*** Memory Copy ***/
    alpaka::mem::view::copy(
            stream,
            accBufBodiesPosition,
            hostBufBodiesPosition,
            extentBodies);

    alpaka::mem::view::copy(
            stream,
            accBufBodiesMass,
            hostBufBodiesMass,
            extentBodies);

    /*** Execution ***/
    auto const kernelExec(
            alpaka::exec::create<TAcc>(
                workDiv,
                kernel,
                alpaka::mem::view::getPtrNative( accBufBodiesPosition ),
                alpaka::mem::view::getPtrNative( accBufBodiesMass ),
                alpaka::mem::view::getPtrNative( accBufForceMatrix ),
                static_cast<std::size_t>(
                    alpaka::mem::view::getPitchBytes<1u>(accBufForceMatrix) /
                    sizeof(TVector)
                ),
                numBodies,
                gravitationalConstant,
                smoothnessFactor
            )
        );

    // Wait for data
    alpaka::wait::wait( stream );

    alpaka::stream::enqueue( stream, kernelExec );

    // Wait for execution
    alpaka::wait::wait( stream );

    /*** Memory Copy back ***/
    alpaka::mem::view::copy(
            stream,
            hostBufForceMatrix,
            accBufForceMatrix,
            extentForceMatrix);

    // Wait for copy operation
    alpaka::wait::wait( stream );

    return forceMatrix;
}

BOOST_AUTO_TEST_CASE( forceMatrix2D )
{
    using Vector2F = Vector<2,float>;
    Vector2F bodiesPosition[2];
    bodiesPosition[0] = Vector2F{1.0f,0.0f};
    bodiesPosition[1] = Vector2F{-1.0f,0.0f};

    float bodiesMass[2] = {
        2.0f,
        2.0f
    };


    Vector2F distance( bodiesPosition[1] - bodiesPosition[0] );

    float forceFactor( 1.0f * bodiesMass[1] * bodiesMass[0] /
            ( pow( distance.absSq(), 1.5f ) ) );

    Vector2F force( forceFactor * distance );

    Vector2F zero(0.0f);

    Vector2F forceMatrixResult[2*2] = {
         zero,  force,
         -force, zero
    };

    // using Acc = alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>, std::size_t>;
    // using Acc = alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<2u>, std::size_t>;
    // using Stream = alpaka::stream::StreamCpuSync;
    // using Stream = alpaka::stream::StreamCudaRtSync;

    printf("Test with CPU\n");
    Vector2F* forceMatrix = createForceMatrix<
        alpaka::acc::AccCpuSerial<
            alpaka::dim::DimInt<2u>,
            std::size_t >,
        alpaka::stream::StreamCpuSync
    >(
            bodiesPosition,
            bodiesMass,
            2,
            1.0f,
            0.0f);


    for( std::size_t i(0); i < 2*2; i++ )
    {
        BOOST_CHECK( forceMatrix[i] == forceMatrixResult[i] );
    }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    printf("Test with CUDA\n");
    forceMatrix = createForceMatrix<
        alpaka::acc::AccGpuCudaRt<
            alpaka::dim::DimInt<2u>,
            std::size_t>,
        alpaka::stream::StreamCudaRtSync
    >(
            bodiesPosition,
            bodiesMass,
            2,
            1.0f,
            0.0f);


    for( std::size_t i(0); i < 2*2; i++ )
    {
        BOOST_CHECK( forceMatrix[i] == forceMatrixResult[i] );
    }
#endif

}

BOOST_AUTO_TEST_CASE( forceMatrix3D )
{
    using Vector3F = Vector<3,float>;
    Vector3F bodiesPosition[2];
    bodiesPosition[0] = Vector3F{1.0f,0.0f,0.0f};
    bodiesPosition[1] = Vector3F{-1.0f,0.0f,0.0f};

    float bodiesMass[2] = {
        2.0f,
        2.0f
    };


    Vector3F distance( bodiesPosition[1] - bodiesPosition[0] );

    float forceFactor( 1.0f * bodiesMass[1] * bodiesMass[0] /
            ( pow( distance.absSq(), 1.5f ) ) );

    Vector3F force( forceFactor * distance );

    Vector3F zero(0.0f);

    Vector3F forceMatrixResult[2*2] = {
         zero,  force,
         -force, zero
    };

    // using Acc = alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>, std::size_t>;
    // using Acc = alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<2u>, std::size_t>;
    // using Stream = alpaka::stream::StreamCpuSync;
    // using Stream = alpaka::stream::StreamCudaRtSync;

    printf("Test with CPU\n");
    Vector3F* forceMatrix = createForceMatrix<
        alpaka::acc::AccCpuSerial<
            alpaka::dim::DimInt<2u>,
            std::size_t >,
        alpaka::stream::StreamCpuSync
    >(
            bodiesPosition,
            bodiesMass,
            2,
            1.0f,
            0.0f);


    for( std::size_t i(0); i < 2*2; i++ )
    {
        BOOST_CHECK( forceMatrix[i] == forceMatrixResult[i] );
    }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    printf("Test with CUDA\n");
    forceMatrix = createForceMatrix<
        alpaka::acc::AccGpuCudaRt<
            alpaka::dim::DimInt<2u>,
            std::size_t>,
        alpaka::stream::StreamCudaRtSync
    >(
            bodiesPosition,
            bodiesMass,
            2,
            1.0f,
            0.0f);


    for( std::size_t i(0); i < 2*2; i++ )
    {
        BOOST_CHECK( forceMatrix[i] == forceMatrixResult[i] );
    }
#endif

}
