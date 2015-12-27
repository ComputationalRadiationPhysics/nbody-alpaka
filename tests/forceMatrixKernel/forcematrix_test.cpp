#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ForceMatrixTest
#include <iostream> // std::cout, std::endl;
#include <alpaka/alpaka.hpp>
#include <simulation/kernels/forceMatrixKernel.hpp> // ForceMatrixKernel
#include <simulation/types/vector.hpp> //Vector
#include <boost/test/unit_test.hpp>

using Vector = nbody::simulation::types::Vector<2,float>;

// equal operator for Vectors. Just for testing purposes.
bool operator==(Vector const a, Vector const b) {
    for(std::size_t i = 0; i < 2; i++) {
        if(a[i] != b[i])
            return false;
    }
    return true;
}

std::ostream& operator<<(std::ostream & s, Vector vec) {
    s << "(" << vec[0] << "," << vec[1] << ")";
    return s;
}

// Run kernel easily in tests
template<
    typename TAcc,
    typename TStream
>
auto
createForceMatrix(
        Vector * bodiesPosition,
        float * bodiesMass,
        std::size_t numBodies,
        float const gravitationalConstant,
        float const smoothnessFactor)
-> Vector*
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
        Vector,
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

    Vector* forceMatrix = new Vector[ numBodies * numBodies ];

    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        Vector,
        alpaka::dim::DimInt<2u>,
        Size>
    hostBufForceMatrix(
            forceMatrix,
            devHost,
            extentForceMatrix);

    /*** Memory Acc ***/
    auto accBufBodiesPosition(
            alpaka::mem::buf::alloc<Vector, Size>(devAcc, extentBodies));

    auto accBufBodiesMass(
            alpaka::mem::buf::alloc<float, Size>(devAcc, extentBodies));

    auto accBufForceMatrix(
            alpaka::mem::buf::alloc<Vector, Size>(devAcc, extentForceMatrix));


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
                    sizeof(Vector)
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

BOOST_AUTO_TEST_CASE( forceMatrix )
{
    Vector bodiesPosition[2];
    bodiesPosition[0] = Vector{1.0f,0.0f};
    bodiesPosition[1] = Vector{-1.0f,0.0f};

    float bodiesMass[2] = {
        2.0f,
        2.0f
    };


    Vector distance( bodiesPosition[1] - bodiesPosition[0] );

    float forceFactor( 1.0f * bodiesMass[1] * bodiesMass[0] /
            ( pow( distance.absSq(), 1.5f ) ) );

    Vector force( forceFactor * distance );

    Vector zero(0.0f);

    Vector forceMatrixResult[2*2] = {
         zero,  force,
         -force, zero
    };

    // using Acc = alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>, std::size_t>;
    // using Acc = alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<2u>, std::size_t>;
    // using Stream = alpaka::stream::StreamCpuSync;
    // using Stream = alpaka::stream::StreamCudaRtSync;

    printf("Test with CPU\n");
    Vector* forceMatrix = createForceMatrix<
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
    Vector* forceMatrix = createForceMatrix<
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

