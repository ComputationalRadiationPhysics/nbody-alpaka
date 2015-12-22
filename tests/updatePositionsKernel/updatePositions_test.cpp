#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ForceMatrixTest
#include <iostream> // std::cout, std::endl;
#include <alpaka/alpaka.hpp>
//updatePositionsKernel
#include <simulation/kernels/updatePositionsKernel.hpp> 
#include <simulation/types/vector.hpp> //Vector
#include <boost/test/unit_test.hpp>

using Vector = nbody::simulation::types::Vector<2,float>;

std::ostream& operator<<(std::ostream & s, Vector vec) {
        s << "(" << vec[0] << "," << vec[1] << ")";
            return s;
}
// equal operator for Vectors. Just for testing purposes.
bool operator==(Vector const a, Vector const b) {
    for(std::size_t i = 0; i < 2; i++) {
        if(a[i] != b[i])
            return false;
    }
    return true;
}

//Run kernel easaliy in test
auto
runUpdatePositionsKernel(
    Vector * forceMatrix,
    Vector * bodiesPosition,
    Vector * bodiesVelocity,
    float * bodiesMass,
    float  const dt,
    std::size_t numBodies
    )
->Vector *
{
    using Kernel = nbody::simulation::kernels::UpdatePositionsKernel;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<alpaka::dim::DimInt<1u>,std::size_t>;
    using Size = std::size_t;
    using Stream = alpaka::stream::StreamCpuSync;

    /*** Kernel ***/
    Kernel kernel;

    /*** Devices ***/
    auto devHost( alpaka::dev::DevManCpu::getDevByIdx(0));

    alpaka::dev::Dev<Acc> devAcc(
            alpaka::dev::DevMan<Acc>::getDevByIdx(0));

    Stream stream(devAcc);

    /*** Work extent ***/
    alpaka::Vec<
        alpaka::dim::DimInt<1u>,Size
    > const extentBodies(numBodies);

    alpaka::Vec<
        alpaka::dim::DimInt<2u>,Size
    > const extentForceMatrix( numBodies, numBodies );

    alpaka::workdiv::WorkDivMembers<
        alpaka::dim::DimInt<1u>,
        Size
    > const workDiv(
        alpaka::workdiv::getValidWorkDiv< Acc>(
            devAcc,
            extentBodies,
            alpaka::Vec<
                alpaka::dim::DimInt<1u>,
                Size
            >::ones(),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted
        )
    );

    std::cout << workDiv <<std::endl;

    /*** Memory Host **/
    //forcematrix
    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        Vector,
        alpaka::dim::DimInt<2u>,
        Size>
    hostBufForceMatrix(
        forceMatrix,
        devHost,
        extentForceMatrix);
    
    //positions
    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        Vector,
        alpaka::dim::DimInt<1u>,
        Size>
    hostBufBodiesPosition(
        bodiesPosition,
        devHost,
        extentBodies);

    //Velocities
    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        Vector,
        alpaka::dim::DimInt<1u>,
        Size>
    hostBufBodiesVelocity(
        bodiesVelocity,
        devHost,
        extentBodies);

    //Masses
    alpaka::mem::view::ViewPlainPtr<
        std::decay<decltype(devHost)>::type,
        float,
        alpaka::dim::DimInt<1u>,
        Size>
    hostBufBodiesMass(
        bodiesMass,
        devHost,
        extentBodies);

    /*** Memory Acc ***/

    auto accBufForceMatrix(
        alpaka::mem::buf::alloc<Vector,Size>(devAcc, extentForceMatrix));

    auto accBufBodiesPosition(
        alpaka::mem::buf::alloc<Vector, Size>(devAcc, extentBodies));

    auto accBufBodiesVelocity(
        alpaka::mem::buf::alloc<Vector, Size>(devAcc, extentBodies));

    auto accBufBodiesMass(
        alpaka::mem::buf::alloc<float, Size>(devAcc, extentBodies));

    /*** Memory copy ***/

    alpaka::mem::view::copy(
        stream,
        accBufForceMatrix,
        hostBufForceMatrix,
        extentForceMatrix);

    alpaka::mem::view::copy(
        stream,
        accBufBodiesPosition,
        hostBufBodiesPosition,
        extentBodies);
    
    alpaka::mem::view::copy(
        stream,
        accBufBodiesVelocity,
        hostBufBodiesVelocity,
        extentBodies);

    alpaka::mem::view::copy(
        stream,
        accBufBodiesMass,
        hostBufBodiesMass,
        extentBodies);

    /*** Execution ***/
    auto const kernelExec(
        alpaka::exec::create<Acc>(
            workDiv,
            kernel,
            alpaka::mem::view::getPtrNative(accBufForceMatrix),
            alpaka::mem::view::getPtrNative(accBufBodiesPosition),
            alpaka::mem::view::getPtrNative(accBufBodiesVelocity),
            alpaka::mem::view::getPtrNative(accBufBodiesMass),
            dt,
            numBodies
        )
    );

    // Wait for data
    alpaka::wait::wait( stream);

    alpaka::stream::enqueue( stream, kernelExec);

    // Wait for excecution
    alpaka::wait::wait(stream);

    //Memory copy back

    alpaka::mem::view::copy(
            stream,
            hostBufBodiesPosition,
            accBufBodiesPosition,
            extentBodies);

    return bodiesPosition;

}

BOOST_AUTO_TEST_CASE(updatePositions)
{
    Vector bodiesPosition[2];
    bodiesPosition[0] = Vector{1.0f,0.0f};
    bodiesPosition[1] =Vector{-1.0f,0.0f};

    float bodiesMass[2] = {
        2.0f,
        1.0f
    };

    Vector forceMatrix[4] = {
        Vector{ 0.0f, 0.0f },
        Vector{ 1.0f,0.0f},
        Vector{-1.0,0.0f},
        Vector{0.0f,0.0f}
    };

    Vector bodiesVelocity[2] = {
        Vector{0.0f,1.0f},
        Vector{0.0,-1.0F}
    };
    float dt =1.0f;
    std::size_t numBodies =2;

    Vector newPositions_test[2];
    
    for(std::size_t i=0; i<numBodies; i++)
    {
        Vector force(0.0f);
        for(std::size_t j=0; j<numBodies;j++)
        {
            force+=forceMatrix[i*numBodies+j];
        }
        newPositions_test[i]= dt*(bodiesVelocity[i]+dt*force/(2*bodiesMass[i]))+bodiesPosition[i];
    }

    Vector* newPositions_Kernel=runUpdatePositionsKernel(
                forceMatrix,
                bodiesPosition, 
                bodiesVelocity,
                bodiesMass,
                dt,
                numBodies
            );

    for (std::size_t i=0; i<numBodies;i++)
    {
        
        std::cout << newPositions_Kernel[i];
        std::cout << newPositions_test[i];
        BOOST_CHECK (newPositions_test[i]==newPositions_Kernel[i]);
    }
}

