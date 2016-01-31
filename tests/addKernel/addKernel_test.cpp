#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ForceMatrixTest
#include <iostream> // std::cout, std::endl;
#include <alpaka/alpaka.hpp>
//updatePositionsKernel
#include <simulation/kernels/addKernel.hpp> 
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
runAddKernel(
    Vector * forceMatrix,
    std::size_t numBodies
    )
->Vector *
{
    using Kernel = nbody::simulation::kernels::AddKernel;
    using Acc = alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>,std::size_t>;
    using Size = std::size_t;
    using Stream = alpaka::stream::StreamCpuSync;

    /*** Kernel ***/
    Kernel kernel;

    /*** Devices ***/
    auto devHost( alpaka::dev::DevManCpu::getDevByIdx(0));

    alpaka::dev::Dev<Acc> devAcc(
            alpaka::dev::DevMan<Acc>::getDevByIdx(0));

    Stream stream(devAcc);

    alpaka::Vec<
        alpaka::dim::DimInt<2u>,Size
    > const extentForceMatrix( numBodies, numBodies );


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
    


    /*** Memory Acc ***/

    auto accBufForceMatrix(
        alpaka::mem::buf::alloc<Vector,Size>(devAcc, extentForceMatrix));


    /*** Memory copy ***/

    alpaka::mem::view::copy(
        stream,
        accBufForceMatrix,
        hostBufForceMatrix,
        extentForceMatrix);

    // Wait for data
    alpaka::wait::wait( stream);

    /*** Execution ***/
    /*** Work extent ***/
    unsigned int width(1);
    while( width < numBodies ) width <<= 1;
    
    do{
        width>>=1;

        alpaka::Vec<
            alpaka::dim::DimInt<2u>,Size
        > extentWorkParallelAdd(numBodies,width);

        auto const workDiv(
            alpaka::workdiv::getValidWorkDiv< Acc>(
                devAcc,
                extentWorkParallelAdd,
                alpaka::Vec<
                    alpaka::dim::DimInt<2u>,
                    Size
                >(4,1),
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted
            )
        );

        std::cout << workDiv <<std::endl;

        auto const kernelExec(
            alpaka::exec::create<Acc>(
                workDiv,
                kernel,
                alpaka::mem::view::getPtrNative(accBufForceMatrix),
                static_cast<std::size_t>(
                    alpaka::mem::view::getPitchBytes<1u>(accBufForceMatrix)
                ),
                numBodies
            )
        );

        alpaka::stream::enqueue( stream, kernelExec);

        // Wait for excecution
        alpaka::wait::wait(stream);
    }while(width>1);



    //Memory copy back

    alpaka::mem::view::copy(
            stream,
            hostBufForceMatrix,
            accBufForceMatrix,
            extentForceMatrix);

    alpaka::wait::wait( stream);

    return forceMatrix;

}

BOOST_AUTO_TEST_CASE(addKernel)
{
    
    std::size_t numBodies =9;
    Vector* forceMatrix = new Vector[numBodies*numBodies];
    for(std::size_t i(0);i<numBodies;i++){
        for(std::size_t j(0);j<numBodies;j++)
            forceMatrix[i*numBodies+j]=Vector(static_cast<float>(j));
    }
    
    Vector* newForceMatrix=runAddKernel(
                forceMatrix,
                numBodies
            );

    for (std::size_t i=0; i<numBodies;i++)
    {
        
        std::cout << newForceMatrix[i*numBodies] << std::endl;
        BOOST_CHECK (Vector(36)==newForceMatrix[i*numBodies]);
    }

    delete[] forceMatrix;
}

