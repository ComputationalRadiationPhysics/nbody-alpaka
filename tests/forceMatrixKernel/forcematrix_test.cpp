#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ForceMatrixTest
#include <alpaka/alpaka.hpp>
#include <simulation/kernels/forceMatrixKernel.hpp> // ForceMatrixKernel
#include <simulation/types/vector.hpp> //Vector
#include <boost/test/unit_test.hpp>

using Vector = nbody::simulation::types::Vector<2,float>;

auto
createForceMatrix(
        Vector const * const bodiesPosition,
        float const * const bodiesMass,
        std::size_t numBodies,
        float const gravitationalConstant,
        float smoothnessFactor)
{
   using Kernel = nbody::simulation::kernels::ForceMatrixKernel;
   using Acc = alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>, std::size_t>;
   using Size = std::size_t;
   Kernel kernel;

   auto devHost(alpaka::dev::DevManCpu::getDevByIdx(0));

   alpaka::dev::Dev<TAcc> devAcc(alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

   Stream stream(devAcc);

   alpaka::Vec<alpaka::dim::DimInt<1u>, Size> const extentBodies(numBodies);
   alpaka::Vec<alpaka::dim::DimInt<2u>, Size> const extendForceMatrix(
           numBodies,numBodies);



}

BOOST_AUTO_TEST_CASE( forceMatrix )
{

   
}
