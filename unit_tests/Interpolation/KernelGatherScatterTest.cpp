//
// Kernel-Based Gather/Scatter Unit Tests
//   Tests scatter_kernel and gather with ES kernel for NUFFT operations
//

#include "Ippl.h"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>

#include "FFT/NUFFT/ESKernel.h"
#include "Interpolation/GatherConfig.h"
#include "Interpolation/ScatterConfig.h"
#include "KernelGatherScatterTestUtils.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

//=============================================================================
// Kernel Scatter/Gather Test Fixture
//=============================================================================

template <typename>
class KernelScatterGatherTest;

template <typename T, typename ExecSpace, unsigned Dim>
class KernelScatterGatherTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    using exec_space              = ExecSpace;
    using complex_type            = Kokkos::complex<T>;
    constexpr static unsigned dim = Dim;

    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using centering_type = typename mesh_type::DefaultCentering;
    using field_type =
        typename ippl::Field<complex_type, Dim, mesh_type, centering_type, ExecSpace>::uniform_type;
    using layout_type = ippl::FieldLayout<Dim>;

    using playout_type = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;
    using bunch_type   = ippl::test::KernelInterpolationBunch<T, playout_type>;
    using kernel_type  = ippl::NUFFT::ESKernel<T>;

    std::shared_ptr<layout_type> layout;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<playout_type> playout;
    std::shared_ptr<bunch_type> bunch;

    ippl::Vector<T, Dim> origin;
    ippl::Vector<T, Dim> extent;
    ippl::Vector<T, Dim> hx;
    ippl::Vector<size_t, Dim> gridSize;

    KernelScatterGatherTest() {
        const T pi = Kokkos::numbers::pi_v<T>;
        for (unsigned d = 0; d < Dim; ++d) {
            origin[d]   = 0.0;
            extent[d]   = 2.0 * pi;
            gridSize[d] = 32;
        }
    }

    void SetUp() override {
        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        for (unsigned d = 0; d < Dim; ++d) {
            domains[d] = ippl::Index(gridSize[d]);
            hx[d]      = extent[d] / gridSize[d];
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout =
            std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel, true);  // periodic
        mesh = std::make_shared<mesh_type>(owned, hx, origin);

        playout = std::make_shared<playout_type>(*layout, *mesh);
        bunch   = std::make_shared<bunch_type>(*playout);
    }

    void runScatterKernelTest(const ippl::Interpolation::ScatterConfig& config, T tolerance_input,
                              size_t nParticles = 1000) {
        kernel_type kernel(tolerance_input);
        int nghost = kernel.width() / 2 + 1;

        field_type field(*mesh, *layout, nghost);

        // Create random particles in domain
        ippl::test::createRandomParticles(*bunch, nParticles, origin, extent);

        // Initialize particle charges with random complex values
        auto QView     = bunch->Q_scatter.getView();
        using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
        RandPool randPool(42 + ippl::Comm->rank());

        Kokkos::parallel_for(
            "init_particle_charges", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) {
                typename RandPool::generator_type gen = randPool.get_state();
                T re                                  = gen.drand(-1.0, 1.0);
                T im                                  = gen.drand(-1.0, 1.0);
                QView(i)                              = complex_type(re, im);
                randPool.free_state(gen);
            });
        Kokkos::fence();

        // Scatter to field using kernel
        field = complex_type(0.0, 0.0);
        bunch->Q_scatter.scatter_kernel(field, bunch->R, kernel, config);
        field.accumulateHalo();

        // Verify: field should have non-zero values where particles scattered
        auto fieldView   = field.getView();
        const auto& lDom = layout->getLocalNDIndex();

        T localMaxMagnitude = 0.0;

        if constexpr (Dim == 3) {
            int ni = lDom[0].length();
            int nj = lDom[1].length();
            int nk = lDom[2].length();

            Kokkos::parallel_reduce(
                "check_scatter",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>({0, 0, 0}, {ni, nj, nk}),
                KOKKOS_LAMBDA(int i, int j, int k, T& maxMag) {
                    complex_type val = fieldView(i + nghost, j + nghost, k + nghost);
                    T mag            = Kokkos::abs(val);
                    if (mag > maxMag)
                        maxMag = mag;
                },
                Kokkos::Max<T>(localMaxMagnitude));
        } else if constexpr (Dim == 2) {
            int ni = lDom[0].length();
            int nj = lDom[1].length();

            Kokkos::parallel_reduce(
                "check_scatter",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>({0, 0}, {ni, nj}),
                KOKKOS_LAMBDA(int i, int j, T& maxMag) {
                    complex_type val = fieldView(i + nghost, j + nghost);
                    T mag            = Kokkos::abs(val);
                    if (mag > maxMag)
                        maxMag = mag;
                },
                Kokkos::Max<T>(localMaxMagnitude));
        }

        Kokkos::fence();

        T globalMaxMagnitude = 0.0;
        MPI_Datatype mpiType = std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE;
        MPI_Allreduce(&localMaxMagnitude, &globalMaxMagnitude, 1, mpiType, MPI_MAX,
                      ippl::Comm->getCommunicator());

        // Verify that scatter produced non-zero field values
        if (ippl::Comm->rank() == 0) {
            EXPECT_GT(globalMaxMagnitude, 0.0);
        }
    }

    void runGatherKernelTest(const ippl::Interpolation::GatherConfig& config, T tolerance_input,
                             size_t nParticles = 1000) {
        kernel_type kernel(tolerance_input);
        int nghost = kernel.width() / 2 + 1;

        field_type field(*mesh, *layout, nghost);

        // Initialize field with random complex values
        ippl::test::initializeRandomFieldComplex(field);

        // Create random particles
        ippl::test::createRandomParticles(*bunch, nParticles, origin, extent);

        // Gather from field to particles
        auto QView = bunch->Q_gather.getView();
        Kokkos::parallel_for(
            "zero_charges", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) { QView(i) = complex_type(0.0, 0.0); });
        Kokkos::fence();

        bunch->Q_gather.gather(field, bunch->R, kernel, false, config);

        // Verify: particles should have gathered non-zero values
        T localMaxMagnitude = 0.0;

        Kokkos::parallel_reduce(
            "check_gather", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i, T& maxMag) {
                T mag = Kokkos::abs(QView(i));
                if (mag > maxMag)
                    maxMag = mag;
            },
            Kokkos::Max<T>(localMaxMagnitude));

        Kokkos::fence();

        T globalMaxMagnitude = 0.0;
        MPI_Datatype mpiType = std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE;
        MPI_Allreduce(&localMaxMagnitude, &globalMaxMagnitude, 1, mpiType, MPI_MAX,
                      ippl::Comm->getCommunicator());

        if (ippl::Comm->rank() == 0) {
            EXPECT_GT(globalMaxMagnitude, 0.0);
        }
    }

    void runAdjointnessTest(const ippl::Interpolation::ScatterConfig& scatterCfg,
                            const ippl::Interpolation::GatherConfig& gatherCfg, T tolerance_input,
                            T testTolerance = ippl::test::tolerance<T>() * 1000) {
        kernel_type kernel(tolerance_input);
        int nghost = kernel.width() / 2 + 1;

        field_type fieldScatter(*mesh, *layout, nghost);
        field_type fieldGather(*mesh, *layout, nghost);

        // Create random particles
        size_t nParticlesPerRank = 1000;
        ippl::test::createRandomParticles(*bunch, nParticlesPerRank, origin, extent);

        // Initialize field and particles with random complex values
        ippl::test::initializeRandomFieldComplex(fieldGather);

        auto QScatterView = bunch->Q_scatter.getView();
        auto QGatherView  = bunch->Q_gather.getView();

        using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
        RandPool randPool(42 + ippl::Comm->rank());

        Kokkos::parallel_for(
            "init_particle_charges", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()), KOKKOS_LAMBDA(size_t i) {
                typename RandPool::generator_type gen = randPool.get_state();
                T re                                  = gen.drand(-1.0, 1.0);
                T im                                  = gen.drand(-1.0, 1.0);
                QScatterView(i)                       = complex_type(re, im);
                QGatherView(i)                        = complex_type(0.0, 0.0);
                randPool.free_state(gen);
            });
        Kokkos::fence();

        // Perform scatter_kernel: S * q_scatter -> fieldScatter
        fieldScatter = complex_type(0.0, 0.0);
        bunch->Q_scatter.scatter_kernel(fieldScatter, bunch->R, kernel, scatterCfg);
        fieldScatter.accumulateHalo();

        // Perform gather with kernel: G * fieldGather -> q_gather
        bunch->Q_gather = complex_type(0.0, 0.0);
        bunch->Q_gather.gather(fieldGather, bunch->R, kernel, false, gatherCfg);

        // Compute left inner product: <S*q, g> = <fieldScatter, fieldGather>
        complex_type leftIP = ippl::innerProduct(fieldScatter, fieldGather);
        // Compute right inner product: <q, G*g> = <q_scatter, q_gather>
        complex_type rightIP = ippl::test::computeParticleInnerProductComplex(
            *bunch, bunch->Q_scatter, bunch->Q_gather);

        // Verify adjointness
        if (ippl::Comm->rank() == 0) {
            bool isAdjoint = ippl::test::verifyAdjointnessComplex(leftIP, rightIP, testTolerance);

            if (!isAdjoint) {
                std::cout << "Left IP:  " << leftIP << std::endl;
                std::cout << "Right IP: " << rightIP << std::endl;
                T diff     = Kokkos::abs(leftIP - rightIP);
                T relError = diff / Kokkos::abs(leftIP);
                std::cout << "Rel Error: " << relError << std::endl;
            }

            EXPECT_TRUE(isAdjoint);
        }
    }
};

TYPED_TEST_SUITE(KernelScatterGatherTest, TestParams::tests<3>);

//=============================================================================
// Scatter Kernel Tests
//=============================================================================

TYPED_TEST(KernelScatterGatherTest, ScatterKernel_Atomic_Tol1e6) {
    ippl::Interpolation::ScatterConfig config;
    config.method = ippl::Interpolation::ScatterMethod::Atomic;
    config.sort   = true;
    this->runScatterKernelTest(config, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, ScatterKernel_Tiled_Tol1e6) {
    ippl::Interpolation::ScatterConfig config;
    config.method = ippl::Interpolation::ScatterMethod::Tiled;
    config.sort   = true;
    this->runScatterKernelTest(config, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, ScatterKernel_OutputFocused_Tol1e6) {
    ippl::Interpolation::ScatterConfig config;
    config.method = ippl::Interpolation::ScatterMethod::OutputFocused;
    config.sort   = true;
    this->runScatterKernelTest(config, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, ScatterKernel_Atomic_Tol1e10) {
    ippl::Interpolation::ScatterConfig config;
    config.method = ippl::Interpolation::ScatterMethod::Atomic;
    config.sort   = true;
    this->runScatterKernelTest(config, 1e-10);
}

//=============================================================================
// Gather Kernel Tests
//=============================================================================

TYPED_TEST(KernelScatterGatherTest, GatherKernel_Atomic_Tol1e6) {
    ippl::Interpolation::GatherConfig config;
    config.method = ippl::Interpolation::GatherMethod::Atomic;
    config.sort   = true;
    this->runGatherKernelTest(config, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, GatherKernel_Tiled_Tol1e6) {
    ippl::Interpolation::GatherConfig config;
    config.method = ippl::Interpolation::GatherMethod::Tiled;
    config.sort   = true;
    this->runGatherKernelTest(config, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, GatherKernel_AtomicSort_Tol1e6) {
    ippl::Interpolation::GatherConfig config;
    config.method = ippl::Interpolation::GatherMethod::AtomicSort;
    config.sort   = true;
    this->runGatherKernelTest(config, 1e-6);
}

// TYPED_TEST(KernelScatterGatherTest, GatherKernel_Native_Tol1e6) {
//     ippl::Interpolation::GatherConfig config;
//     config.method = ippl::Interpolation::GatherMethod::Native;
//     config.sort   = true;
//     this->runGatherKernelTest(config, 1e-6);
// }

//=============================================================================
// Adjointness Tests with ES Kernel
//=============================================================================

TYPED_TEST(KernelScatterGatherTest, Adjointness_Atomic_Atomic_Tol1e6) {
    ippl::Interpolation::ScatterConfig scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    scatterCfg.sort   = true;

    ippl::Interpolation::GatherConfig gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::Atomic;
    gatherCfg.sort   = true;

    this->runAdjointnessTest(scatterCfg, gatherCfg, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, Adjointness_Atomic_AtomicSort_Tol1e6) {
    ippl::Interpolation::ScatterConfig scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    scatterCfg.sort   = true;

    ippl::Interpolation::GatherConfig gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::AtomicSort;
    gatherCfg.sort   = true;

    this->runAdjointnessTest(scatterCfg, gatherCfg, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, Adjointness_Tiled_Tiled_Tol1e6) {
    ippl::Interpolation::ScatterConfig scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Tiled;
    scatterCfg.sort   = true;

    ippl::Interpolation::GatherConfig gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::Tiled;
    gatherCfg.sort   = true;

    this->runAdjointnessTest(scatterCfg, gatherCfg, 1e-6);
}

// TYPED_TEST(KernelScatterGatherTest, Adjointness_Tiled_Native_Tol1e6) {
//     ippl::Interpolation::ScatterConfig scatterCfg;
//     scatterCfg.method = ippl::Interpolation::ScatterMethod::Tiled;
//     scatterCfg.sort   = true;
//
//     ippl::Interpolation::GatherConfig gatherCfg;
//     gatherCfg.method = ippl::Interpolation::GatherMethod::Native;
//     gatherCfg.sort   = true;
//
//     this->runAdjointnessTest(scatterCfg, gatherCfg, 1e-6);
// }

TYPED_TEST(KernelScatterGatherTest, Adjointness_OutputFocused_Atomic_Tol1e6) {
    ippl::Interpolation::ScatterConfig scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
    scatterCfg.sort   = true;

    ippl::Interpolation::GatherConfig gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::Atomic;
    gatherCfg.sort   = true;

    this->runAdjointnessTest(scatterCfg, gatherCfg, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, Adjointness_Atomic_Tiled_Tol1e6) {
    ippl::Interpolation::ScatterConfig scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    scatterCfg.sort   = true;

    ippl::Interpolation::GatherConfig gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::Tiled;
    gatherCfg.sort   = true;

    this->runAdjointnessTest(scatterCfg, gatherCfg, 1e-6);
}

TYPED_TEST(KernelScatterGatherTest, Adjointness_OutputFocused_AtomicSort_Tol1e6) {
    ippl::Interpolation::ScatterConfig scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
    scatterCfg.sort   = true;

    ippl::Interpolation::GatherConfig gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::AtomicSort;
    gatherCfg.sort   = true;

    this->runAdjointnessTest(scatterCfg, gatherCfg, 1e-6);
}

// Higher tolerance tests
TYPED_TEST(KernelScatterGatherTest, Adjointness_Atomic_Atomic_Tol1e10) {
    ippl::Interpolation::ScatterConfig scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    scatterCfg.sort   = true;

    ippl::Interpolation::GatherConfig gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::Atomic;
    gatherCfg.sort   = true;

    this->runAdjointnessTest(scatterCfg, gatherCfg, 1e-10,
                             ippl::test::tolerance<typename TestFixture::value_type>() * 10000);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
