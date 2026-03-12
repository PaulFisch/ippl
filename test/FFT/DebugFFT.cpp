#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include <gtest/gtest.h>
#include <random>

#include "Field/Field.h"

#include "FFT/NUFFT/ESKernel.h"
#include "Interpolation/Scatter/ScatterConfig.h"
#include "Particle/ParticleAttrib.h"

constexpr unsigned Dim = 3;
using T                = double;

//=============================================================================
// Helper Particle Class
//=============================================================================
template <class PLayout>
class TestParticles : public ippl::ParticleBase<PLayout> {
public:
    ippl::ParticleAttrib<T, typename PLayout::position_execution_space> Q;
    TestParticles(PLayout& layout)
        : ippl::ParticleBase<PLayout>(layout) {
        this->addAttribute(Q);
    }
};

//=============================================================================
// Test Environment Wrapper for a specific Execution Space
//=============================================================================
template <typename ExecSpace>
struct NUFFTEnv {
    using Mesh_t    = ippl::UniformCartesian<T, Dim>;
    using Layout_t  = ippl::FieldLayout<Dim>;
    using PLayout_t = ippl::ParticleSpatialLayout<T, Dim, Mesh_t, ExecSpace>;
    using centering_type  = typename Mesh_t::DefaultCentering;
    using Field_t      = typename ippl::Field<Kokkos::complex<T>, Dim, Mesh_t, centering_type,
                                                 ExecSpace>::uniform_type;
    using Particles_t = TestParticles<PLayout_t>;

    std::shared_ptr<Layout_t> layout;
    std::shared_ptr<Mesh_t> mesh;
    std::shared_ptr<PLayout_t> playout;
    std::shared_ptr<Particles_t> bunch;
    Field_t field;

    void init(const std::array<size_t, Dim>& gridSize, size_t numParticles, int kernelWidth) {
        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<T, Dim> hx, origin;
        for (unsigned d = 0; d < Dim; d++) {
            domains[d] = ippl::Index(gridSize[d]);
            hx[d]      = (2.0 * Kokkos::numbers::pi_v<T>) / gridSize[d];
            origin[d]  = 0.0;
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout     = std::make_shared<Layout_t>(MPI_COMM_WORLD, owned, isParallel);
        mesh       = std::make_shared<Mesh_t>(owned, hx, origin);

        // Match the ghost layer logic of NativeNUFFT
        int nghost = kernelWidth / 2 + 1;
        field.initialize(*mesh, *layout, nghost);

        playout = std::make_shared<PLayout_t>(*layout, *mesh);
        bunch   = std::make_shared<Particles_t>(*playout);
        bunch->setParticleBC(ippl::BC::PERIODIC);

        size_t nloc = numParticles / ippl::Comm->size();
        bunch->create(nloc);
    }

    void zeroField() { Kokkos::deep_copy(field.getView(), Kokkos::complex<T>(0.0, 0.0)); }
};

//=============================================================================
// Divergence Test Fixture
//=============================================================================
class NUFFTDivergenceTest : public ::testing::Test {
public:
    using HostSpace   = Kokkos::DefaultHostExecutionSpace;
    using DeviceSpace = Kokkos::DefaultExecutionSpace;

    NUFFTEnv<HostSpace> hostEnv;
    NUFFTEnv<DeviceSpace> devEnv;
    ippl::nufft::ESKernel<T> kernel;

    const std::array<size_t, Dim> gridSize = {16, 16, 16};
    const size_t totalParticles            = 4096;

    NUFFTDivergenceTest()
        : kernel(1e-6) {}  // Match typical config.tol

    void SetUp() override {
        hostEnv.init(gridSize, totalParticles, kernel.width());
        devEnv.init(gridSize, totalParticles, kernel.width());
        syncParticles();
    }

    // Generates random particles on host, then deep copies to BOTH host and device envs
    void syncParticles() {
        size_t nloc = hostEnv.bunch->getLocalNum();

        // 1. Create temporary host views
        Kokkos::View<ippl::Vector<T, Dim>*, Kokkos::HostSpace> h_R("h_R", nloc);
        Kokkos::View<T*, Kokkos::HostSpace> h_Q("h_Q", nloc);

        std::mt19937_64 eng(42 + ippl::Comm->rank());
        std::uniform_real_distribution<T> unifPos(0.0, 2.0 * Kokkos::numbers::pi_v<T>);
        std::uniform_real_distribution<T> unifCharge(0.1, 1.0);

        for (size_t i = 0; i < nloc; ++i) {
            for (unsigned d = 0; d < Dim; ++d) {
                h_R(i)[d] = unifPos(eng);
            }
            h_Q(i) = unifCharge(eng);
        }

        // 2. Copy to Host Env
        Kokkos::deep_copy(hostEnv.bunch->R.getView(), h_R);
        Kokkos::deep_copy(hostEnv.bunch->Q.getView(), h_Q);
        hostEnv.bunch->update();

        // 3. Copy to Device Env
        Kokkos::deep_copy(devEnv.bunch->R.getView(), h_R);
        Kokkos::deep_copy(devEnv.bunch->Q.getView(), h_Q);
        devEnv.bunch->update();

        Kokkos::fence();
    }

    // Helper to compare complex fields between Host and Device
    void compareFields(const std::string& stepName, double tol = 1e-10) {
        // Note: Tolerance is 1e-10 because atomics on GPU add in non-deterministic
        // orders, meaning floating-point accumulations WILL differ slightly from CPU.

        auto h_expected = hostEnv.field.getHostMirror();
        Kokkos::deep_copy(h_expected, hostEnv.field.getView());

        auto h_actual =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devEnv.field.getView());

        int nghost = hostEnv.field.getNghost();
        auto lDom  = hostEnv.field.getLayout().getLocalNDIndex();

        int extX = lDom[0].length() + 2 * nghost;
        int extY = lDom[1].length() + 2 * nghost;
        int extZ = lDom[2].length() + 2 * nghost;

        double maxDiff = 0.0;
        int errCount   = 0;

        for (int i = 0; i < extX; ++i) {
            for (int j = 0; j < extY; ++j) {
                for (int k = 0; k < extZ; ++k) {
                    auto v_exp = h_expected(i, j, k);
                    auto v_act = h_actual(i, j, k);

                    double diffR = std::abs(v_exp.real() - v_act.real());
                    double diffI = std::abs(v_exp.imag() - v_act.imag());
                    maxDiff      = std::max({maxDiff, diffR, diffI});

                    if (diffR > tol || diffI > tol) {
                        errCount++;
                        if (errCount <= 5 && ippl::Comm->rank() == 0) {
                            std::cout << "[" << stepName << "] Divergence at (" << i << "," << j
                                      << "," << k << ") Exp: " << v_exp << " Act: " << v_act
                                      << " Diff: " << std::max(diffR, diffI) << "\n";
                        }
                    }
                }
            }
        }

        if (ippl::Comm->rank() == 0) {
            std::cout << "[" << stepName << "] Max absolute difference: " << maxDiff << "\n";
        }
        EXPECT_EQ(errCount, 0) << stepName << " failed with " << errCount
                               << " diverging grid cells.";
    }
};

//=============================================================================
// Step-by-Step Tests
//=============================================================================

TEST_F(NUFFTDivergenceTest, ScatterAtomicNoSort) {
    hostEnv.zeroField();
    devEnv.zeroField();

    ippl::Interpolation::ScatterConfig<Dim> cfg;
    cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    cfg.sort   = false;

    // Run Host
    hostEnv.bunch->Q.scatter_kernel(hostEnv.field, hostEnv.bunch->R, kernel, cfg);

    // Run Device
    devEnv.bunch->Q.scatter_kernel(devEnv.field, devEnv.bunch->R, kernel, cfg);
    Kokkos::fence();

    compareFields("Scatter_Atomic_NoSort");
}

TEST_F(NUFFTDivergenceTest, ScatterAtomicWithSort) {
    hostEnv.zeroField();
    devEnv.zeroField();

    ippl::Interpolation::ScatterConfig<Dim> cfg;
    cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    cfg.sort   = true;

    // Run Host
    hostEnv.bunch->Q.scatter_kernel(hostEnv.field, hostEnv.bunch->R, kernel, cfg);

    // Run Device
    devEnv.bunch->Q.scatter_kernel(devEnv.field, devEnv.bunch->R, kernel, cfg);
    Kokkos::fence();

    compareFields("Scatter_Atomic_WithSort");
}

TEST_F(NUFFTDivergenceTest, ScatterTiled) {
    hostEnv.zeroField();
    devEnv.zeroField();

    ippl::Interpolation::ScatterConfig<Dim> cfg;
    cfg.method = ippl::Interpolation::ScatterMethod::Tiled;

    // Run Host
    hostEnv.bunch->Q.scatter_kernel(hostEnv.field, hostEnv.bunch->R, kernel, cfg);

    // Run Device
    devEnv.bunch->Q.scatter_kernel(devEnv.field, devEnv.bunch->R, kernel, cfg);
    Kokkos::fence();

    compareFields("Scatter_Tiled");
}

TEST_F(NUFFTDivergenceTest, ScatterOutputFocused) {
    hostEnv.zeroField();
    devEnv.zeroField();

    ippl::Interpolation::ScatterConfig<Dim> cfg;
    cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;

    // Run Host
    hostEnv.bunch->Q.scatter_kernel(hostEnv.field, hostEnv.bunch->R, kernel, cfg);

    // Run Device
    devEnv.bunch->Q.scatter_kernel(devEnv.field, devEnv.bunch->R, kernel, cfg);
    Kokkos::fence();

    compareFields("Scatter_OutputFocused");
}

TEST_F(NUFFTDivergenceTest, PrecomputeDeconvolutionFactors) {
    // Check if the kernel weights evaluate the exact same way on CPU and GPU
    size_t nModes = 16;
    size_t nGrid  = 32;

    Kokkos::View<Kokkos::complex<T>*, HostSpace> h_factors("h_factors", nModes);
    Kokkos::View<Kokkos::complex<T>*, DeviceSpace> d_factors("d_factors", nModes);

    ippl::nufft::compute_deconvolution_factors<HostSpace, T>(h_factors, nModes, nGrid, kernel);
    ippl::nufft::compute_deconvolution_factors<DeviceSpace, T>(d_factors, nModes, nGrid, kernel);
    Kokkos::fence();

    auto d_factors_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_factors);

    for (size_t i = 0; i < nModes; ++i) {
        EXPECT_NEAR(h_factors(i).real(), d_factors_mirror(i).real(), 1e-12);
        EXPECT_NEAR(h_factors(i).imag(), d_factors_mirror(i).imag(), 1e-12);
    }
}

TEST_F(NUFFTDivergenceTest, DistributedCPUCorrectness_Type1) {
    // 1. Setup host environment
    hostEnv.zeroField();

    // Pick a low-frequency test mode
    std::array<int, Dim> testMode = {3, 2, 1};

    // 2. Compute Exact DFT manually on CPU (Distributed)
    // Formula: F(k) = sum_j Q_j * exp(i * k * R_j)
    Kokkos::complex<T> local_dft(0.0, 0.0);

    auto h_R = hostEnv.bunch->R.getHostMirror();
    auto h_Q = hostEnv.bunch->Q.getHostMirror();
    Kokkos::deep_copy(h_R, hostEnv.bunch->R.getView());
    Kokkos::deep_copy(h_Q, hostEnv.bunch->Q.getView());

    size_t nloc = hostEnv.bunch->getLocalNum();
    for (size_t i = 0; i < nloc; ++i) {
        T phase = testMode[0] * h_R(i)[0] + testMode[1] * h_R(i)[1] + testMode[2] * h_R(i)[2];
        local_dft += h_Q(i) * Kokkos::complex<T>(std::cos(phase), -std::sin(phase));
    }

    // Reduce exact DFT across all MPI ranks
    T sendBuf[2] = {local_dft.real(), local_dft.imag()};
    T recvBuf[2] = {0.0, 0.0};
    MPI_Allreduce(sendBuf, recvBuf, 2, std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM,
                  ippl::Comm->getCommunicator());
    Kokkos::complex<T> exact_dft(recvBuf[0], recvBuf[1]);

    // 3. Run full NativeNUFFT Type 1 on CPU
    ippl::Vector<size_t, Dim> nModesVec = {gridSize[0], gridSize[1], gridSize[2]};

    typename ippl::nufft::NativeNUFFT<Dim, T, HostSpace>::Config cfg;
    cfg.tol                   = kernel.tol();
    cfg.scatter_config.method = ippl::Interpolation::ScatterMethod::Tiled;

    ippl::nufft::NativeNUFFT<Dim, T, HostSpace> cpuNufft(nModesVec, false, cfg);
    cpuNufft.initialize(*hostEnv.layout, ippl::Comm->getCommunicator());

    // Allocate Output field
    typename NUFFTEnv<HostSpace>::Field_t outputField;
    outputField.initialize(*hostEnv.mesh, *hostEnv.layout, 1);
    Kokkos::deep_copy(outputField.getView(), Kokkos::complex<T>(0.0, 0.0));

    // Run Transform
    cpuNufft.type1(hostEnv.bunch->R, hostEnv.bunch->Q, outputField, false);

    // 4. Extract mode from outputField
    // In DC-corner format, positive frequencies map directly to their index
    Kokkos::complex<T> nufft_val(0.0, 0.0);
    const auto& lDom = outputField.getLayout().getLocalNDIndex();

    // Check if this rank owns the requested test mode
    bool isOwned = true;
    for (unsigned d = 0; d < Dim; ++d) {
        if (testMode[d] < lDom[d].first() || testMode[d] > lDom[d].last()) {
            isOwned = false;
        }
    }

    if (isOwned) {
        auto h_out = outputField.getHostMirror();
        Kokkos::deep_copy(h_out, outputField.getView());

        int nghost = outputField.getNghost();
        int li     = testMode[0] - lDom[0].first() + nghost;
        int lj     = testMode[1] - lDom[1].first() + nghost;
        int lk     = testMode[2] - lDom[2].first() + nghost;

        nufft_val = h_out(li, lj, lk);
    }

    // Broadcast the result from the rank that owned it
    T sendVal[2] = {nufft_val.real(), nufft_val.imag()};
    T recvVal[2] = {0.0, 0.0};
    MPI_Allreduce(sendVal, recvVal, 2, std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM,
                  ippl::Comm->getCommunicator());
    nufft_val = Kokkos::complex<T>(recvVal[0], recvVal[1]);

    // 5. Compare Expected vs Actual
    double diffR    = std::abs(exact_dft.real() - nufft_val.real());
    double diffI    = std::abs(exact_dft.imag() - nufft_val.imag());
    double relError = std::max(diffR, diffI) / std::max(1.0, std::abs(exact_dft.real()));

    if (ippl::Comm->rank() == 0) {
        std::cout << "[Distributed CPU Check] Mode: (" << testMode[0] << "," << testMode[1] << ","
                  << testMode[2] << ")\n";
        std::cout << "  Exact DFT: " << exact_dft << "\n";
        std::cout << "  NUFFT val: " << nufft_val << "\n";
        std::cout << "  Rel Error: " << relError << "\n";
    }

    // Expect the relative error to be within tolerance * 100
    EXPECT_LT(relError, cfg.tol * 100) << "Distributed CPU NUFFT failed to match exact DFT!";
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);

        if (ippl::Comm->rank() == 0) {
            std::cout << "========================================================\n";
            std::cout << " CPU vs GPU NUFFT Component Divergence Isolation Suite\n";
            std::cout << "========================================================\n";
        }

        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}