/**
 * SingleNodeBenchmark.cpp
 *
 * Generates all single-node data for the paper:
 *   1. Kernel throughput vs tolerance for all spreading/interpolation methods
 *      (including cuFINUFFT comparison) — sweeps over rho and tolerance
 *   2. Full NUFFT timing breakdown by component — sweeps over grid sizes
 *   3. Density sweep: throughput vs particle density at fixed tolerance
 *
 * Usage:
 *   ./SingleNodeBenchmark --mode <kernels|breakdown|density|all>
 *                         [--grid <N>]           # for kernel/density mode (default: 128)
 *                         [--grids <N1,N2,...>]   # for breakdown mode (default: 64,128,192,256)
 *                         [--rhos <r1,r2,...>]    # particle densities for kernels (default: 10)
 *                         [--tols <t1,t2,...>]    # tolerances (default: 1e-2,...,1e-8)
 *                         [--warmup <N>]          # warmup iterations (default: 5)
 *                         [--runs <N>]            # timed iterations (default: 20)
 *                         [--breakdown-tol <t>]   # tolerance for breakdown (default: 1e-4)
 *                         [--breakdown-rho <r>]   # rho for breakdown, float (default: 10)
 *                         [--density-rhos <...>]  # rhos for density mode
 *                                                 # (default: 0.1,0.5,1,2,5,10,20,50)
 *                         [--density-tol <t>]     # tolerance for density mode (default: 1e-4)
 *                         [--max-mem-gb <G>]      # estimated GPU memory limit (default: 40)
 *                         [--output-prefix <s>]   # CSV prefix (default: "bench")
 *
 * Output CSVs:
 *   <prefix>_kernels.csv    — columns: grid,rho,tolerance,method,type,time_ms,throughput_mpts
 *   <prefix>_breakdown.csv  — columns: grid,rho,tolerance,type,component,time_ms,fraction
 *   <prefix>_density.csv    — columns: grid,rho,tolerance,method,type,time_ms,throughput_mpts
 */

#include "Ippl.h"
#include "Utility/IpplTimings.h"
#include "Utility/ParameterList.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef ENABLE_FINUFFT
#include "finufft_wrapper.h"
#endif

// ============================================================
// Particle container
// ============================================================
template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }
    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};

// ============================================================
// Random generators
// ============================================================
template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random_particles_with_charges {
    using view_type        = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type       = typename T::value_type;
    using view_type_scalar = typename ippl::detail::ViewType<value_type, 1>::view_type;

    view_type x;
    view_type_scalar Q;
    GeneratorPool rand_pool;
    T minU, maxU;

    generate_random_particles_with_charges(view_type x_, view_type_scalar Q_,
                                          GeneratorPool rand_pool_, T& minU_, T& maxU_)
        : x(x_), Q(Q_), rand_pool(rand_pool_), minU(minU_), maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();
        for (unsigned d = 0; d < Dim; ++d) {
            x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
        }
        Q(i) = rand_gen.drand(0.0, 1.0);
        rand_pool.free_state(rand_gen);
    }
};

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random_field {
    using view_type = typename ippl::detail::ViewType<T, Dim>::view_type;
    view_type f;
    GeneratorPool rand_pool;

    generate_random_field(view_type f_, GeneratorPool rand_pool_)
        : f(f_), rand_pool(rand_pool_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i, const size_t j, const size_t k) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();
        f(i, j, k).real() = rand_gen.drand(0.0, 1.0);
        f(i, j, k).imag() = rand_gen.drand(0.0, 1.0);
        rand_pool.free_state(rand_gen);
    }
};

// ============================================================
// Helper: parse comma-separated list
// ============================================================
template <typename T>
std::vector<T> parseList(const std::string& s) {
    std::vector<T> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        std::istringstream conv(token);
        T val;
        conv >> val;
        result.push_back(val);
    }
    return result;
}

// ============================================================
// Kernel benchmark result
// ============================================================
struct KernelResult {
    int grid;
    double rho;
    double tolerance;
    std::string method;
    std::string type;  // "1" or "2"
    double time_ms;
    double throughput_mpts;
};

// ============================================================
// Breakdown result
// ============================================================
struct BreakdownResult {
    int grid;
    double rho;
    double tolerance;
    std::string type;       // "1" or "2"
    std::string component;  // "Spreading", "FFT", "Deconvolution", "Halo", etc.
    double time_ms;
    double fraction;
};

// ============================================================
// Memory estimation: returns estimated GPU memory in GB
// Accounts for particles, oversampled field, and NUFFT workspace
// ============================================================
static double estimateMemoryGB(int grid_size, double rho, int nranks) {
    using size_type = ippl::detail::size_type;
    size_type Np   = static_cast<size_type>(double(grid_size) * grid_size * grid_size * rho);
    size_type nloc = Np / nranks;
    size_t oversampled = std::bit_ceil<size_t>(static_cast<size_t>(std::ceil(2.0 * grid_size)));

    double mem_particles_gb = double(nloc) * 32.0 / 1e9;  // 3 doubles pos + 1 double Q
    double mem_field_gb     = double(oversampled * oversampled * oversampled) * 16.0 / 1e9;
    return mem_particles_gb + mem_field_gb;
}

// ============================================================
// Type 1 benchmark: measure only spreading+FFT+deconv (full NUFFT)
// Returns time in ms
// ============================================================
template <typename FFT_type, typename Field, typename BunchT>
double benchmarkType1(FFT_type& fft, Field& field, BunchT& bunch,
                      int warmup_runs, int benchmark_runs) {
    for (int i = 0; i < warmup_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
}

// ============================================================
// Type 2 benchmark
// ============================================================
template <typename FFT_type, typename Field, typename BunchT>
double benchmarkType2(FFT_type& fft, Field& field, BunchT& bunch,
                      int warmup_runs, int benchmark_runs) {
    for (int i = 0; i < warmup_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
}

// ============================================================
// cuFINUFFT Type 1 benchmark
// ============================================================
#ifdef ENABLE_FINUFFT
template <typename PosView, typename StrengthView, typename FieldView>
double benchmarkType1FINUFFT(PosView& positions, StrengthView& strengths, FieldView& field,
                             int64_t grid_size, int64_t n_particles, double tol, int nghost,
                             int warmup_runs, int benchmark_runs) {
    using T            = double;
    using complex_type = Kokkos::complex<T>;
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

#ifdef ENABLE_GPU_NUFFT
    using finufft_complex = cuDoubleComplex;
#else
    using finufft_complex = fftw_complex;
#endif

    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> x("x", n_particles);
    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> y("y", n_particles);
    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> z("z", n_particles);
    Kokkos::View<complex_type*, Kokkos::LayoutLeft, memory_space> c("c", n_particles);
    Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space> f("f", grid_size, grid_size,
                                                                       grid_size);

    Kokkos::parallel_for("copy_positions", n_particles, KOKKOS_LAMBDA(const int64_t i) {
        x(i) = positions(i)[0];
        y(i) = positions(i)[1];
        z(i) = positions(i)[2];
        c(i) = complex_type(strengths(i), T(0));
    });
    Kokkos::fence();

    finufft_wrapper::Config<T> config;
    config.tolerance = tol;
    config.type      = 1;
#ifdef ENABLE_GPU_NUFFT
    config.gpu_method     = 3;
    config.gpu_sort       = 1;
    config.gpu_kerevalmeth = 1;
#else
    config.nthreads          = 0;
    config.spread_sort       = 2;
    config.spread_kerevalmeth = 1;
#endif

    finufft_wrapper::DirectNUFFT3D<T> nufft;
    int64_t n_modes[3] = {grid_size, grid_size, grid_size};
    nufft.make_plan(n_modes, config);
    nufft.set_points(n_particles, x.data(), y.data(), z.data());

    auto run_transform = [&]() {
        Kokkos::parallel_for("copy_strengths", n_particles, KOKKOS_LAMBDA(const int64_t i) {
            c(i) = complex_type(strengths(i), T(0));
        });
        Kokkos::fence();
        nufft.execute(reinterpret_cast<finufft_complex*>(c.data()),
                      reinterpret_cast<finufft_complex*>(f.data()));
        Kokkos::fence();
    };

    for (int i = 0; i < warmup_runs; ++i) run_transform();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) run_transform();
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
}

template <typename PosView, typename OutputView, typename FieldView>
double benchmarkType2FINUFFT(PosView& positions, OutputView& output, FieldView& field,
                             int64_t grid_size, int64_t n_particles, double tol, int nghost,
                             int warmup_runs, int benchmark_runs) {
    using T            = double;
    using complex_type = Kokkos::complex<T>;
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

#ifdef ENABLE_GPU_NUFFT
    using finufft_complex = cuDoubleComplex;
#else
    using finufft_complex = fftw_complex;
#endif

    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> x("x", n_particles);
    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> y("y", n_particles);
    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> z("z", n_particles);
    Kokkos::View<complex_type*, Kokkos::LayoutLeft, memory_space> c("c", n_particles);
    Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space> f("f", grid_size, grid_size,
                                                                       grid_size);

    Kokkos::parallel_for("copy_positions", n_particles, KOKKOS_LAMBDA(const int64_t i) {
        x(i) = positions(i)[0];
        y(i) = positions(i)[1];
        z(i) = positions(i)[2];
    });
    Kokkos::fence();

    finufft_wrapper::Config<T> config;
    config.tolerance = tol;
    config.type      = 2;
#ifdef ENABLE_GPU_NUFFT
    config.gpu_method     = 1;  // 1=non-uniform pts driven, 2=subproblem (3 is invalid)
    config.gpu_sort       = 1;
    config.gpu_kerevalmeth = 1;
#else
    config.nthreads          = 0;
    config.spread_sort       = 2;
    config.spread_kerevalmeth = 1;
#endif

    finufft_wrapper::DirectNUFFT3D<T> nufft;
    int64_t n_modes[3] = {grid_size, grid_size, grid_size};
    nufft.make_plan(n_modes, config);
    nufft.set_points(n_particles, x.data(), y.data(), z.data());

    auto run_transform = [&]() {
        const int64_t nx = grid_size, ny = grid_size, nz = grid_size;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for("copy_from_field", mdrange_type({0, 0, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(const int64_t i, const int64_t j, const int64_t k) {
                f(i, j, k) = field(i + nghost, j + nghost, k + nghost);
            });
        Kokkos::fence();
        nufft.execute(reinterpret_cast<finufft_complex*>(c.data()),
                      reinterpret_cast<finufft_complex*>(f.data()));
        Kokkos::fence();
    };

    for (int i = 0; i < warmup_runs; ++i) run_transform();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) run_transform();
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
}
#endif  // ENABLE_FINUFFT

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t      = ippl::UniformCartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        using size_type   = ippl::detail::size_type;

        const double pi = std::acos(-1.0);

        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;
        typedef ippl::Vector<double, 3> Vector_t;
        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type
            field_type;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type real_field_type;
        typedef ippl::FFT<ippl::NUFFTransform, real_field_type> FFT_type;

        // Prevent TileSizeCache from overriding the NUFFT's internal scatter
        // configs.  Without this, a CSV from a previous TileSweep run silently
        // replaces tile/team/osub/z_batches, corrupting benchmark results.
        ippl::Interpolation::TileSizeCache::instance().clear();

        // ============================================================
        // Parse command-line arguments
        // ============================================================
        std::string mode          = "all";
        int kernel_grid           = 128;
        std::string grids_str     = "64,128,192,256";
        std::string rhos_str      = "10";
        std::string tols_str      = "1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8";
        int warmup_runs           = 5;
        int benchmark_runs        = 20;
        double breakdown_tol      = 1e-4;
        double breakdown_rho      = 10;
        std::string density_rhos_str = "0.1,0.5,1,2,5,10,20,50";
        double density_tol        = 1e-4;
        double max_mem_gb         = 40.0;
        std::string output_prefix = "bench";

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--mode" && i + 1 < argc)            mode = argv[++i];
            else if (arg == "--grid" && i + 1 < argc)       kernel_grid = std::atoi(argv[++i]);
            else if (arg == "--grids" && i + 1 < argc)      grids_str = argv[++i];
            else if (arg == "--rhos" && i + 1 < argc)       rhos_str = argv[++i];
            else if (arg == "--tols" && i + 1 < argc)       tols_str = argv[++i];
            else if (arg == "--warmup" && i + 1 < argc)     warmup_runs = std::atoi(argv[++i]);
            else if (arg == "--runs" && i + 1 < argc)       benchmark_runs = std::atoi(argv[++i]);
            else if (arg == "--breakdown-tol" && i + 1 < argc) breakdown_tol = std::stod(argv[++i]);
            else if (arg == "--breakdown-rho" && i + 1 < argc) breakdown_rho = std::stod(argv[++i]);
            else if (arg == "--density-rhos" && i + 1 < argc)  density_rhos_str = argv[++i];
            else if (arg == "--density-tol" && i + 1 < argc)   density_tol = std::stod(argv[++i]);
            else if (arg == "--max-mem-gb" && i + 1 < argc)    max_mem_gb = std::stod(argv[++i]);
            else if (arg == "--output-prefix" && i + 1 < argc) output_prefix = argv[++i];
        }

        auto rhos = parseList<double>(rhos_str);
        auto tols = parseList<double>(tols_str);
        auto breakdown_grids = parseList<int>(grids_str);
        auto density_rhos = parseList<double>(density_rhos_str);

        bool do_kernels   = (mode == "kernels" || mode == "all");
        bool do_breakdown = (mode == "breakdown" || mode == "all");
        bool do_density   = (mode == "density");

        const int rank = ippl::Comm->rank();
        const int nranks = ippl::Comm->size();

        if (rank == 0) {
            std::cout << "========================================\n";
            std::cout << "Single-Node NUFFT Benchmark\n";
            std::cout << "========================================\n";
            std::cout << "Mode:            " << mode << "\n";
            std::cout << "Kernel grid:     " << kernel_grid << "^3\n";
            std::cout << "Breakdown grids: " << grids_str << "\n";
            std::cout << "Rhos (kernels):  " << rhos_str << "\n";
            std::cout << "Tolerances:      " << tols_str << "\n";
            std::cout << "Warmup:          " << warmup_runs << "\n";
            std::cout << "Runs:            " << benchmark_runs << "\n";
            if (do_density) {
                std::cout << "Density rhos:    " << density_rhos_str << "\n";
                std::cout << "Density tol:     " << std::scientific << density_tol << "\n";
            }
            std::cout << "Max GPU memory:  " << std::fixed << std::setprecision(0)
                      << max_mem_gb << " GB\n";
            std::cout << "========================================\n\n";
        }

        // ============================================================
        // Spreading/interpolation method descriptors
        // ============================================================
        struct SpreadConfig {
            std::string name;
            std::string spread_method;
            bool lock_method;
        };
        std::vector<SpreadConfig> spread_methods = {
            {"Atomic",        "atomic",          true },
            {"Tiled",         "tiled",           true},
            {"Grid-Parallel", "output_focused",  true},
        };

        struct GatherConfig {
            std::string name;
            std::string gather_method;
        };
        std::vector<GatherConfig> gather_methods = {
            {"Direct", "atomic"},
            {"Sorted", "atomic_sort"},
        };

        // Collect results
        std::vector<KernelResult> kernel_results;
        std::vector<BreakdownResult> breakdown_results;
        std::vector<KernelResult> density_results;

        // ============================================================
        // Part 1: Kernel throughput vs tolerance
        // ============================================================
        if (do_kernels) {
            if (rank == 0) {
                std::cout << "============================================================\n";
                std::cout << "KERNEL THROUGHPUT vs TOLERANCE (grid=" << kernel_grid << "^3)\n";
                std::cout << "============================================================\n\n";
            }

            const int gs = kernel_grid;

            for (double rho : rhos) {
                // Memory estimation: skip if estimated base memory exceeds 40%
                // of max GPU memory (leaving headroom for NUFFT workspace/FFT plans)
                double mem_est_gb = estimateMemoryGB(gs, rho, nranks);
                if (mem_est_gb > max_mem_gb * 0.4) {
                    if (rank == 0) {
                        std::cout << "--- rho = " << rho << " ---\n";
                        std::cout << "  [SKIP] Estimated memory " << std::fixed
                                  << std::setprecision(1) << mem_est_gb
                                  << " GB exceeds limit (" << max_mem_gb * 0.4
                                  << " GB = 40% of " << max_mem_gb << " GB)\n\n";
                    }
                    continue;
                }

                size_type Np   = static_cast<size_type>(double(gs) * gs * gs * rho);
                size_type nloc = Np / nranks;

                if (rank == 0) {
                    std::cout << "--- rho = " << rho << " ---\n";
                    std::cout << "  Np=" << Np << "  est. memory: " << std::fixed
                              << std::setprecision(1) << mem_est_gb << " GB\n";
                }

                try {

                // Setup mesh & layout (single-rank, no domain decomposition)
                ippl::Vector<int, dim> pt = {gs, gs, gs};
                auto owned = ippl::NDIndex<dim>(ippl::Index(pt[0]), ippl::Index(pt[1]),
                                         ippl::Index(pt[2]));

                std::array<bool, dim> isParallel;
                isParallel.fill(false);
                ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

                Vector_t minU = {0, 0, 0};
                Vector_t maxU = {2*pi, 2*pi, 2*pi};
                std::array<double, dim> dx = {
                    (maxU[0] - minU[0]) / double(pt[0]),
                    (maxU[1] - minU[1]) / double(pt[1]),
                    (maxU[2] - minU[2]) / double(pt[2]),
                };
                Vector_t hx     = {dx[0], dx[1], dx[2]};
                Vector_t origin = {minU[0], minU[1], minU[2]};
                Mesh_t mesh(owned, hx, origin);

                playout_type pl(layout, mesh);

                // Create particles
                bunch_type bunch(pl);
                bunch.setParticleBC(ippl::BC::PERIODIC);
                bunch.create(nloc);

                field_type field(mesh, layout);

                Kokkos::Random_XorShift64_Pool<> rand_pool64(size_type(42 + int(rho * 1000)));
                Kokkos::parallel_for(
                    nloc,
                    generate_random_particles_with_charges<Vector_t,
                                                          Kokkos::Random_XorShift64_Pool<>, dim>(
                        bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));
                Kokkos::fence();

                // Fill field for Type 2
                const int nghost = field.getNghost();
                using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
                auto fview = field.getView();
                Kokkos::parallel_for(
                    mdrange_type({nghost, nghost, nghost},
                                 {fview.extent(0) - nghost, fview.extent(1) - nghost,
                                  fview.extent(2) - nghost}),
                    generate_random_field<Kokkos::complex<double>,
                                         Kokkos::Random_XorShift64_Pool<>, dim>(
                        field.getView(), rand_pool64));
                Kokkos::fence();

                for (double tol : tols) {
                    // ---- Type 1: Spreading methods ----
                    for (const auto& sc : spread_methods) {
                        try {
                            ippl::ParameterList fftParams;
                            fftParams.add("tolerance", tol);
                            fftParams.add("use_finufft_defaults", false);
                            fftParams.add("use_kokkos_nufft", false);
                            fftParams.add("spread_method", sc.spread_method);
                            fftParams.add("lock_method", true);

                            auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                            double t = benchmarkType1(*fft, field, bunch,
                                                      warmup_runs, benchmark_runs);
                            double throughput = Np / t * 1000.0 / 1e6;

                            kernel_results.push_back(
                                {gs, rho, tol, sc.name, "1", t, throughput});

                            if (rank == 0) {
                                std::cout << "  Type1 " << std::setw(15) << sc.name
                                          << " tol=" << std::scientific << std::setprecision(0)
                                          << tol << "  " << std::fixed << std::setprecision(1)
                                          << throughput << " Mpts/s  (" << std::setprecision(3)
                                          << t << " ms)\n";
                            }
                        } catch (const std::exception& e) {
                            if (rank == 0)
                                std::cerr << "  FAILED: Type1 " << sc.name
                                          << " tol=" << tol << ": " << e.what() << "\n";
                        }
                    }

                    // ---- Type 2: Interpolation methods ----
                    for (const auto& gc : gather_methods) {
                        try {
                            ippl::ParameterList fftParams;
                            fftParams.add("tolerance", tol);
                            fftParams.add("use_finufft_defaults", false);
                            fftParams.add("use_kokkos_nufft", false);
                            fftParams.add("gather_method", gc.gather_method);

                            auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                            double t = benchmarkType2(*fft, field, bunch,
                                                      warmup_runs, benchmark_runs);
                            double throughput = Np / t * 1000.0 / 1e6;

                            kernel_results.push_back(
                                {gs, rho, tol, gc.name, "2", t, throughput});

                            if (rank == 0) {
                                std::cout << "  Type2 " << std::setw(15) << gc.name
                                          << " tol=" << std::scientific << std::setprecision(0)
                                          << tol << "  " << std::fixed << std::setprecision(1)
                                          << throughput << " Mpts/s  (" << std::setprecision(3)
                                          << t << " ms)\n";
                            }
                        } catch (const std::exception& e) {
                            if (rank == 0)
                                std::cerr << "  FAILED: Type2 " << gc.name
                                          << " tol=" << tol << ": " << e.what() << "\n";
                        }
                    }

                    // ---- cuFINUFFT / FINUFFT comparison ----
#ifdef ENABLE_FINUFFT
                    {
                        auto Rview = bunch.R.getView();
                        auto Qview = bunch.Q.getView();
                        auto fview_local = field.getView();

#ifdef ENABLE_GPU_NUFFT
                        std::string finufft_label = "cuFINUFFT";
#else
                        std::string finufft_label = "FINUFFT";
#endif
                        // Type 1
                        try {
                            double t = benchmarkType1FINUFFT(
                                Rview, Qview, fview_local, gs, nloc, tol, nghost,
                                warmup_runs, benchmark_runs);
                            double throughput = Np / t * 1000.0 / 1e6;

                            kernel_results.push_back(
                                {gs, rho, tol, finufft_label, "1", t, throughput});

                            if (rank == 0) {
                                std::cout << "  Type1 " << std::setw(15) << finufft_label
                                          << " tol=" << std::scientific << std::setprecision(0)
                                          << tol << "  " << std::fixed << std::setprecision(1)
                                          << throughput << " Mpts/s  (" << std::setprecision(3)
                                          << t << " ms)\n";
                            }
                        } catch (const std::exception& e) {
                            if (rank == 0)
                                std::cerr << "  FAILED: " << finufft_label
                                          << " Type1: " << e.what() << "\n";
                        }

                        // Type 2
                        try {
                            double t = benchmarkType2FINUFFT(
                                Rview, Qview, fview_local, gs, nloc, tol, nghost,
                                warmup_runs, benchmark_runs);
                            double throughput = Np / t * 1000.0 / 1e6;

                            kernel_results.push_back(
                                {gs, rho, tol, finufft_label, "2", t, throughput});

                            if (rank == 0) {
                                std::cout << "  Type2 " << std::setw(15) << finufft_label
                                          << " tol=" << std::scientific << std::setprecision(0)
                                          << tol << "  " << std::fixed << std::setprecision(1)
                                          << throughput << " Mpts/s  (" << std::setprecision(3)
                                          << t << " ms)\n";
                            }
                        } catch (const std::exception& e) {
                            if (rank == 0)
                                std::cerr << "  FAILED: " << finufft_label
                                          << " Type2: " << e.what() << "\n";
                        }
                    }
#endif  // ENABLE_FINUFFT

                }  // tol loop
                if (rank == 0) std::cout << "\n";

                } catch (const std::exception& e) {
                    if (rank == 0)
                        std::cerr << "  [OOM] Skipping rho=" << rho
                                  << " grid=" << gs << ": " << e.what() << "\n\n";
                }
            }  // rho loop
        }  // do_kernels

        // ============================================================
        // Part 2: NUFFT Timing Breakdown
        // ============================================================
        if (do_breakdown) {
            if (rank == 0) {
                std::cout << "============================================================\n";
                std::cout << "NUFFT TIMING BREAKDOWN (rho=" << breakdown_rho
                          << ", tol=" << std::scientific << std::setprecision(0)
                          << breakdown_tol << std::fixed << ")\n";
                std::cout << "============================================================\n\n";
            }

            for (int gs : breakdown_grids) {
                // Memory check for breakdown
                double mem_est_gb = estimateMemoryGB(gs, breakdown_rho, nranks);
                if (mem_est_gb > max_mem_gb * 0.4) {
                    if (rank == 0) {
                        std::cout << "--- grid = " << gs << "^3 ---\n";
                        std::cout << "  [SKIP] Estimated memory " << std::fixed
                                  << std::setprecision(1) << mem_est_gb
                                  << " GB exceeds limit\n\n";
                    }
                    continue;
                }

                if (rank == 0) {
                    std::cout << "--- grid = " << gs << "^3 ---\n";
                }

                ippl::Vector<int, dim> pt = {gs, gs, gs};
                auto owned = ippl::NDIndex<dim>(ippl::Index(pt[0]), ippl::Index(pt[1]),
                                         ippl::Index(pt[2]));

                std::array<bool, dim> isParallel;
                isParallel.fill(false);
                ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

                Vector_t minU = {0, 0, 0};
                Vector_t maxU = {2*pi, 2*pi, 2*pi};
                std::array<double, dim> dx = {
                    (maxU[0] - minU[0]) / double(pt[0]),
                    (maxU[1] - minU[1]) / double(pt[1]),
                    (maxU[2] - minU[2]) / double(pt[2]),
                };
                Vector_t hx     = {dx[0], dx[1], dx[2]};
                Vector_t origin = {minU[0], minU[1], minU[2]};
                Mesh_t mesh(owned, hx, origin);

                playout_type pl(layout, mesh);

                size_type Np   = static_cast<size_type>(double(gs) * gs * gs * breakdown_rho);
                size_type nloc = Np / nranks;

                bunch_type bunch(pl);
                bunch.setParticleBC(ippl::BC::PERIODIC);
                bunch.create(nloc);

                field_type field(mesh, layout);

                Kokkos::Random_XorShift64_Pool<> rand_pool64(size_type(123));
                Kokkos::parallel_for(
                    nloc,
                    generate_random_particles_with_charges<Vector_t,
                                                          Kokkos::Random_XorShift64_Pool<>, dim>(
                        bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));
                Kokkos::fence();

                const int nghost = field.getNghost();
                using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
                auto fview = field.getView();
                Kokkos::parallel_for(
                    mdrange_type({nghost, nghost, nghost},
                                 {fview.extent(0) - nghost, fview.extent(1) - nghost,
                                  fview.extent(2) - nghost}),
                    generate_random_field<Kokkos::complex<double>,
                                         Kokkos::Random_XorShift64_Pool<>, dim>(
                        field.getView(), rand_pool64));
                Kokkos::fence();

                // Use best methods: Grid-Parallel spreading, Sorted interpolation
                ippl::ParameterList fftParams;
                fftParams.add("tolerance", breakdown_tol);
                fftParams.add("use_finufft_defaults", false);
                fftParams.add("use_kokkos_nufft", false);
                fftParams.add("spread_method", "output_focused");
                fftParams.add("gather_method", "atomic_sort");
                fftParams.add("lock_method", true);

                auto fft_type1 = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                auto fft_type2 = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);

                // Warmup
                for (int i = 0; i < warmup_runs; ++i) {
                    fft_type1->transform(bunch.R, bunch.Q, field);
                    Kokkos::fence();
                    fft_type2->transform(bunch.R, bunch.Q, field);
                    Kokkos::fence();
                }

                // Reset timers after warmup
                IpplTimings::resetAllTimers();

                // Timed runs
                for (int i = 0; i < benchmark_runs; ++i) {
                    fft_type1->transform(bunch.R, bunch.Q, field);
                    Kokkos::fence();
                    fft_type2->transform(bunch.R, bunch.Q, field);
                    Kokkos::fence();
                }

                // Extract component timings
                // Type 1 components
                struct TimerEntry {
                    std::string timer_name;
                    std::string label;
                    std::string type;
                };
                std::vector<TimerEntry> timers = {
                    {"scatterTimerNUFFT1",     "Spreading",      "1"},
                    {"accumulateHaloNUFFT1",   "Halo",           "1"},
                    {"FFTNUFFT1",              "FFT",            "1"},
                    {"deconvolutionNUFFT1",    "Deconvolution",  "1"},
                    {"PrecorrectionNUFFT2",    "Precorrection",  "2"},
                    {"FFTNUFFT2",              "FFT",            "2"},
                    {"FillHaloNUFFT2",         "Halo",           "2"},
                    {"GatherNUFFT2",           "Interpolation",  "2"},
                };

                // Compute total time per type for fraction calculation
                std::map<std::string, double> type_totals;  // type -> total ms

                for (const auto& te : timers) {
                    const auto& measurements = IpplTimings::getMeasurements(te.timer_name);
                    if (!measurements.empty()) {
                        double sum = 0.0;
                        for (double m : measurements) sum += m;
                        double mean_ms = (sum / measurements.size()) * 1000.0;
                        type_totals[te.type] += mean_ms;
                    }
                }

                for (const auto& te : timers) {
                    const auto& measurements = IpplTimings::getMeasurements(te.timer_name);
                    if (!measurements.empty()) {
                        double sum = 0.0;
                        for (double m : measurements) sum += m;
                        double mean_ms  = (sum / measurements.size()) * 1000.0;
                        double fraction = (type_totals[te.type] > 0)
                                              ? mean_ms / type_totals[te.type]
                                              : 0.0;

                        breakdown_results.push_back(
                            {gs, breakdown_rho, breakdown_tol, te.type, te.label,
                             mean_ms, fraction});

                        if (rank == 0) {
                            std::cout << "  Type" << te.type << " " << std::setw(16)
                                      << te.label << ": " << std::fixed << std::setprecision(3)
                                      << mean_ms << " ms  ("
                                      << std::setprecision(1) << fraction * 100.0 << "%)\n";
                        }
                    }
                }

                if (rank == 0) std::cout << "\n";
            }  // grid loop
        }  // do_breakdown

        // ============================================================
        // Part 3: Density sweep (throughput vs rho at fixed tolerance)
        // ============================================================
        if (do_density) {
            if (rank == 0) {
                std::cout << "============================================================\n";
                std::cout << "DENSITY SWEEP (grid=" << kernel_grid
                          << "^3, tol=" << std::scientific << std::setprecision(0)
                          << density_tol << std::fixed << ")\n";
                std::cout << "============================================================\n\n";
            }

            const int gs = kernel_grid;

            for (double rho : density_rhos) {
                // Memory check
                double mem_est_gb = estimateMemoryGB(gs, rho, nranks);
                if (mem_est_gb > max_mem_gb * 0.4) {
                    if (rank == 0) {
                        std::cout << "--- rho = " << rho << " ---\n";
                        std::cout << "  [SKIP] Estimated memory " << std::fixed
                                  << std::setprecision(1) << mem_est_gb
                                  << " GB exceeds limit (" << max_mem_gb * 0.4
                                  << " GB)\n\n";
                    }
                    continue;
                }

                size_type Np   = static_cast<size_type>(double(gs) * gs * gs * rho);
                size_type nloc = Np / nranks;

                if (rank == 0) {
                    std::cout << "--- rho = " << rho << " ---\n";
                    std::cout << "  Np=" << Np << "  est. memory: " << std::fixed
                              << std::setprecision(1) << mem_est_gb << " GB\n";
                }

                try {

                // Setup mesh & layout
                ippl::Vector<int, dim> pt = {gs, gs, gs};
                auto owned = ippl::NDIndex<dim>(ippl::Index(pt[0]), ippl::Index(pt[1]),
                                         ippl::Index(pt[2]));

                std::array<bool, dim> isParallel;
                isParallel.fill(false);
                ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

                Vector_t minU = {0, 0, 0};
                Vector_t maxU = {2*pi, 2*pi, 2*pi};
                std::array<double, dim> dx = {
                    (maxU[0] - minU[0]) / double(pt[0]),
                    (maxU[1] - minU[1]) / double(pt[1]),
                    (maxU[2] - minU[2]) / double(pt[2]),
                };
                Vector_t hx     = {dx[0], dx[1], dx[2]};
                Vector_t origin = {minU[0], minU[1], minU[2]};
                Mesh_t mesh(owned, hx, origin);

                playout_type pl(layout, mesh);

                bunch_type bunch(pl);
                bunch.setParticleBC(ippl::BC::PERIODIC);
                bunch.create(nloc);

                field_type field(mesh, layout);

                Kokkos::Random_XorShift64_Pool<> rand_pool64(size_type(42 + int(rho * 1000)));
                Kokkos::parallel_for(
                    nloc,
                    generate_random_particles_with_charges<Vector_t,
                                                          Kokkos::Random_XorShift64_Pool<>, dim>(
                        bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));
                Kokkos::fence();

                // Fill field for Type 2
                const int nghost = field.getNghost();
                using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
                auto fview = field.getView();
                Kokkos::parallel_for(
                    mdrange_type({nghost, nghost, nghost},
                                 {fview.extent(0) - nghost, fview.extent(1) - nghost,
                                  fview.extent(2) - nghost}),
                    generate_random_field<Kokkos::complex<double>,
                                         Kokkos::Random_XorShift64_Pool<>, dim>(
                        field.getView(), rand_pool64));
                Kokkos::fence();

                // ---- Type 1: Spreading methods ----
                for (const auto& sc : spread_methods) {
                    try {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", density_tol);
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", false);
                        fftParams.add("spread_method", sc.spread_method);
                        fftParams.add("lock_method", true);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                        double t = benchmarkType1(*fft, field, bunch,
                                                  warmup_runs, benchmark_runs);
                        double throughput = Np / t * 1000.0 / 1e6;

                        density_results.push_back(
                            {gs, rho, density_tol, sc.name, "1", t, throughput});

                        if (rank == 0) {
                            std::cout << "  Type1 " << std::setw(15) << sc.name
                                      << "  " << std::fixed << std::setprecision(1)
                                      << throughput << " Mpts/s  (" << std::setprecision(3)
                                      << t << " ms)\n";
                        }
                    } catch (const std::exception& e) {
                        if (rank == 0)
                            std::cerr << "  FAILED: Type1 " << sc.name
                                      << ": " << e.what() << "\n";
                    }
                }

                // ---- Type 2: Interpolation methods ----
                for (const auto& gc : gather_methods) {
                    try {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", density_tol);
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", false);
                        fftParams.add("gather_method", gc.gather_method);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                        double t = benchmarkType2(*fft, field, bunch,
                                                  warmup_runs, benchmark_runs);
                        double throughput = Np / t * 1000.0 / 1e6;

                        density_results.push_back(
                            {gs, rho, density_tol, gc.name, "2", t, throughput});

                        if (rank == 0) {
                            std::cout << "  Type2 " << std::setw(15) << gc.name
                                      << "  " << std::fixed << std::setprecision(1)
                                      << throughput << " Mpts/s  (" << std::setprecision(3)
                                      << t << " ms)\n";
                        }
                    } catch (const std::exception& e) {
                        if (rank == 0)
                            std::cerr << "  FAILED: Type2 " << gc.name
                                      << ": " << e.what() << "\n";
                    }
                }

                // ---- cuFINUFFT / FINUFFT comparison ----
#ifdef ENABLE_FINUFFT
                {
                    auto Rview = bunch.R.getView();
                    auto Qview = bunch.Q.getView();
                    auto fview_local = field.getView();

#ifdef ENABLE_GPU_NUFFT
                    std::string finufft_label = "cuFINUFFT";
#else
                    std::string finufft_label = "FINUFFT";
#endif
                    // Type 1
                    try {
                        double t = benchmarkType1FINUFFT(
                            Rview, Qview, fview_local, gs, nloc, density_tol, nghost,
                            warmup_runs, benchmark_runs);
                        double throughput = Np / t * 1000.0 / 1e6;

                        density_results.push_back(
                            {gs, rho, density_tol, finufft_label, "1", t, throughput});

                        if (rank == 0) {
                            std::cout << "  Type1 " << std::setw(15) << finufft_label
                                      << "  " << std::fixed << std::setprecision(1)
                                      << throughput << " Mpts/s  (" << std::setprecision(3)
                                      << t << " ms)\n";
                        }
                    } catch (const std::exception& e) {
                        if (rank == 0)
                            std::cerr << "  FAILED: " << finufft_label
                                      << " Type1: " << e.what() << "\n";
                    }

                    // Type 2
                    try {
                        double t = benchmarkType2FINUFFT(
                            Rview, Qview, fview_local, gs, nloc, density_tol, nghost,
                            warmup_runs, benchmark_runs);
                        double throughput = Np / t * 1000.0 / 1e6;

                        density_results.push_back(
                            {gs, rho, density_tol, finufft_label, "2", t, throughput});

                        if (rank == 0) {
                            std::cout << "  Type2 " << std::setw(15) << finufft_label
                                      << "  " << std::fixed << std::setprecision(1)
                                      << throughput << " Mpts/s  (" << std::setprecision(3)
                                      << t << " ms)\n";
                        }
                    } catch (const std::exception& e) {
                        if (rank == 0)
                            std::cerr << "  FAILED: " << finufft_label
                                      << " Type2: " << e.what() << "\n";
                    }
                }
#endif  // ENABLE_FINUFFT

                if (rank == 0) std::cout << "\n";

                } catch (const std::exception& e) {
                    if (rank == 0)
                        std::cerr << "  [OOM] Skipping rho=" << rho
                                  << " grid=" << gs << ": " << e.what() << "\n\n";
                }
            }  // rho loop
        }  // do_density

        // ============================================================
        // Write CSV outputs
        // ============================================================
        if (rank == 0) {
            // Kernel results CSV
            if (!kernel_results.empty()) {
                std::string fname = output_prefix + "_kernels.csv";
                std::ofstream ofs(fname);
                ofs << "grid,rho,tolerance,method,type,time_ms,throughput_mpts\n";
                for (const auto& r : kernel_results) {
                    ofs << r.grid << "," << r.rho << ","
                        << std::scientific << std::setprecision(1) << r.tolerance << ","
                        << r.method << "," << r.type << ","
                        << std::fixed << std::setprecision(4) << r.time_ms << ","
                        << std::setprecision(2) << r.throughput_mpts << "\n";
                }
                ofs.close();
                std::cout << "Kernel results written to: " << fname << "\n";
            }

            // Breakdown results CSV
            if (!breakdown_results.empty()) {
                std::string fname = output_prefix + "_breakdown.csv";
                std::ofstream ofs(fname);
                ofs << "grid,rho,tolerance,type,component,time_ms,fraction\n";
                for (const auto& r : breakdown_results) {
                    ofs << r.grid << "," << r.rho << ","
                        << std::scientific << std::setprecision(1) << r.tolerance << ","
                        << r.type << "," << r.component << ","
                        << std::fixed << std::setprecision(4) << r.time_ms << ","
                        << std::setprecision(4) << r.fraction << "\n";
                }
                ofs.close();
                std::cout << "Breakdown results written to: " << fname << "\n";
            }

            // Density sweep results CSV
            if (!density_results.empty()) {
                std::string fname = output_prefix + "_density.csv";
                std::ofstream ofs(fname);
                ofs << "grid,rho,tolerance,method,type,time_ms,throughput_mpts\n";
                for (const auto& r : density_results) {
                    ofs << r.grid << "," << r.rho << ","
                        << std::scientific << std::setprecision(1) << r.tolerance << ","
                        << r.method << "," << r.type << ","
                        << std::fixed << std::setprecision(4) << r.time_ms << ","
                        << std::setprecision(2) << r.throughput_mpts << "\n";
                }
                ofs.close();
                std::cout << "Density results written to: " << fname << "\n";
            }
        }
    }
    ippl::finalize();
    return 0;
}
