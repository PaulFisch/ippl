#include "Ippl.h"
#include "Utility/ParameterList.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>

#ifdef ENABLE_FINUFFT
#include "finufft_wrapper.h"
#endif

// ============================================================================
// Particle Bunch
// ============================================================================

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

// ============================================================================
// Random Generation Kernels
// ============================================================================

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

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const {
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

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i, const size_t j, const size_t k) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();
        f(i, j, k).real() = rand_gen.drand(0.0, 1.0);
        f(i, j, k).imag() = rand_gen.drand(0.0, 1.0);
        rand_pool.free_state(rand_gen);
    }
};

// ============================================================================
// Benchmark Functions
// ============================================================================

template<typename FFT_type, typename Field, typename BunchType>
double benchmarkType1(FFT_type& fft, Field& field, BunchType& bunch,
                      int warmup_runs = 3, int benchmark_runs = 10) {
    for (int i = 0; i < warmup_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
    }
#ifdef ENABLE_GPU_NUFFT
    cudaDeviceSynchronize();
#endif

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
    }
    Kokkos::fence();
#ifdef ENABLE_GPU_NUFFT
    cudaDeviceSynchronize();
#endif
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
}

template<typename FFT_type, typename Field, typename BunchType>
double benchmarkType2(FFT_type& fft, Field& field, BunchType& bunch,
                      int warmup_runs = 3, int benchmark_runs = 10) {
    for (int i = 0; i < warmup_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
    }
#ifdef ENABLE_GPU_NUFFT
    cudaDeviceSynchronize();
#endif

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
    }
    Kokkos::fence();
#ifdef ENABLE_GPU_NUFFT
    cudaDeviceSynchronize();
#endif
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
}

#ifdef ENABLE_FINUFFT
template<typename PosView, typename StrengthView, typename FieldView>
double benchmarkType1Direct(PosView& positions, StrengthView& strengths, FieldView& field,
                            int64_t grid_size, int64_t n_particles, double tol, int nghost,
                            int warmup_runs = 3, int benchmark_runs = 10) {
    using T = double;
    using complex_type = Kokkos::complex<T>;
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;

#ifdef ENABLE_GPU_NUFFT
    using finufft_complex = cuDoubleComplex;
#else
    using finufft_complex = fftw_complex;
#endif

    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> x("x", n_particles);
    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> y("y", n_particles);
    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> z("z", n_particles);
    Kokkos::View<complex_type*, Kokkos::LayoutLeft, memory_space> c("c", n_particles);
    Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space> f("f", grid_size, grid_size, grid_size);

    Kokkos::parallel_for("copy_positions", n_particles,
        KOKKOS_LAMBDA(const int64_t i) {
            x(i) = positions(i)[0];
            y(i) = positions(i)[1];
            z(i) = positions(i)[2];
            c(i) = complex_type(strengths(i), T(0));
        });
    Kokkos::fence();

    finufft_wrapper::Config<T> config;
    config.tolerance = tol;
    config.type = 1;
#ifdef ENABLE_GPU_NUFFT
    config.gpu_method = 1;
    config.gpu_sort = 1;
    config.gpu_kerevalmeth = 1;
#else
    config.nthreads = 0;
    config.spread_sort = 2;
    config.spread_kerevalmeth = 1;
#endif

    finufft_wrapper::DirectNUFFT3D<T> nufft;
    int64_t n_modes[3] = {grid_size, grid_size, grid_size};
    nufft.make_plan(n_modes, config);
    nufft.set_points(n_particles, x.data(), y.data(), z.data());

    auto run_transform = [&]() {
        Kokkos::parallel_for("copy_strengths", n_particles,
            KOKKOS_LAMBDA(const int64_t i) {
                c(i) = complex_type(strengths(i), T(0));
            });
        Kokkos::fence();

        nufft.execute(reinterpret_cast<finufft_complex*>(c.data()),
                      reinterpret_cast<finufft_complex*>(f.data()));
        Kokkos::fence();
#ifdef ENABLE_GPU_NUFFT
        cudaDeviceSynchronize();
#endif
    };

    for (int i = 0; i < warmup_runs; ++i) run_transform();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) run_transform();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
}

template<typename PosView, typename OutputView, typename FieldView>
double benchmarkType2Direct(PosView& positions, OutputView& output, FieldView& field,
                            int64_t grid_size, int64_t n_particles, double tol, int nghost,
                            int warmup_runs = 3, int benchmark_runs = 10) {
    using T = double;
    using complex_type = Kokkos::complex<T>;
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;

#ifdef ENABLE_GPU_NUFFT
    using finufft_complex = cuDoubleComplex;
#else
    using finufft_complex = fftw_complex;
#endif

    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> x("x", n_particles);
    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> y("y", n_particles);
    Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> z("z", n_particles);
    Kokkos::View<complex_type*, Kokkos::LayoutLeft, memory_space> c("c", n_particles);
    Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space> f("f", grid_size, grid_size, grid_size);

    Kokkos::parallel_for("copy_positions", n_particles,
        KOKKOS_LAMBDA(const int64_t i) {
            x(i) = positions(i)[0];
            y(i) = positions(i)[1];
            z(i) = positions(i)[2];
        });

    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    Kokkos::parallel_for("copy_field", mdrange_type({0, 0, 0}, {grid_size, grid_size, grid_size}),
        KOKKOS_LAMBDA(const int64_t i, const int64_t j, const int64_t k) {
            f(i, j, k) = field(i + nghost, j + nghost, k + nghost);
        });
    Kokkos::fence();

    finufft_wrapper::Config<T> config;
    config.tolerance = tol;
    config.type = 2;
#ifdef ENABLE_GPU_NUFFT
    config.gpu_method = 1;
    config.gpu_sort = 1;
    config.gpu_kerevalmeth = 1;
#else
    config.nthreads = 0;
    config.spread_sort = 2;
    config.spread_kerevalmeth = 1;
#endif

    finufft_wrapper::DirectNUFFT3D<T> nufft;
    int64_t n_modes[3] = {grid_size, grid_size, grid_size};
    nufft.make_plan(n_modes, config);
    nufft.set_points(n_particles, x.data(), y.data(), z.data());

    auto run_transform = [&]() {
        nufft.execute(reinterpret_cast<finufft_complex*>(c.data()),
                      reinterpret_cast<finufft_complex*>(f.data()));
        Kokkos::fence();
#ifdef ENABLE_GPU_NUFFT
        cudaDeviceSynchronize();
#endif
    };

    for (int i = 0; i < warmup_runs; ++i) run_transform();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) run_transform();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
}
#endif

// ============================================================================
// Result Structure
// ============================================================================

struct BenchResult {
    double density;
    int64_t n_particles;
    int64_t n_modes;
    int type;
    std::string method;
    double time_ms;
    double throughput_mpts;
};

// ============================================================================
// Main
// ============================================================================

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
        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type field_type;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type real_field_type;
        typedef ippl::FFT<ippl::NUFFTransform, real_field_type> FFT_type;

        // Configuration
        int grid_size = 64;
        double tol = 1e-6;
        int warmup_runs = 3;
        int benchmark_runs = 10;
        std::string output_file = "density_sweep_results.csv";

        // Parse arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--grid" && i + 1 < argc) grid_size = std::stoi(argv[++i]);
            else if (arg == "--tol" && i + 1 < argc) tol = std::stod(argv[++i]);
            else if (arg == "--warmup" && i + 1 < argc) warmup_runs = std::stoi(argv[++i]);
            else if (arg == "--runs" && i + 1 < argc) benchmark_runs = std::stoi(argv[++i]);
            else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
            else if (arg == "--help") {
                if (ippl::Comm->rank() == 0) {
                    std::cout << "Usage: " << argv[0] << " [options]\n"
                              << "  --grid N      Grid size (default: 64)\n"
                              << "  --tol T       Tolerance (default: 1e-6)\n"
                              << "  --warmup N    Warmup runs (default: 3)\n"
                              << "  --runs N      Benchmark runs (default: 10)\n"
                              << "  --output FILE Output CSV file\n";
                }
                ippl::finalize();
                return 0;
            }
        }

        // Density sweep: particles per mode (N^3 modes total)
        std::vector<double> densities = {
            1e-6, 1e-5, 1e-4, 1e-3,
            0.01, 0.1,
            0.5, 1.0, 2.0,
            5.0, 10.0, 50.0, 100.0,
            500.0, 1000.0, 5000.0, 10000.0
        };

        int64_t n_modes = static_cast<int64_t>(grid_size) * grid_size * grid_size;
        std::vector<BenchResult> results;

        if (ippl::Comm->rank() == 0) {
            std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║       NUFFT Density Sweep Benchmark (3D)                     ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║  Grid:      " << std::setw(6) << grid_size << "^3 = " 
                      << std::setw(10) << n_modes << " modes                  ║\n";
            std::cout << "║  Tolerance: " << std::scientific << std::setprecision(1) 
                      << std::setw(10) << tol << "                                  ║\n";
            std::cout << "║  Warmup:    " << std::setw(6) << warmup_runs << "                                        ║\n";
            std::cout << "║  Runs:      " << std::setw(6) << benchmark_runs << "                                        ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
        }

        // Setup mesh and layout (once, reused)
        ippl::Vector<int, dim> pt = {grid_size, grid_size, grid_size};
        ippl::Index I(pt[0]), J(pt[1]), K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);
        std::array<bool, dim> isParallel = {false, false, false};
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        Vector_t minU = {0, 0, 0};
        Vector_t maxU = {2 * pi, 2 * pi, 2 * pi};
        std::array<double, dim> dx = {
            (maxU[0] - minU[0]) / double(pt[0]),
            (maxU[1] - minU[1]) / double(pt[1]),
            (maxU[2] - minU[2]) / double(pt[2]),
        };
        Vector_t hx = {dx[0], dx[1], dx[2]};
        Vector_t origin = {minU[0], minU[1], minU[2]};
        ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);
        playout_type pl(layout, mesh);

        // Sweep over densities
        for (double density : densities) {
            int64_t Np = static_cast<int64_t>(density * n_modes);
            if (Np < 1) Np = 1;
            size_type nloc = Np / ippl::Comm->size();
            if (nloc < 1) nloc = 1;

            if (ippl::Comm->rank() == 0) {
                std::cout << "\n" << std::string(70, '=') << "\n";
                std::cout << "Density: " << std::scientific << std::setprecision(2) << density
                          << " | Particles: " << Np << " | M/N ratio: " 
                          << std::fixed << std::setprecision(4) << double(Np) / n_modes << "\n";
                std::cout << std::string(70, '-') << "\n";
            }

            // ================================================================
            // TYPE 1 BENCHMARKS
            // ================================================================
            {
                bunch_type bunch(pl);
                bunch.setParticleBC(ippl::BC::PERIODIC);
                bunch.create(nloc);
                field_type field(mesh, layout);

                Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + int(density * 1000)));
                Kokkos::parallel_for(nloc,
                    generate_random_particles_with_charges<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                        bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));
                Kokkos::fence();

                // OutputFocused
                {
                    ippl::ParameterList fftParams;
                    fftParams.add("tolerance", tol);
                    fftParams.add("use_finufft_defaults", false);
                    fftParams.add("use_kokkos_nufft", false);
                    fftParams.add("spread_method", "output_focused");
                    fftParams.add("tile_size_3d", 4);

                    auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                    double time_ms = benchmarkType1(*fft, field, bunch, warmup_runs, benchmark_runs);
                    double throughput = Np / time_ms * 1000.0 / 1e6;

                    results.push_back({density, Np, n_modes, 1, "Native_OutputFocused", time_ms, throughput});
                    if (ippl::Comm->rank() == 0) {
                        std::cout << "  Type1 OutputFocused: " << std::fixed << std::setprecision(3) 
                                  << time_ms << " ms (" << throughput << " Mpts/s)\n";
                    }
                }

                // Tiled
                {
                    ippl::ParameterList fftParams;
                    fftParams.add("tolerance", tol);
                    fftParams.add("use_finufft_defaults", false);
                    fftParams.add("use_kokkos_nufft", false);
                    fftParams.add("spread_method", "tiled");
                    fftParams.add("tile_size_3d", 3);

                    auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                    double time_ms = benchmarkType1(*fft, field, bunch, warmup_runs, benchmark_runs);
                    double throughput = Np / time_ms * 1000.0 / 1e6;

                    results.push_back({density, Np, n_modes, 1, "Native_Tiled", time_ms, throughput});
                    if (ippl::Comm->rank() == 0) {
                        std::cout << "  Type1 Tiled:         " << std::fixed << std::setprecision(3) 
                                  << time_ms << " ms (" << throughput << " Mpts/s)\n";
                    }
                }

                // Atomic
                {
                    ippl::ParameterList fftParams;
                    fftParams.add("tolerance", tol);
                    fftParams.add("use_finufft_defaults", false);
                    fftParams.add("use_kokkos_nufft", false);
                    fftParams.add("spread_method", "atomic");

                    auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                    double time_ms = benchmarkType1(*fft, field, bunch, warmup_runs, benchmark_runs);
                    double throughput = Np / time_ms * 1000.0 / 1e6;

                    results.push_back({density, Np, n_modes, 1, "Native_Atomic", time_ms, throughput});
                    if (ippl::Comm->rank() == 0) {
                        std::cout << "  Type1 Atomic:        " << std::fixed << std::setprecision(3) 
                                  << time_ms << " ms (" << throughput << " Mpts/s)\n";
                    }
                }

                // ZBatched
                {
                    ippl::ParameterList fftParams;
                    fftParams.add("tolerance", tol);
                    fftParams.add("use_finufft_defaults", false);
                    fftParams.add("use_kokkos_nufft", false);
                    fftParams.add("spread_method", "output_focused_zbatched");
                    fftParams.add("tile_size_3d", 16);
                    fftParams.add("z_tiles", 1);

                    auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                    double time_ms = benchmarkType1(*fft, field, bunch, warmup_runs, benchmark_runs);
                    double throughput = Np / time_ms * 1000.0 / 1e6;

                    results.push_back({density, Np, n_modes, 1, "Native_OutputFocusedZBatch", time_ms, throughput});
                    if (ippl::Comm->rank() == 0) {
                        std::cout << "  Type1 OutputFocusedZBatch:        " << std::fixed << std::setprecision(3)
                                  << time_ms << " ms (" << throughput << " Mpts/s)\n";
                    }
                }

#ifdef ENABLE_FINUFFT
                // cuFINUFFT
                {
                    const int nghost = field.getNghost();
                    auto Rview = bunch.R.getView();
                    auto Qview = bunch.Q.getView();
                    auto fview = field.getView();

                    try {
                        double time_ms = benchmarkType1Direct(Rview, Qview, fview,
                                                              grid_size, nloc, tol, nghost,
                                                              warmup_runs, benchmark_runs);
                        double throughput = Np / time_ms * 1000.0 / 1e6;

                        results.push_back({density, Np, n_modes, 1, "cuFINUFFT", time_ms, throughput});
                        if (ippl::Comm->rank() == 0) {
                            std::cout << "  Type1 cuFINUFFT:     " << std::fixed << std::setprecision(3) 
                                      << time_ms << " ms (" << throughput << " Mpts/s)\n";
                        }
                    } catch (const std::exception& e) {
                        if (ippl::Comm->rank() == 0)
                            std::cerr << "  Type1 cuFINUFFT: FAILED - " << e.what() << "\n";
                    }
                }
#endif
            }

            // ================================================================
            // TYPE 2 BENCHMARKS
            // ================================================================
            {
                bunch_type bunch(pl);
                bunch.setParticleBC(ippl::BC::PERIODIC);
                bunch.create(nloc);
                field_type field(mesh, layout);

                Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(123 + int(density * 1000)));

                // Generate random particles
                Kokkos::parallel_for(nloc,
                    generate_random_particles_with_charges<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                        bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));

                // Generate random field
                const int nghost = field.getNghost();
                using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
                auto fview = field.getView();
                Kokkos::parallel_for(
                    mdrange_type({nghost, nghost, nghost},
                                {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
                    generate_random_field<Kokkos::complex<double>, Kokkos::Random_XorShift64_Pool<>, dim>(
                        fview, rand_pool64));
                Kokkos::fence();

                // Atomic
                {
                    ippl::ParameterList fftParams;
                    fftParams.add("tolerance", tol);
                    fftParams.add("use_finufft_defaults", false);
                    fftParams.add("use_kokkos_nufft", false);
                    fftParams.add("gather_method", "atomic");

                    auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                    double time_ms = benchmarkType2(*fft, field, bunch, warmup_runs, benchmark_runs);
                    double throughput = Np / time_ms * 1000.0 / 1e6;

                    results.push_back({density, Np, n_modes, 2, "Native_Atomic", time_ms, throughput});
                    if (ippl::Comm->rank() == 0) {
                        std::cout << "  Type2 Atomic:        " << std::fixed << std::setprecision(3) 
                                  << time_ms << " ms (" << throughput << " Mpts/s)\n";
                    }
                }

                // Atomic Sort
                {
                    ippl::ParameterList fftParams;
                    fftParams.add("tolerance", tol);
                    fftParams.add("use_finufft_defaults", false);
                    fftParams.add("use_kokkos_nufft", false);
                    fftParams.add("gather_method", "atomic_sort");

                    auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                    double time_ms = benchmarkType2(*fft, field, bunch, warmup_runs, benchmark_runs);
                    double throughput = Np / time_ms * 1000.0 / 1e6;

                    results.push_back({density, Np, n_modes, 2, "Native_AtomicSort", time_ms, throughput});
                    if (ippl::Comm->rank() == 0) {
                        std::cout << "  Type2 AtomicSort:    " << std::fixed << std::setprecision(3) 
                                  << time_ms << " ms (" << throughput << " Mpts/s)\n";
                    }
                }

#ifdef ENABLE_FINUFFT
                // cuFINUFFT Type 2
                {
                    auto Rview = bunch.R.getView();
                    auto Qview = bunch.Q.getView();
                    auto fview_local = field.getView();

                    try {
                        double time_ms = benchmarkType2Direct(Rview, Qview, fview_local,
                                                              grid_size, nloc, tol, nghost,
                                                              warmup_runs, benchmark_runs);
                        double throughput = Np / time_ms * 1000.0 / 1e6;

                        results.push_back({density, Np, n_modes, 2, "cuFINUFFT", time_ms, throughput});
                        if (ippl::Comm->rank() == 0) {
                            std::cout << "  Type2 cuFINUFFT:     " << std::fixed << std::setprecision(3) 
                                      << time_ms << " ms (" << throughput << " Mpts/s)\n";
                        }
                    } catch (const std::exception& e) {
                        if (ippl::Comm->rank() == 0)
                            std::cerr << "  Type2 cuFINUFFT: FAILED - " << e.what() << "\n";
                    }
                }
#endif
            }
        }

        // Write CSV
        if (ippl::Comm->rank() == 0) {
            std::ofstream csv(output_file);
            csv << "density,n_particles,n_modes,type,method,time_ms,throughput_mpts\n";
            for (const auto& r : results) {
                csv << std::scientific << std::setprecision(6) << r.density << ","
                    << r.n_particles << ","
                    << r.n_modes << ","
                    << r.type << ","
                    << r.method << ","
                    << std::fixed << std::setprecision(6) << r.time_ms << ","
                    << r.throughput_mpts << "\n";
            }
            std::cout << "\n" << std::string(70, '=') << "\n";
            std::cout << "Results written to: " << output_file << "\n";
            std::cout << std::string(70, '=') << "\n";
        }
    }
    ippl::finalize();
    return 0;
}