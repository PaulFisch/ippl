/**
 * @file BenchmarkAtomicScatter.cpp
 * @brief CORRECTED Performance model validation for AtomicScatter kernel
 *
 * FIXES APPLIED:
 *   1. Uses windowed m_L (cache-line multiplicity over particles-in-flight)
 *      NOT phi_global (which is just average reuse, not contention)
 *   2. No waves factor - uses global A_eff directly: T_atomic = N*S*eta / A_eff
 *   3. Gets SM_count from device properties (not hardcoded 80)
 *   4. Separates scatter timing from halo exchange timing
 *   5. Uses BW_scatter (not BW_stream)
 *   6. Calibration uses TeamPolicy with FIXED working set
 *
 * Usage: ./BenchmarkAtomicScatter [options]
 *   --grid N        Grid size per dimension (default: 64)
 *   --rho R         Particles per grid point (default: 10)
 *   --tol T         Kernel tolerance (default: 1e-6)
 *   --runs N        Number of benchmark runs (default: 20)
 *   --sweep-w       Sweep kernel widths w=2..10
 *   --dist D        Distribution: uniform, clustered (default: uniform)
 *   --calibrate     Run calibration microbenchmarks
 *   -v, --verbose   Verbose output
 */

#include "Ippl.h"
#include <Kokkos_Random.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>

using namespace ippl;

// ============================================================================
// Timer
// ============================================================================

class ManualTimer {
public:
    void start() {
        Kokkos::fence();
        start_ = std::chrono::high_resolution_clock::now();
    }
    double stop() {
        Kokkos::fence();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Parameters
// ============================================================================

struct BenchParams {
    int n_grid = 64;
    double rho = 10.0;
    double kernel_tol = 1e-6;
    int warmup_runs = 5;
    int benchmark_runs = 20;
    std::string output_prefix = "atomic_scatter";
    std::string distribution = "uniform";
    bool verbose = false;
    bool sweep_w = false;
    bool calibrate = false;

    size_t n_particles() const {
        return static_cast<size_t>(rho * n_grid * n_grid * n_grid);
    }
};

BenchParams parse_args(int argc, char* argv[]) {
    BenchParams p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--grid" && i + 1 < argc) p.n_grid = std::atoi(argv[++i]);
        else if (arg == "--rho" && i + 1 < argc) p.rho = std::atof(argv[++i]);
        else if (arg == "--tol" && i + 1 < argc) p.kernel_tol = std::atof(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) p.warmup_runs = std::atoi(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) p.benchmark_runs = std::atoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) p.output_prefix = argv[++i];
        else if (arg == "--dist" && i + 1 < argc) p.distribution = argv[++i];
        else if (arg == "--sweep-w") p.sweep_w = true;
        else if (arg == "--calibrate") p.calibrate = true;
        else if (arg == "-v" || arg == "--verbose") p.verbose = true;
    }
    return p;
}

// ============================================================================
// Statistics
// ============================================================================

struct TimingStats {
    double mean_ms, stddev_ms, min_ms, max_ms, median_ms;
};

TimingStats compute_stats(const std::vector<double>& times_sec) {
    TimingStats s{};
    size_t n = times_sec.size();
    if (n == 0) return s;

    std::vector<double> ms(n);
    for (size_t i = 0; i < n; ++i) ms[i] = times_sec[i] * 1000.0;

    s.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / n;
    double sq = 0;
    for (double t : ms) sq += (t - s.mean_ms) * (t - s.mean_ms);
    s.stddev_ms = (n > 1) ? std::sqrt(sq / (n - 1)) : 0;
    s.min_ms = *std::min_element(ms.begin(), ms.end());
    s.max_ms = *std::max_element(ms.begin(), ms.end());

    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    s.median_ms = (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2 : sorted[n/2];
    return s;
}

// ============================================================================
// CORRECTED: Windowed Collision Metrics
// ============================================================================

template <unsigned Dim>
struct WindowedMetrics {
    // What the model should use:
    double m_L;                     // Updates per cache line within particles-in-flight window
    double m_L_stddev;
    double m_L_max;
    size_t avg_unique_lines_per_window;
    size_t window_size;

    // For reference only (NOT for model):
    double phi_global;              // = N*S / unique_points (meaningless for contention)
    double phi_block_mean;          // Within-block overlap (misses inter-block contention)
    size_t total_cache_lines;
    size_t total_updates;
};

template <typename real_type, unsigned Dim>
WindowedMetrics<Dim> compute_windowed_metrics(
    const Kokkos::View<ippl::Vector<real_type, Dim>*>& R_view,
    size_t n_particles,
    int n_grid,
    int kernel_width,
    size_t window_size)
{
    WindowedMetrics<Dim> m{};
    m.window_size = window_size;

    auto R_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R_view);

    double h = 2.0 * M_PI / n_grid;
    int S = 1;
    for (unsigned d = 0; d < Dim; ++d) S *= kernel_width;

    // Cache line parameters (128B lines, 16B per complex<double>)
    constexpr size_t LINE_SIZE = 128;
    constexpr size_t ELEM_SIZE = 16;
    constexpr size_t ELEMS_PER_LINE = LINE_SIZE / ELEM_SIZE;

    auto idx_to_line = [&](size_t idx) { return idx / ELEMS_PER_LINE; };

    // Global statistics (for reference only)
    std::unordered_set<size_t> global_lines;
    m.total_updates = n_particles * S;

    // Enumerate all grid points touched
    for (size_t i = 0; i < n_particles; ++i) {
        int base[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            base[d] = static_cast<int>(std::floor(R_host(i)[d] / h)) - kernel_width / 2;
        }

        if constexpr (Dim == 3) {
            for (int iz = 0; iz < kernel_width; ++iz) {
                for (int iy = 0; iy < kernel_width; ++iy) {
                    for (int ix = 0; ix < kernel_width; ++ix) {
                        int gx = (base[0] + ix + n_grid) % n_grid;
                        int gy = (base[1] + iy + n_grid) % n_grid;
                        int gz = (base[2] + iz + n_grid) % n_grid;
                        size_t idx = gx + n_grid * (gy + n_grid * gz);
                        global_lines.insert(idx_to_line(idx));
                    }
                }
            }
        }
    }

    m.total_cache_lines = global_lines.size();
    m.phi_global = static_cast<double>(m.total_updates) / global_lines.size();

    // WINDOWED analysis - THE KEY METRIC
    size_t n_windows = (n_particles + window_size - 1) / window_size;
    std::vector<double> m_L_per_window(n_windows);
    size_t sum_unique_lines = 0;

    for (size_t w = 0; w < n_windows; ++w) {
        size_t w_start = w * window_size;
        size_t w_end = std::min(w_start + window_size, n_particles);

        std::unordered_set<size_t> window_lines;
        size_t window_updates = 0;

        for (size_t i = w_start; i < w_end; ++i) {
            int base[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                base[d] = static_cast<int>(std::floor(R_host(i)[d] / h)) - kernel_width / 2;
            }

            if constexpr (Dim == 3) {
                for (int iz = 0; iz < kernel_width; ++iz) {
                    for (int iy = 0; iy < kernel_width; ++iy) {
                        for (int ix = 0; ix < kernel_width; ++ix) {
                            int gx = (base[0] + ix + n_grid) % n_grid;
                            int gy = (base[1] + iy + n_grid) % n_grid;
                            int gz = (base[2] + iz + n_grid) % n_grid;
                            size_t idx = gx + n_grid * (gy + n_grid * gz);
                            window_lines.insert(idx_to_line(idx));
                            window_updates++;
                        }
                    }
                }
            }
        }

        size_t unique = window_lines.size();
        sum_unique_lines += unique;
        m_L_per_window[w] = (unique > 0) ? static_cast<double>(window_updates) / unique : 1.0;
    }

    // Aggregate windowed metrics
    double sum = 0, max_val = 0;
    for (double x : m_L_per_window) {
        sum += x;
        max_val = std::max(max_val, x);
    }
    m.m_L = sum / n_windows;
    m.m_L_max = max_val;
    m.avg_unique_lines_per_window = sum_unique_lines / n_windows;

    double sq = 0;
    for (double x : m_L_per_window) sq += (x - m.m_L) * (x - m.m_L);
    m.m_L_stddev = (n_windows > 1) ? std::sqrt(sq / (n_windows - 1)) : 0;

    // Per-block (for reference, misses inter-block contention)
    int team_size = 4;
    size_t n_blocks = (n_particles + team_size - 1) / team_size;
    double sum_phi = 0;
    for (size_t b = 0; b < n_blocks; ++b) {
        std::unordered_set<size_t> block_pts;
        size_t b_start = b * team_size;
        size_t b_end = std::min(b_start + team_size, n_particles);
        for (size_t i = b_start; i < b_end; ++i) {
            int base[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                base[d] = static_cast<int>(std::floor(R_host(i)[d] / h)) - kernel_width / 2;
            }
            if constexpr (Dim == 3) {
                for (int iz = 0; iz < kernel_width; ++iz) {
                    for (int iy = 0; iy < kernel_width; ++iy) {
                        for (int ix = 0; ix < kernel_width; ++ix) {
                            int gx = (base[0] + ix + n_grid) % n_grid;
                            int gy = (base[1] + iy + n_grid) % n_grid;
                            int gz = (base[2] + iz + n_grid) % n_grid;
                            block_pts.insert(gx + n_grid * (gy + n_grid * gz));
                        }
                    }
                }
            }
        }
        size_t updates = (b_end - b_start) * S;
        sum_phi += (block_pts.size() > 0) ? static_cast<double>(updates) / block_pts.size() : 1.0;
    }
    m.phi_block_mean = sum_phi / n_blocks;

    return m;
}

// ============================================================================
// CORRECTED: Device Constants
// ============================================================================

struct DeviceConstants {
    std::string device_name = "Unknown";
    int SM_count = 80;

    // Calibrated with TeamPolicy, FIXED L2-resident array
    double A_L2_unique = 1e11;      // Atomics/sec, unique addresses
    double A_L2_hot = 5e8;          // Atomics/sec, full contention

    // Scatter bandwidth (NOT streaming)
    double BW_scatter = 550e9;      // bytes/sec

    size_t particles_in_flight() const {
        // Conservative: SM_count * blocks_per_SM * team_size
        return SM_count * 8 * 4;
    }
};

// ============================================================================
// CORRECTED: Performance Model (NO WAVES)
// ============================================================================

struct ModelPrediction {
    double m_L;
    double A_eff;
    double T_atomic_sec;
    double T_memory_sec;
    double T_kernel_sec;
    double throughput_Mpts;
    std::string bottleneck;
};

ModelPrediction predict_time(
    const DeviceConstants& dev,
    const WindowedMetrics<3>& metrics,
    int kernel_width,
    size_t N)
{
    ModelPrediction p{};

    int S = kernel_width * kernel_width * kernel_width;
    int eta = 2;  // Real + imag atomics per grid point

    p.m_L = metrics.m_L;

    // A_eff from windowed cache-line contention
    // A_eff(m_L) = (1/A_L2_unique + (m_L - 1)/A_L2_hot)^(-1)
    if (p.m_L <= 1.0) {
        p.A_eff = dev.A_L2_unique;
    } else {
        p.A_eff = 1.0 / (1.0 / dev.A_L2_unique + (p.m_L - 1.0) / dev.A_L2_hot);
    }

    // CORRECTED: Global atomic time, NO WAVES
    // T_atomic = (N * S * eta) / A_eff
    size_t total_atomics = N * S * eta;
    p.T_atomic_sec = static_cast<double>(total_atomics) / p.A_eff;

    // Memory time: 2 * cache_line * unique_lines / BW_scatter
    constexpr size_t LINE_SIZE = 128;
    double bytes = 2.0 * LINE_SIZE * metrics.total_cache_lines;
    p.T_memory_sec = bytes / dev.BW_scatter;

    // Total (Phase A negligible for now)
    p.T_kernel_sec = std::max(p.T_atomic_sec, p.T_memory_sec);
    p.throughput_Mpts = (N / p.T_kernel_sec) / 1e6;

    p.bottleneck = (p.T_atomic_sec > p.T_memory_sec) ? "Atomic" : "Memory";

    return p;
}

// ============================================================================
// CORRECTED: Calibration with TeamPolicy and FIXED working set
// ============================================================================

template <typename ExecSpace>
double calibrate_A_L2_unique(size_t N, int runs) {
    using MemSpace = typename ExecSpace::memory_space;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
    using Member = typename TeamPolicy::member_type;

    // FIXED: L2-resident array (does NOT change with contention level)
    size_t array_size = 1024 * 1024;  // ~8MB
    Kokkos::View<double*, MemSpace> data("data", array_size);
    Kokkos::deep_copy(data, 0.0);

    int team_size = 4;
    int vector_length = 1;
    size_t n_teams = N / team_size;
    auto policy = TeamPolicy(n_teams, team_size, vector_length);

    ManualTimer timer;
    std::vector<double> rates;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        Kokkos::parallel_for("warmup", policy,
            KOKKOS_LAMBDA(const Member& team) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size), [&](int t) {
                    size_t idx = (team.league_rank() * team_size + t) % array_size;
                    Kokkos::atomic_add(&data(idx), 1.0);
                });
            });
        Kokkos::fence();
    }

    for (int run = 0; run < runs; ++run) {
        timer.start();
        Kokkos::parallel_for("bench", policy,
            KOKKOS_LAMBDA(const Member& team) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size), [&](int t) {
                    size_t idx = (team.league_rank() * team_size + t) % array_size;
                    Kokkos::atomic_add(&data(idx), 1.0);
                });
            });
        double t = timer.stop();
        rates.push_back(N / t);
    }

    std::sort(rates.begin(), rates.end());
    return rates[runs / 2];  // Median
}

template <typename ExecSpace>
double calibrate_A_L2_hot(size_t N, int runs) {
    using MemSpace = typename ExecSpace::memory_space;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
    using Member = typename TeamPolicy::member_type;

    // Single cache line - full contention
    Kokkos::View<double*, MemSpace> data("data", 8);
    Kokkos::deep_copy(data, 0.0);

    int team_size = 4;
    int vector_length = 1;
    size_t n_teams = N / team_size;
    auto policy = TeamPolicy(n_teams, team_size, vector_length);

    ManualTimer timer;
    std::vector<double> rates;

    for (int i = 0; i < 5; ++i) {
        Kokkos::parallel_for("warmup", policy,
            KOKKOS_LAMBDA(const Member& team) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size), [&](int) {
                    Kokkos::atomic_add(&data(0), 1.0);
                });
            });
        Kokkos::fence();
    }

    for (int run = 0; run < runs; ++run) {
        timer.start();
        Kokkos::parallel_for("bench", policy,
            KOKKOS_LAMBDA(const Member& team) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size), [&](int) {
                    Kokkos::atomic_add(&data(0), 1.0);
                });
            });
        double t = timer.stop();
        rates.push_back(N / t);
    }

    std::sort(rates.begin(), rates.end());
    return rates[runs / 2];
}

template <typename ExecSpace>
double calibrate_BW_scatter(size_t N, int runs) {
    using MemSpace = typename ExecSpace::memory_space;
    using complex_type = Kokkos::complex<double>;

    size_t grid_size = 64 * 64 * 64;
    Kokkos::View<complex_type*, MemSpace> grid("grid", grid_size);
    Kokkos::View<size_t*, MemSpace> indices("idx", N);

    // Strided pattern
    Kokkos::parallel_for("init", Kokkos::RangePolicy<ExecSpace>(0, N),
        KOKKOS_LAMBDA(size_t i) { indices(i) = (i * 7) % grid_size; });
    Kokkos::fence();

    ManualTimer timer;
    std::vector<double> bw;

    for (int i = 0; i < 5; ++i) {
        Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(size_t i) {
                Kokkos::atomic_add(&grid(indices(i)).real(), 1.0);
                Kokkos::atomic_add(&grid(indices(i)).imag(), 1.0);
            });
        Kokkos::fence();
    }

    for (int run = 0; run < runs; ++run) {
        timer.start();
        Kokkos::parallel_for("bench", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(size_t i) {
                Kokkos::atomic_add(&grid(indices(i)).real(), 1.0);
                Kokkos::atomic_add(&grid(indices(i)).imag(), 1.0);
            });
        double t = timer.stop();
        size_t unique_lines = std::min(N, grid_size / 8);
        double bytes = 2.0 * 128 * unique_lines;
        bw.push_back(bytes / t);
    }

    std::sort(bw.begin(), bw.end());
    return bw[runs / 2];
}

// ============================================================================
// Main Benchmark
// ============================================================================

template <typename ExecSpace>
class AtomicScatterBenchmark {
public:
    static constexpr unsigned Dim = 3;
    using real_type = double;
    using complex_type = Kokkos::complex<real_type>;
    using MemSpace = typename ExecSpace::memory_space;

    using Mesh_t = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t = typename Mesh_t::DefaultCentering;
    using Field_t = ippl::Field<complex_type, Dim, Mesh_t, Centering_t>;
    using PLayout_t = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t = ippl::ParticleBase<PLayout_t>;

    AtomicScatterBenchmark(const BenchParams& params) : params_(params) {}

    void run() {
        print_header();

        DeviceConstants dev;
        get_device_info(dev);

        if (params_.calibrate) {
            calibrate_device(dev);
        }

        std::vector<std::tuple<int, double, double, double, double, ModelPrediction>> results;

        if (params_.sweep_w) {
            for (int w = 2; w <= 10; ++w) {
                double tol = std::pow(10.0, -(w - 1));
                auto [scatter_ms, halo_ms, metrics] = run_benchmark(tol, dev);
                auto pred = predict_time(dev, metrics, w, params_.n_particles());
                results.emplace_back(w, scatter_ms, halo_ms, metrics.m_L, metrics.phi_global, pred);
            }
        } else {
            ippl::NUFFT::ESKernel<real_type> kernel(params_.kernel_tol);
            int w = kernel.width();
            auto [scatter_ms, halo_ms, metrics] = run_benchmark(params_.kernel_tol, dev);
            auto pred = predict_time(dev, metrics, w, params_.n_particles());
            results.emplace_back(w, scatter_ms, halo_ms, metrics.m_L, metrics.phi_global, pred);
        }

        print_results(results, dev);
        write_csv(results, dev);
    }

    void get_device_info(DeviceConstants& dev) {
#ifdef KOKKOS_ENABLE_CUDA
        int d; cudaGetDevice(&d);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, d);
        dev.device_name = prop.name;
        dev.SM_count = prop.multiProcessorCount;
#elif defined(KOKKOS_ENABLE_HIP)
        int d; hipGetDevice(&d);
        hipDeviceProp_t prop; hipGetDeviceProperties(&prop, d);
        dev.device_name = prop.name;
        dev.SM_count = prop.multiProcessorCount;
#else
        dev.device_name = "CPU";
        dev.SM_count = Kokkos::DefaultExecutionSpace().concurrency();
#endif
    }

    void calibrate_device(DeviceConstants& dev) {
        if (ippl::Comm->rank() == 0) {
            std::cout << "\n=== Calibrating (CORRECTED: TeamPolicy, fixed array) ===\n";
            std::cout << "Device: " << dev.device_name << ", SMs: " << dev.SM_count << "\n";
        }

        size_t N = 10000000;
        int runs = 10;

        dev.A_L2_unique = calibrate_A_L2_unique<ExecSpace>(N, runs);
        dev.A_L2_hot = calibrate_A_L2_hot<ExecSpace>(N, runs);
        dev.BW_scatter = calibrate_BW_scatter<ExecSpace>(N, runs);

        if (ippl::Comm->rank() == 0) {
            std::cout << "  A_L2_unique:  " << std::scientific << dev.A_L2_unique << " atomics/sec\n";
            std::cout << "  A_L2_hot:     " << dev.A_L2_hot << " atomics/sec\n";
            std::cout << "  Ratio:        " << std::fixed << dev.A_L2_unique / dev.A_L2_hot << "x\n";
            std::cout << "  BW_scatter:   " << dev.BW_scatter / 1e9 << " GB/s\n";
            std::cout << "  P_in_flight:  " << dev.particles_in_flight() << "\n";
            std::cout << "===================================================\n\n";
        }
    }

    std::tuple<double, double, WindowedMetrics<Dim>> run_benchmark(double tol, const DeviceConstants& dev) {
        ippl::NUFFT::ESKernel<real_type> kernel(tol);
        int w = kernel.width();
        int nghost = w / 2 + 1;

        setup_domain();
        initialize(nghost);

        size_t n_local = bunch_->getLocalNum();

        // Compute windowed metrics
        size_t window_size = dev.particles_in_flight();
        auto metrics = compute_windowed_metrics<real_type, Dim>(
            R_.getView(), n_local, params_.n_grid, w, window_size);

        if (params_.verbose && ippl::Comm->rank() == 0) {
            std::cout << "  w=" << w << ", m_L=" << metrics.m_L
                      << " (phi_global=" << metrics.phi_global << " NOT USED)\n";
        }

        auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
        cfg.method = ippl::Interpolation::ScatterMethod::Atomic;

        ManualTimer timer;

        // Warmup
        for (int i = 0; i < params_.warmup_runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            Q_.scatter_kernel(*grid_, R_, kernel, cfg);
            grid_->accumulateHalo();
        }
        Kokkos::fence();

        // CORRECTED: Separate scatter vs halo timing
        std::vector<double> scatter_times, halo_times;

        for (int i = 0; i < params_.benchmark_runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            Kokkos::fence();

            timer.start();
            Q_.scatter_kernel(*grid_, R_, kernel, cfg);
            scatter_times.push_back(timer.stop());

            timer.start();
            grid_->accumulateHalo();
            halo_times.push_back(timer.stop());
        }

        cleanup();

        return {compute_stats(scatter_times).median_ms,
                compute_stats(halo_times).median_ms, metrics};
    }

    void setup_domain() {
        for (unsigned d = 0; d < Dim; ++d) n_grid_[d] = params_.n_grid;

        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d) domain[d] = ippl::Index(n_grid_[d]);

        std::array<bool, Dim> isParallel; isParallel.fill(true);
        layout_ = std::make_unique<ippl::FieldLayout<Dim>>(MPI_COMM_WORLD, domain, isParallel, true);

        for (unsigned d = 0; d < Dim; ++d) {
            origin_[d] = 0.0;
            hx_[d] = 2.0 * M_PI / static_cast<real_type>(n_grid_[d]);
        }
        mesh_ = std::make_unique<Mesh_t>(domain, hx_, origin_);
    }

    void initialize(int nghost) {
        grid_ = std::make_unique<Field_t>(*mesh_, *layout_, nghost);
        playout_ = std::make_unique<PLayout_t>(*layout_, *mesh_);
        bunch_ = std::make_unique<Bunch_t>(*playout_);

        bunch_->addAttribute(R_);
        bunch_->addAttribute(Q_);
        bunch_->setParticleBC(ippl::BC::PERIODIC);

        size_t n_local = params_.n_particles() / ippl::Comm->size();
        bunch_->create(n_local);

        auto R_view = R_.getView();
        Kokkos::Random_XorShift64_Pool<> pool(42 + ippl::Comm->rank());

        if (params_.distribution == "uniform") {
            Kokkos::parallel_for("init", n_local, KOKKOS_LAMBDA(size_t i) {
                auto gen = pool.get_state();
                for (unsigned d = 0; d < Dim; ++d)
                    R_view(i)[d] = gen.drand() * 2.0 * M_PI;
                pool.free_state(gen);
            });
        } else {
            Kokkos::parallel_for("init", n_local, KOKKOS_LAMBDA(size_t i) {
                auto gen = pool.get_state();
                for (unsigned d = 0; d < Dim; ++d) {
                    double u1 = gen.drand(), u2 = gen.drand();
                    double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-10)) * Kokkos::cos(2.0 * M_PI * u2);
                    R_view(i)[d] = M_PI + 0.3 * z;
                    while (R_view(i)[d] < 0) R_view(i)[d] += 2.0 * M_PI;
                    while (R_view(i)[d] >= 2.0 * M_PI) R_view(i)[d] -= 2.0 * M_PI;
                }
                pool.free_state(gen);
            });
        }

        auto Q_view = Q_.getView();
        Kokkos::parallel_for("init_Q", n_local, KOKKOS_LAMBDA(size_t i) {
            Q_view(i) = complex_type(1.0, 0.0);
        });
        Kokkos::fence();
    }

    void cleanup() {
        bunch_.reset(); playout_.reset(); grid_.reset(); mesh_.reset(); layout_.reset();
    }

    void print_header() {
        if (ippl::Comm->rank() != 0) return;
        std::cout << "\n================================================================\n"
                  << "     AtomicScatter Model (CORRECTED: windowed m_L, no waves)\n"
                  << "================================================================\n"
                  << "Grid: " << params_.n_grid << "^3, Particles: " << params_.n_particles()
                  << " (rho=" << params_.rho << ")\n"
                  << "Distribution: " << params_.distribution << "\n"
                  << "================================================================\n\n";
    }

    void print_results(
        const std::vector<std::tuple<int, double, double, double, double, ModelPrediction>>& results,
        const DeviceConstants& dev)
    {
        if (ippl::Comm->rank() != 0) return;

        std::cout << "\n=== Results (Scatter ONLY, halo separate) ===\n\n"
                  << std::left << std::setw(4) << "w"
                  << std::setw(14) << "Scatter(ms)"
                  << std::setw(12) << "Halo(ms)"
                  << std::setw(14) << "Pred(ms)"
                  << std::setw(10) << "Error%"
                  << std::setw(10) << "m_L"
                  << std::setw(14) << "phi_global"
                  << std::setw(10) << "Mpts/s"
                  << std::setw(10) << "Bottleneck" << "\n"
                  << std::string(106, '-') << "\n";

        for (const auto& [w, scatter_ms, halo_ms, m_L, phi_global, pred] : results) {
            double pred_ms = pred.T_kernel_sec * 1000;
            double err = 100.0 * std::abs(scatter_ms - pred_ms) / scatter_ms;
            double mpts = params_.n_particles() / (scatter_ms * 1e-3) / 1e6;

            std::cout << std::left << std::setw(4) << w
                      << std::fixed << std::setprecision(3) << std::setw(14) << scatter_ms
                      << std::setw(12) << halo_ms
                      << std::setw(14) << pred_ms
                      << std::setprecision(1) << std::setw(10) << err
                      << std::setprecision(2) << std::setw(10) << m_L
                      << std::setprecision(0) << std::setw(14) << phi_global
                      << std::setprecision(1) << std::setw(10) << mpts
                      << std::setw(10) << pred.bottleneck << "\n";
        }

        std::cout << "\nModel: T = (N*S*eta) / A_eff(m_L)  [NO WAVES]\n"
                  << "  A_eff(m_L) = (1/A_L2_unique + (m_L-1)/A_L2_hot)^(-1)\n"
                  << "  A_L2_unique = " << std::scientific << dev.A_L2_unique << "\n"
                  << "  A_L2_hot    = " << dev.A_L2_hot << "\n"
                  << "  BW_scatter  = " << std::fixed << dev.BW_scatter / 1e9 << " GB/s\n";
    }

    void write_csv(
        const std::vector<std::tuple<int, double, double, double, double, ModelPrediction>>& results,
        const DeviceConstants& dev)
    {
        if (ippl::Comm->rank() != 0) return;

        std::ofstream out(params_.output_prefix + "_v2.csv");
        out << "w,n_particles,scatter_ms,halo_ms,pred_ms,error_pct,m_L,phi_global,"
            << "A_eff,T_atomic_ms,T_mem_ms,bottleneck,Mpts_meas,Mpts_pred\n";

        for (const auto& [w, scatter_ms, halo_ms, m_L, phi_global, pred] : results) {
            double pred_ms = pred.T_kernel_sec * 1000;
            double err = 100.0 * std::abs(scatter_ms - pred_ms) / scatter_ms;
            double mpts_meas = params_.n_particles() / (scatter_ms * 1e-3) / 1e6;

            out << w << "," << params_.n_particles() << ","
                << std::fixed << std::setprecision(4) << scatter_ms << ","
                << halo_ms << "," << pred_ms << ","
                << std::setprecision(1) << err << ","
                << std::setprecision(3) << m_L << "," << phi_global << ","
                << std::scientific << pred.A_eff << ","
                << std::fixed << pred.T_atomic_sec * 1000 << ","
                << pred.T_memory_sec * 1000 << ","
                << pred.bottleneck << "," << mpts_meas << "," << pred.throughput_Mpts << "\n";
        }
        out.close();
        std::cout << "\nWrote: " << params_.output_prefix << "_v2.csv\n";
    }

private:
    BenchParams params_;
    ippl::Vector<std::size_t, Dim> n_grid_;
    ippl::Vector<real_type, Dim> origin_, hx_;

    std::unique_ptr<ippl::FieldLayout<Dim>> layout_;
    std::unique_ptr<Mesh_t> mesh_;
    std::unique_ptr<Field_t> grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t> bunch_;

    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R_;
    ippl::ParticleAttrib<complex_type> Q_;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        auto params = parse_args(argc, argv);
        AtomicScatterBenchmark<Kokkos::DefaultExecutionSpace> bench(params);
        bench.run();
    }
    ippl::finalize();
    return 0;
}