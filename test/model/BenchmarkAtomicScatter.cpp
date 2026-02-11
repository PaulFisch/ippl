/**
 * @file BenchmarkAtomicScatter.cpp
 * @brief Performance model calibration and validation for AtomicScatter kernel
 *
 * This benchmark:
 *   1. Measures actual AtomicScatter kernel performance with ESKernel
 *   2. Computes collision factor φ from particle distribution
 *   3. Validates the performance model predictions
 *
 * Usage: ./BenchmarkAtomicScatter [options]
 *   --grid N        Grid size per dimension (default: 64)
 *   --rho R         Particles per grid point (default: 10)
 *   --tol T         Kernel tolerance (default: 1e-6, gives w~7)
 *   --warmup N      Number of warmup runs (default: 5)
 *   --runs N        Number of benchmark runs (default: 20)
 *   --output FILE   Output CSV file prefix (default: atomic_scatter)
 *   --sweep-w       Sweep kernel widths w=2..10
 *   --sweep-rho     Sweep particle densities
 *   --dist D        Distribution: uniform, clustered, sorted (default: uniform)
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
// Manual Timer Class (matches BenchmarkRoofline style)
// ============================================================================

class ManualTimer {
public:
    using clock_type = std::chrono::high_resolution_clock;
    using time_point = clock_type::time_point;

    void start() {
        Kokkos::fence();  // Ensure all previous GPU work is complete
        start_time_ = clock_type::now();
    }

    double stop() {
        Kokkos::fence();  // Ensure all GPU work from this section is complete
        auto end_time = clock_type::now();
        auto duration = std::chrono::duration<double>(end_time - start_time_);
        return duration.count();  // Returns seconds
    }

private:
    time_point start_time_;
};

// ============================================================================
// Benchmark Parameters
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
    bool sweep_rho = false;
    bool calibrate = false;
    
    size_t n_particles() const {
        return static_cast<size_t>(rho * n_grid * n_grid * n_grid);
    }
};

BenchParams parse_args(int argc, char* argv[]) {
    BenchParams p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--grid" || arg == "-n") && i + 1 < argc) {
            p.n_grid = std::atoi(argv[++i]);
        } else if (arg == "--rho" && i + 1 < argc) {
            p.rho = std::atof(argv[++i]);
        } else if ((arg == "--tol" || arg == "-t") && i + 1 < argc) {
            p.kernel_tol = std::atof(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            p.warmup_runs = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            p.benchmark_runs = std::atoi(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            p.output_prefix = argv[++i];
        } else if (arg == "--dist" && i + 1 < argc) {
            p.distribution = argv[++i];
        } else if (arg == "--sweep-w") {
            p.sweep_w = true;
        } else if (arg == "--sweep-rho") {
            p.sweep_rho = true;
        } else if (arg == "--calibrate") {
            p.calibrate = true;
        } else if (arg == "-v" || arg == "--verbose") {
            p.verbose = true;
        }
    }
    return p;
}

// ============================================================================
// Statistics Helper (matches BenchmarkRoofline style)
// ============================================================================

struct TimingStats {
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    double median_ms;
    size_t count;
};

TimingStats compute_stats(const std::vector<double>& times_sec) {
    TimingStats stats{};
    stats.count = times_sec.size();

    if (stats.count == 0) {
        return stats;
    }

    // Convert to milliseconds
    std::vector<double> times_ms(stats.count);
    for (size_t i = 0; i < stats.count; ++i) {
        times_ms[i] = times_sec[i] * 1000.0;
    }

    // Mean
    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    stats.mean_ms = sum / stats.count;

    // Stddev
    double sq_sum = 0.0;
    for (double t : times_ms) {
        sq_sum += (t - stats.mean_ms) * (t - stats.mean_ms);
    }
    stats.stddev_ms = (stats.count > 1) ? std::sqrt(sq_sum / (stats.count - 1)) : 0.0;

    // Min/Max
    stats.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    stats.max_ms = *std::max_element(times_ms.begin(), times_ms.end());

    // Median
    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    if (stats.count % 2 == 0) {
        stats.median_ms = (sorted[stats.count/2 - 1] + sorted[stats.count/2]) / 2.0;
    } else {
        stats.median_ms = sorted[stats.count/2];
    }

    return stats;
}

// ============================================================================
// Collision Factor Analysis
// ============================================================================

template <typename RView, unsigned Dim>
struct CollisionAnalysis {
    double phi_global;           // Global collision factor
    double phi_per_block_mean;   // Mean φ across blocks
    double phi_per_block_max;    // Max φ across blocks
    double phi_L;                // Cache-line collision factor
    size_t unique_grid_points;
    size_t total_updates;
    
    // Per-block statistics
    std::vector<double> phi_per_block;
};

template <typename real_type, unsigned Dim>
CollisionAnalysis<void, Dim> analyze_collisions(
    const Kokkos::View<ippl::Vector<real_type, Dim>*>& R_view,
    size_t n_particles,
    int n_grid,
    int kernel_width,
    int team_size = 4)
{
    CollisionAnalysis<void, Dim> result{};
    
    // Copy positions to host for analysis
    auto R_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R_view);
    
    double h = 2.0 * M_PI / n_grid;
    int S = 1;
    for (unsigned d = 0; d < Dim; ++d) S *= kernel_width;
    
    // Global analysis: count unique grid points touched
    std::unordered_set<size_t> global_touched;
    size_t total_updates = 0;
    
    // Per-block analysis
    size_t n_blocks = (n_particles + team_size - 1) / team_size;
    result.phi_per_block.resize(n_blocks, 0.0);
    
    for (size_t block = 0; block < n_blocks; ++block) {
        std::unordered_set<size_t> block_touched;
        size_t block_start = block * team_size;
        size_t block_end = std::min(block_start + team_size, n_particles);
        
        for (size_t i = block_start; i < block_end; ++i) {
            // Compute base grid index for this particle
            int base[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                base[d] = static_cast<int>(std::floor(R_host(i)[d] / h)) - kernel_width / 2;
            }
            
            // Enumerate all grid points in the kernel stencil
            if constexpr (Dim == 3) {
                for (int iz = 0; iz < kernel_width; ++iz) {
                    for (int iy = 0; iy < kernel_width; ++iy) {
                        for (int ix = 0; ix < kernel_width; ++ix) {
                            int gx = (base[0] + ix + n_grid) % n_grid;
                            int gy = (base[1] + iy + n_grid) % n_grid;
                            int gz = (base[2] + iz + n_grid) % n_grid;
                            size_t idx = gx + n_grid * (gy + n_grid * gz);
                            
                            global_touched.insert(idx);
                            block_touched.insert(idx);
                            total_updates++;
                        }
                    }
                }
            } else if constexpr (Dim == 2) {
                for (int iy = 0; iy < kernel_width; ++iy) {
                    for (int ix = 0; ix < kernel_width; ++ix) {
                        int gx = (base[0] + ix + n_grid) % n_grid;
                        int gy = (base[1] + iy + n_grid) % n_grid;
                        size_t idx = gx + n_grid * gy;
                        
                        global_touched.insert(idx);
                        block_touched.insert(idx);
                        total_updates++;
                    }
                }
            } else {
                for (int ix = 0; ix < kernel_width; ++ix) {
                    int gx = (base[0] + ix + n_grid) % n_grid;
                    global_touched.insert(gx);
                    block_touched.insert(gx);
                    total_updates++;
                }
            }
        }
        
        size_t block_updates = (block_end - block_start) * S;
        size_t block_unique = block_touched.size();
        result.phi_per_block[block] = (block_unique > 0) 
            ? static_cast<double>(block_updates) / block_unique 
            : 1.0;
    }
    
    result.unique_grid_points = global_touched.size();
    result.total_updates = total_updates;
    result.phi_global = static_cast<double>(total_updates) / result.unique_grid_points;
    
    // Per-block statistics
    double sum = 0, max_phi = 0;
    for (double phi : result.phi_per_block) {
        sum += phi;
        max_phi = std::max(max_phi, phi);
    }
    result.phi_per_block_mean = sum / n_blocks;
    result.phi_per_block_max = max_phi;
    
    // Cache-line collision factor (simplified: assume 128-byte lines, 16-byte complex<double>)
    constexpr size_t CACHE_LINE = 128;
    constexpr size_t ELEM_SIZE = sizeof(Kokkos::complex<real_type>);
    constexpr size_t ELEMS_PER_LINE = CACHE_LINE / ELEM_SIZE;
    
    std::unordered_set<size_t> cache_lines_touched;
    for (size_t idx : global_touched) {
        cache_lines_touched.insert(idx / ELEMS_PER_LINE);
    }
    result.phi_L = static_cast<double>(global_touched.size()) / cache_lines_touched.size();
    
    return result;
}

// ============================================================================
// Performance Model
// ============================================================================

struct DeviceConstants {
    double A_1 = 1e10;          // Unique-address atomic rate (atomics/sec)
    double A_inf = 1e8;         // Hotspot atomic rate (atomics/sec)
    double BW_HBM = 1e12;       // HBM bandwidth (bytes/sec)
    double P_eff = 1e12;        // Effective compute throughput (FLOP/sec)
    double t_barrier = 1e-7;    // Barrier cost per block (seconds)
    int SM_count = 80;          // Number of SMs
    int blocks_per_SM = 8;      // Achieved blocks per SM
};

struct ModelPrediction {
    double t_A_compute;         // Phase A compute time per particle
    double t_A_memory;          // Phase A memory time per particle
    double t_A;                 // Phase A total per particle
    double t_B_atomic;          // Phase B atomic time per particle
    double t_B_memory;          // Phase B memory time per particle  
    double t_B;                 // Phase B total per particle
    double t_total_per_particle;
    double T_kernel;            // Total kernel time
    double throughput_Mpts;     // Million particles per second
    std::string bottleneck;     // "Phase A", "Phase B atomic", "Phase B memory"
};

ModelPrediction predict_kernel_time(
    const DeviceConstants& dev,
    int kernel_width,
    int Dim,
    size_t N,
    double phi,
    double rho_residency = 0.5)
{
    ModelPrediction pred{};
    
    int S = 1;
    for (int d = 0; d < Dim; ++d) S *= kernel_width;
    
    // Phase A: Weight computation
    // FLOPs per particle: coordinate transform + kernel evaluation
    double F_xform = 5.0 * Dim;  // Coordinate transform
    double F_kern = 10.0 * kernel_width * Dim;  // Kernel evaluation (exp, sqrt, etc.)
    double F_A = F_xform + F_kern;
    
    // Memory for Phase A: read position (Dim * 8 bytes)
    double B_A = Dim * sizeof(double);
    
    pred.t_A_compute = F_A / dev.P_eff;
    pred.t_A_memory = B_A / dev.BW_HBM;
    pred.t_A = std::max(pred.t_A_compute, pred.t_A_memory);
    
    // Phase B: Atomic scatter
    // Effective atomic rate with collision factor
    double A_eff = 1.0 / (1.0 / dev.A_1 + (phi - 1.0) / dev.A_inf);
    
    pred.t_B_atomic = S / A_eff;
    
    // Memory for Phase B: read-modify-write grid values
    // Assume rho_residency fraction miss L2 cache
    double bytes_per_update = 2 * sizeof(Kokkos::complex<double>);  // Read + write
    double B_B = S * bytes_per_update * rho_residency;
    
    pred.t_B_memory = B_B / dev.BW_HBM;
    pred.t_B = std::max(pred.t_B_atomic, pred.t_B_memory);
    
    // Total time per particle
    pred.t_total_per_particle = pred.t_A + pred.t_B;
    
    // Kernel execution: particles processed in parallel across SMs
    int total_active_blocks = dev.SM_count * dev.blocks_per_SM;
    int team_size = 4;  // Typical team size
    double particles_per_wave = total_active_blocks * team_size;
    double n_waves = std::ceil(N / particles_per_wave);
    
    pred.T_kernel = n_waves * pred.t_total_per_particle;
    pred.throughput_Mpts = (N / pred.T_kernel) / 1e6;
    
    // Identify bottleneck
    if (pred.t_A > pred.t_B) {
        pred.bottleneck = "Phase A";
    } else if (pred.t_B_atomic > pred.t_B_memory) {
        pred.bottleneck = "Phase B atomic";
    } else {
        pred.bottleneck = "Phase B memory";
    }
    
    return pred;
}

// ============================================================================
// Atomic Rate Microbenchmarks (for calibration)
// ============================================================================

template <typename ExecSpace>
double calibrate_A1(size_t N, int num_runs) {
    // Measure atomic throughput with unique addresses (no collisions)
    using MemSpace = typename ExecSpace::memory_space;
    
    Kokkos::View<double*, MemSpace> data("data", N);
    Kokkos::deep_copy(data, 0.0);
    
    ManualTimer timer;
    std::vector<double> times;
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        Kokkos::parallel_for("warmup_A1", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(const size_t i) {
                Kokkos::atomic_add(&data(i), 1.0);
            });
        Kokkos::fence();
    }
    
    // Benchmark
    for (int run = 0; run < num_runs; ++run) {
        timer.start();
        Kokkos::parallel_for("bench_A1", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(const size_t i) {
                Kokkos::atomic_add(&data(i), 1.0);
            });
        times.push_back(timer.stop());
    }
    
    TimingStats s = compute_stats(times);
    return N / (s.median_ms * 1e-3);  // atomics/sec
}

template <typename ExecSpace>
double calibrate_Ainf(size_t N, int num_runs) {
    // Measure atomic throughput with all updates to same address (full contention)
    using MemSpace = typename ExecSpace::memory_space;
    
    Kokkos::View<double*, MemSpace> data("data", 1);
    Kokkos::deep_copy(data, 0.0);
    
    ManualTimer timer;
    std::vector<double> times;
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        Kokkos::parallel_for("warmup_Ainf", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(const size_t) {
                Kokkos::atomic_add(&data(0), 1.0);
            });
        Kokkos::fence();
    }
    
    // Benchmark  
    for (int run = 0; run < num_runs; ++run) {
        timer.start();
        Kokkos::parallel_for("bench_Ainf", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(const size_t) {
                Kokkos::atomic_add(&data(0), 1.0);
            });
        times.push_back(timer.stop());
    }
    
    TimingStats s = compute_stats(times);
    return N / (s.median_ms * 1e-3);  // atomics/sec
}

template <typename ExecSpace>
double calibrate_HBM_bandwidth(size_t N, int num_runs) {
    // Measure HBM bandwidth with streaming access
    using MemSpace = typename ExecSpace::memory_space;
    
    Kokkos::View<double*, MemSpace> src("src", N);
    Kokkos::View<double*, MemSpace> dst("dst", N);
    
    Kokkos::deep_copy(src, 1.0);
    
    ManualTimer timer;
    std::vector<double> times;
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        Kokkos::parallel_for("warmup_bw", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(const size_t i) {
                dst(i) = src(i);
            });
        Kokkos::fence();
    }
    
    // Benchmark
    for (int run = 0; run < num_runs; ++run) {
        timer.start();
        Kokkos::parallel_for("bench_bw", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(const size_t i) {
                dst(i) = src(i);
            });
        times.push_back(timer.stop());
    }
    
    TimingStats s = compute_stats(times);
    double bytes = 2.0 * N * sizeof(double);  // Read + write
    return bytes / (s.median_ms * 1e-3);  // bytes/sec
}

// ============================================================================
// Main Benchmark Class
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
        
        // Calibrate device constants if requested
        DeviceConstants dev;
        if (params_.calibrate) {
            calibrate_device(dev);
        }
        
        std::vector<std::tuple<int, double, double, double, ModelPrediction>> results;
        
        if (params_.sweep_w) {
            // Sweep kernel widths
            for (int w = 2; w <= 10; ++w) {
                double tol = std::pow(10.0, -(w - 1));
                auto [measured, phi] = run_single_benchmark(tol);
                auto pred = predict_kernel_time(dev, w, Dim, params_.n_particles(), phi);
                results.emplace_back(w, tol, measured, phi, pred);
            }
        } else if (params_.sweep_rho) {
            // Sweep particle densities
            ippl::NUFFT::ESKernel<real_type> kernel(params_.kernel_tol);
            int w = kernel.width();
            
            for (double rho : {1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0}) {
                BenchParams p = params_;
                p.rho = rho;
                auto [measured, phi] = run_single_benchmark(params_.kernel_tol, &p);
                auto pred = predict_kernel_time(dev, w, Dim, p.n_particles(), phi);
                results.emplace_back(w, params_.kernel_tol, measured, phi, pred);
            }
        } else {
            // Single run
            auto [measured, phi] = run_single_benchmark(params_.kernel_tol);
            ippl::NUFFT::ESKernel<real_type> kernel(params_.kernel_tol);
            auto pred = predict_kernel_time(dev, kernel.width(), Dim, params_.n_particles(), phi);
            results.emplace_back(kernel.width(), params_.kernel_tol, measured, phi, pred);
        }
        
        // Output results
        print_results(results, dev);
        write_csv(results, dev);
    }
    
    void calibrate_device(DeviceConstants& dev) {
        if (ippl::Comm->rank() == 0) {
            std::cout << "\n=== Calibrating Device Constants ===\n";
        }
        
        size_t N_calib = 10000000;
        int n_runs = 10;
        
        dev.A_1 = calibrate_A1<ExecSpace>(N_calib, n_runs);
        dev.A_inf = calibrate_Ainf<ExecSpace>(N_calib, n_runs);
        dev.BW_HBM = calibrate_HBM_bandwidth<ExecSpace>(N_calib, n_runs);
        
        if (ippl::Comm->rank() == 0) {
            std::cout << "  A_1 (unique):   " << std::scientific << dev.A_1 << " atomics/sec\n";
            std::cout << "  A_∞ (hotspot):  " << dev.A_inf << " atomics/sec\n";
            std::cout << "  BW_HBM:         " << dev.BW_HBM / 1e9 << " GB/s\n";
            std::cout << "  A_1/A_∞ ratio:  " << std::fixed << dev.A_1 / dev.A_inf << "x\n";
            std::cout << "================================\n\n";
        }
    }
    
    std::pair<double, double> run_single_benchmark(double tol, const BenchParams* override_params = nullptr) {
        const BenchParams& p = override_params ? *override_params : params_;
        
        ippl::NUFFT::ESKernel<real_type> kernel(tol);
        int w = kernel.width();
        int nghost = w / 2 + 1;
        
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "Running w=" << w << ", N=" << p.n_particles() << "\n";
        }
        
        // Setup
        setup_domain(p, nghost);
        initialize(p, kernel, nghost);
        
        size_t n_local = bunch_->getLocalNum();
        
        // Analyze collision factor
        auto collision = analyze_collisions<real_type, Dim>(
            R_.getView(), n_local, p.n_grid, w);
        
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "  φ_global = " << collision.phi_global 
                      << ", φ_block_mean = " << collision.phi_per_block_mean << "\n";
        }
        
        // Configure atomic scatter (matching BenchmarkRoofline style)
        auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
        cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
        
        ManualTimer timer;
        
        // Warmup runs (not timed)
        for (int i = 0; i < p.warmup_runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            Q_.scatter_kernel(*grid_, R_, kernel, cfg);
            grid_->accumulateHalo();
        }
        Kokkos::fence();  // Ensure warmup is complete
        
        // Benchmark runs with manual timing
        std::vector<double> times;
        times.reserve(p.benchmark_runs);
        
        for (int i = 0; i < p.benchmark_runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            Kokkos::fence();  // Ensure grid reset is complete
            
            timer.start();
            Q_.scatter_kernel(*grid_, R_, kernel, cfg);
            grid_->accumulateHalo();
            double elapsed = timer.stop();
            
            times.push_back(elapsed);
        }
        
        cleanup();
        
        TimingStats s = compute_stats(times);
        return {s.median_ms, collision.phi_global};  // Return ms and phi
    }
    
    void setup_domain(const BenchParams& p, int /*nghost*/) {
        for (unsigned d = 0; d < Dim; ++d) {
            n_grid_[d] = p.n_grid;
        }
        
        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d) {
            domain[d] = ippl::Index(n_grid_[d]);
        }
        
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        
        layout_ = std::make_unique<ippl::FieldLayout<Dim>>(
            MPI_COMM_WORLD, domain, isParallel, true);
        
        for (unsigned d = 0; d < Dim; ++d) {
            origin_[d] = 0.0;
            hx_[d] = 2.0 * M_PI / static_cast<real_type>(n_grid_[d]);
        }
        
        mesh_ = std::make_unique<Mesh_t>(domain, hx_, origin_);
    }
    
    void initialize(const BenchParams& p, 
                    const ippl::NUFFT::ESKernel<real_type>& /*kernel*/, 
                    int nghost) {
        grid_ = std::make_unique<Field_t>(*mesh_, *layout_, nghost);
        playout_ = std::make_unique<PLayout_t>(*layout_, *mesh_);
        bunch_ = std::make_unique<Bunch_t>(*playout_);
        
        bunch_->addAttribute(R_);
        bunch_->addAttribute(Q_);
        bunch_->setParticleBC(ippl::BC::PERIODIC);
        
        size_t n_local = p.n_particles() / ippl::Comm->size();
        bunch_->create(n_local);
        
        auto R_view = R_.getView();
        Kokkos::Random_XorShift64_Pool<> rand_pool(42 + ippl::Comm->rank());
        
        if (p.distribution == "uniform") {
            Kokkos::parallel_for("init_uniform", n_local,
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        R_view(i)[d] = gen.drand() * 2.0 * M_PI;
                    }
                    rand_pool.free_state(gen);
                });
        } else if (p.distribution == "clustered") {
            Kokkos::parallel_for("init_clustered", n_local,
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        double u1 = gen.drand();
                        double u2 = gen.drand();
                        double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-10))
                                   * Kokkos::cos(2.0 * M_PI * u2);
                        R_view(i)[d] = M_PI + 0.3 * z;
                        while (R_view(i)[d] < 0) R_view(i)[d] += 2.0 * M_PI;
                        while (R_view(i)[d] >= 2.0 * M_PI) R_view(i)[d] -= 2.0 * M_PI;
                    }
                    rand_pool.free_state(gen);
                });
        }
        
        auto Q_view = Q_.getView();
        Kokkos::parallel_for("init_Q", n_local,
            KOKKOS_LAMBDA(const size_t i) {
                Q_view(i) = complex_type(1.0, 0.0);
            });
        
        Kokkos::fence();
    }
    
    void cleanup() {
        bunch_.reset();
        playout_.reset();
        grid_.reset();
        mesh_.reset();
        layout_.reset();
    }
    
    void print_header() {
        if (ippl::Comm->rank() != 0) return;
        
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "     AtomicScatter Performance Model Validation\n";
        std::cout << "================================================================\n";
        std::cout << "Grid size:       " << params_.n_grid << "^3\n";
        std::cout << "Particles/grid:  " << params_.rho << "\n";
        std::cout << "Total particles: " << params_.n_particles() << "\n";
        std::cout << "Distribution:    " << params_.distribution << "\n";
        std::cout << "================================================================\n\n";
    }
    
    void print_results(
        const std::vector<std::tuple<int, double, double, double, ModelPrediction>>& results,
        const DeviceConstants& dev) 
    {
        if (ippl::Comm->rank() != 0) return;
        
        std::cout << "\n=== Results ===\n\n";
        std::cout << std::left << std::setw(6) << "w"
                  << std::setw(12) << "Measured"
                  << std::setw(12) << "Predicted"
                  << std::setw(10) << "Error"
                  << std::setw(10) << "φ"
                  << std::setw(14) << "Mpts/s"
                  << std::setw(16) << "Bottleneck" << "\n";
        std::cout << std::string(80, '-') << "\n";
        
        for (const auto& [w, tol, measured_ms, phi, pred] : results) {
            double error_pct = 100.0 * std::abs(measured_ms - pred.T_kernel * 1000) / measured_ms;
            double mpts = params_.n_particles() / (measured_ms * 1e-3) / 1e6;
            
            std::cout << std::left << std::setw(6) << w
                      << std::fixed << std::setprecision(3)
                      << std::setw(12) << measured_ms << " ms"
                      << std::setw(12) << (pred.T_kernel * 1000) << " ms"
                      << std::setw(10) << error_pct << "%"
                      << std::setw(10) << phi
                      << std::setw(14) << mpts
                      << std::setw(16) << pred.bottleneck << "\n";
        }
        
        std::cout << "\nDevice Constants Used:\n";
        std::cout << "  A_1 = " << std::scientific << dev.A_1 << " atomics/sec\n";
        std::cout << "  A_∞ = " << dev.A_inf << " atomics/sec\n";
        std::cout << "  BW  = " << dev.BW_HBM / 1e9 << " GB/s\n";
    }
    
    void write_csv(
        const std::vector<std::tuple<int, double, double, double, ModelPrediction>>& results,
        const DeviceConstants& dev)
    {
        if (ippl::Comm->rank() != 0) return;
        
        std::string filename = params_.output_prefix + "_model_validation.csv";
        std::ofstream out(filename);
        
        out << "w,tolerance,n_particles,n_grid,rho,distribution,"
            << "measured_ms,predicted_ms,error_pct,phi,"
            << "t_A_ns,t_B_atomic_ns,t_B_memory_ns,bottleneck,"
            << "throughput_Mpts,A_1,A_inf,BW_HBM\n";
        
        for (const auto& [w, tol, measured_ms, phi, pred] : results) {
            double error_pct = 100.0 * std::abs(measured_ms - pred.T_kernel * 1000) / measured_ms;
            double mpts = params_.n_particles() / (measured_ms * 1e-3) / 1e6;
            
            out << w << ","
                << std::scientific << tol << ","
                << params_.n_particles() << ","
                << params_.n_grid << ","
                << std::fixed << params_.rho << ","
                << params_.distribution << ","
                << std::setprecision(4) << measured_ms << ","
                << (pred.T_kernel * 1000) << ","
                << error_pct << ","
                << phi << ","
                << (pred.t_A * 1e9) << ","
                << (pred.t_B_atomic * 1e9) << ","
                << (pred.t_B_memory * 1e9) << ","
                << pred.bottleneck << ","
                << mpts << ","
                << std::scientific << dev.A_1 << ","
                << dev.A_inf << ","
                << dev.BW_HBM << "\n";
        }
        
        out.close();
        std::cout << "\nWrote results to: " << filename << "\n";
    }
    
private:
    BenchParams params_;
    
    ippl::Vector<std::size_t, Dim> n_grid_;
    ippl::Vector<real_type, Dim> origin_;
    ippl::Vector<real_type, Dim> hx_;
    
    std::unique_ptr<ippl::FieldLayout<Dim>> layout_;
    std::unique_ptr<Mesh_t> mesh_;
    std::unique_ptr<Field_t> grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t> bunch_;
    
    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R_;
    ippl::ParticleAttrib<complex_type> Q_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    
    {
        auto params = parse_args(argc, argv);
        AtomicScatterBenchmark<Kokkos::DefaultExecutionSpace> benchmark(params);
        benchmark.run();
    }
    
    ippl::finalize();
    return EXIT_SUCCESS;
}
