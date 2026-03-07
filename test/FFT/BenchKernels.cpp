/**
 * @file BenchmarkRoofline.cpp
 * @brief Throughput benchmark for scatter/gather kernels
 *
 * Generates data for:
 *   Plot 1: Kernel variant comparison at fixed parameters
 *   Plot 2: Throughput vs kernel width w (accuracy sensitivity)
 *
 * For roofline analysis, use Nsight Compute with --set roofline:
 *   ncu --set roofline ./BenchmarkRoofline --ncu-mode
 *
 * Usage: ./BenchmarkRoofline [options]
 *   --grid N        Grid size per dimension (default: 256)
 *   --rho R         Particles per grid point (default: 10)
 *   --tol T         Kernel tolerance (default: 1e-6, gives w~7)
 *   --warmup N      Number of warmup runs (default: 5)
 *   --runs N        Number of benchmark runs (default: 20)
 *   --output FILE   Output CSV file prefix (default: benchmark)
 *   --ncu-mode      Single run mode for Nsight Compute profiling
 *   --dist D        Particle distribution: uniform, clustered (default: uniform)
 *   --real          Use real-valued field and particles instead of complex
 *   --method M      Only benchmark scatter method M: Atomic, Tiled, OutputFocused
 *                   (default: all).  Tile sizes and team params are loaded from
 *                   the TileSizeCache CSV (IPPL_TILE_CSV env var or auto-discovered)
 *                   for the chosen method; auto-method-selection is suppressed.
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

using namespace ippl;

// ============================================================================
// Manual Timer Class
// ============================================================================

class ManualTimer {
public:
    using clock_type = std::chrono::high_resolution_clock;
    using time_point = clock_type::time_point;

    void start() {
        Kokkos::fence();
        start_time_ = clock_type::now();
    }

    double stop() {
        Kokkos::fence();
        auto end_time = clock_type::now();
        auto duration = std::chrono::duration<double>(end_time - start_time_);
        return duration.count();
    }

private:
    time_point start_time_;
};

// ============================================================================
// Benchmark Parameters
// ============================================================================

struct BenchParams {
    int n_grid = 16;
    double rho = 10.0;
    double kernel_tol = 1e-6;
    int warmup_runs = 3;
    int benchmark_runs = 5;
    std::string output_prefix = "benchmark";
    std::string distribution = "uniform";
    bool verbose = false;
    bool ncu_mode = false;
    bool use_real = false;          // use real-valued field/particles
    std::string method_filter = ""; // if non-empty, only run this scatter method

    size_t n_particles() const {
        return static_cast<size_t>(rho * n_grid * n_grid * n_grid);
    }
};

BenchParams parse_bench_args(int argc, char* argv[]) {
    BenchParams params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--grid" || arg == "-n") && i + 1 < argc) {
            params.n_grid = std::atoi(argv[++i]);
        } else if (arg == "--rho" && i + 1 < argc) {
            params.rho = std::atof(argv[++i]);
        } else if ((arg == "--tol" || arg == "-t") && i + 1 < argc) {
            params.kernel_tol = std::atof(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            params.warmup_runs = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            params.benchmark_runs = std::atoi(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            params.output_prefix = argv[++i];
        } else if (arg == "--dist" && i + 1 < argc) {
            params.distribution = argv[++i];
        } else if (arg == "--ncu-mode") {
            params.ncu_mode = true;
            params.warmup_runs = 1;
            params.benchmark_runs = 1;
        } else if (arg == "--real") {
            params.use_real = true;
        } else if (arg == "--method" && i + 1 < argc) {
            params.method_filter = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        }
    }
    return params;
}

// ============================================================================
// Statistics Helper
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

    if (stats.count == 0) return stats;

    std::vector<double> times_ms(stats.count);
    for (size_t i = 0; i < stats.count; ++i)
        times_ms[i] = times_sec[i] * 1000.0;

    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    stats.mean_ms = sum / stats.count;

    double sq_sum = 0.0;
    for (double t : times_ms)
        sq_sum += (t - stats.mean_ms) * (t - stats.mean_ms);
    stats.stddev_ms = (stats.count > 1) ? std::sqrt(sq_sum / (stats.count - 1)) : 0.0;

    stats.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    stats.max_ms = *std::max_element(times_ms.begin(), times_ms.end());

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    if (stats.count % 2 == 0)
        stats.median_ms = (sorted[stats.count/2 - 1] + sorted[stats.count/2]) / 2.0;
    else
        stats.median_ms = sorted[stats.count/2];

    return stats;
}

// ============================================================================
// Throughput Metrics
// ============================================================================

struct ThroughputMetrics {
    std::string kernel_name;
    std::string operation;
    std::string distribution;
    std::string value_type;         // NEW: "complex" or "real"
    int kernel_width;
    double tolerance;
    size_t n_particles;
    size_t n_grid;
    double rho;

    TimingStats total_stats;
    TimingStats kernel_only_stats;
    TimingStats sort_stats;

    std::vector<double> total_times_sec;
    std::vector<double> kernel_times_sec;
    std::vector<double> sort_times_sec;

    double throughput_Mpts_per_sec() const {
        return (n_particles / (total_stats.mean_ms * 1e-3)) / 1e6;
    }

    double time_per_point_ns() const {
        return (total_stats.mean_ms * 1e6) / n_particles;
    }
};

// ============================================================================
// Throughput Benchmark Implementation
// ============================================================================

// ValueT is either Kokkos::complex<double> or double
template <typename ExecSpace, typename ValueT>
class ThroughputBenchmark {
public:
    static constexpr unsigned Dim = 3;
    using real_type    = double;
    using value_type   = ValueT;                        // NEW: templated value type
    using MemSpace     = typename ExecSpace::memory_space;

    using Mesh_t       = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t  = typename Mesh_t::DefaultCentering;
    using Field_t      = ippl::Field<value_type, Dim, Mesh_t, Centering_t>;  // uses ValueT
    using PLayout_t    = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t      = ippl::ParticleBase<PLayout_t>;

    // Helper: string tag shown in output and CSV
    static constexpr bool is_complex = !std::is_same_v<ValueT, real_type>;
    static const char* value_type_str() { return is_complex ? "complex" : "real"; }

    // Zero-value initialiser that works for both real and complex
    static constexpr KOKKOS_INLINE_FUNCTION value_type zero() { return value_type(0); }
    static constexpr KOKKOS_INLINE_FUNCTION value_type one()  { return value_type(1); }

    ThroughputBenchmark(const BenchParams& params)
        : params_(params)
        , kernel_(params.kernel_tol) {}

    void run() {
        print_header();

        std::vector<ThroughputMetrics> results;

        run_all_kernels(kernel_, results);

        if (!params_.ncu_mode) {
            std::vector<double> tolerances = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9};
            for (double tol : tolerances) {
                if (true || std::abs(tol - params_.kernel_tol) > 1e-15) {
                    ippl::nufft::ESKernel<real_type> sweep_kernel(tol);
                    run_all_kernels(sweep_kernel, results);
                }
            }
        }

        write_csv(results);
        write_raw_csv(results);
        print_summary(results);
    }

    void print_header() {
        if (ippl::Comm->rank() != 0) return;

        int w = kernel_.width();
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "     Throughput Benchmark for Scatter/Gather Kernels\n";
        std::cout << "================================================================\n";
        std::cout << "Grid size:       " << params_.n_grid << "^3 = "
                  << (params_.n_grid * params_.n_grid * params_.n_grid) << " points\n";
        std::cout << "Particles/grid:  " << params_.rho << "\n";
        std::cout << "Total particles: " << params_.n_particles() << "\n";
        std::cout << "Distribution:    " << params_.distribution << "\n";
        std::cout << "Value type:      " << value_type_str() << "\n";
        if (!params_.method_filter.empty())
            std::cout << "Method filter:   " << params_.method_filter << " (auto-selection suppressed)\n";
        std::cout << "Tolerance:       " << params_.kernel_tol << "\n";
        std::cout << "Kernel width:    " << w << "\n";
        std::cout << "Warmup runs:     " << params_.warmup_runs << "\n";
        std::cout << "Benchmark runs:  " << params_.benchmark_runs << "\n";
        if (params_.ncu_mode)
            std::cout << "Mode:            NCU profiling (single run)\n";
        std::cout << "================================================================\n\n";
    }

    // Returns true if `name` matches the method_filter (or filter is empty).
    bool method_matches(const std::string& name) const {
        return params_.method_filter.empty() || params_.method_filter == name;
    }

    void run_all_kernels(const ippl::nufft::ESKernel<real_type>& kernel,
                         std::vector<ThroughputMetrics>& results) {
        int w = kernel.width();
        int nghost = w / 2 + 1;

        if (ippl::Comm->rank() == 0 && params_.verbose)
            std::cout << "Running benchmarks with w=" << w << "\n";

        setup_domain(nghost);
        initialize(kernel, nghost);

        size_t n_particles = bunch_->getLocalNum();

        // ===== SCATTER =====
        // When --method is given, lock_method=true prevents TileSizeCache from
        // overriding the explicitly chosen method.  Tile sizes and team params
        // are still loaded from the cache for the chosen method (resolve_config).
        // When no --method is given, all methods are benchmarked and cache-based
        // auto-selection is also suppressed (lock_method=true per method), so
        // each entry reflects the performance of exactly that method.
        if (method_matches("Atomic")) {
            auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
            cfg.lock_method = true;
            results.push_back(benchmark_scatter("Atomic", cfg, kernel, nghost, n_particles));
        }
        if (method_matches("Tiled")) {
            auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
            cfg.lock_method = true;
            // Tile sizes are loaded from cache when available; the conservative
            // default (tile=2) is used otherwise.  clamp_tile_to_shmem ensures
            // the launch never exceeds shared-memory limits.
            results.push_back(benchmark_scatter("Tiled", cfg, kernel, nghost, n_particles));
        }
        if (method_matches("OutputFocused")) {
            auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
            cfg.lock_method = true;
            // GridParallelScatter scratch ∝ team_size × htot.  The HIP/CUDA
            // execution-space defaults (team_size=64) cause shared-memory
            // exhaustion even at the minimum tile=(1,1,1) for large W.
            // team_size=1 is the safe starting point; the cache will override
            // this with the profiled optimum (e.g. 1–4 warps from TileSweep).
            // clamp_tile_to_shmem will further halve team_size if needed.
            cfg.team_size = 1;
            results.push_back(benchmark_scatter("OutputFocused", cfg, kernel, nghost, n_particles));
        }

        // ===== GATHER =====
        {
            auto cfg = ippl::Interpolation::GatherConfig<Dim>::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::Atomic;
            results.push_back(benchmark_gather("Direct", cfg, kernel, nghost, n_particles));
        }
        {
            auto cfg = ippl::Interpolation::GatherConfig<Dim>::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::AtomicSort;
            results.push_back(benchmark_gather("Sorted", cfg, kernel, nghost, n_particles));
        }

        cleanup();
    }

    ThroughputMetrics benchmark_scatter(const std::string& name,
                                         const ippl::Interpolation::ScatterConfig<Dim>& cfg,
                                         const ippl::nufft::ESKernel<real_type>& kernel,
                                         int /*nghost*/,
                                         size_t n_particles) {
        if (ippl::Comm->rank() == 0 && params_.verbose)
            std::cout << "  Benchmarking scatter: " << name << "\n";

        ManualTimer timer;

        for (int i = 0; i < params_.warmup_runs; ++i) {
            *grid_ = zero();
            Q_.scatter_kernel(*grid_, R_, kernel, cfg);
            grid_->accumulateHalo();
        }
        Kokkos::fence();

        std::vector<double> total_times;
        total_times.reserve(params_.benchmark_runs);

        for (int i = 0; i < params_.benchmark_runs; ++i) {
            *grid_ = zero();
            Kokkos::fence();

            timer.start();
            Q_.scatter_kernel(*grid_, R_, kernel, cfg);
            grid_->accumulateHalo();
            total_times.push_back(timer.stop());
        }

        return make_metrics(name, "scatter", kernel, n_particles, total_times);
    }

    ThroughputMetrics benchmark_gather(const std::string& name,
                                        const ippl::Interpolation::GatherConfig<Dim>& cfg,
                                        const ippl::nufft::ESKernel<real_type>& kernel,
                                        int /*nghost*/,
                                        size_t n_particles) {
        if (ippl::Comm->rank() == 0 && params_.verbose)
            std::cout << "  Benchmarking gather: " << name << "\n";

        ManualTimer timer;

        for (int i = 0; i < params_.warmup_runs; ++i) {
            Q_result_ = zero();
            Q_result_.gather(*grid_, R_, kernel, false, cfg);
        }
        Kokkos::fence();

        std::vector<double> total_times;
        total_times.reserve(params_.benchmark_runs);

        for (int i = 0; i < params_.benchmark_runs; ++i) {
            Q_result_ = zero();
            Kokkos::fence();

            timer.start();
            Q_result_.gather(*grid_, R_, kernel, false, cfg);
            total_times.push_back(timer.stop());
        }

        return make_metrics(name, "gather", kernel, n_particles, total_times);
    }

    // -----------------------------------------------------------------------
    // Shared metric builder (avoids duplication between scatter/gather)       NEW helper
    // -----------------------------------------------------------------------
    ThroughputMetrics make_metrics(const std::string& name,
                                    const std::string& op,
                                    const ippl::nufft::ESKernel<real_type>& kernel,
                                    size_t n_particles,
                                    const std::vector<double>& total_times) {
        ThroughputMetrics m;
        m.kernel_name      = name;
        m.operation        = op;
        m.distribution     = params_.distribution;
        m.value_type       = value_type_str();           // NEW
        m.kernel_width     = kernel.width();
        m.tolerance        = std::pow(10.0, -(kernel.width() - 1));
        m.n_particles      = n_particles;
        m.n_grid           = params_.n_grid;
        m.rho              = params_.rho;
        m.total_times_sec  = total_times;
        m.total_stats      = compute_stats(total_times);
        m.kernel_only_stats = m.total_stats;
        m.sort_stats        = TimingStats{};
        return m;
    }

    void setup_domain(int /*nghost*/) {
        for (unsigned d = 0; d < Dim; ++d)
            n_grid_[d] = params_.n_grid;

        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d)
            domain[d] = ippl::Index(n_grid_[d]);

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

    void initialize(const ippl::nufft::ESKernel<real_type>& /*kernel*/, int nghost) {
        grid_    = std::make_unique<Field_t>(*mesh_, *layout_, nghost);
        playout_ = std::make_unique<PLayout_t>(*layout_, *mesh_);
        bunch_   = std::make_unique<Bunch_t>(*playout_);

        bunch_->addAttribute(R_);
        bunch_->addAttribute(Q_);
        bunch_->addAttribute(Q_result_);
        bunch_->setParticleBC(ippl::BC::PERIODIC);

        size_t n_local = params_.n_particles() / ippl::Comm->size();
        bunch_->create(n_local);

        auto R_view = R_.getView();
        Kokkos::Random_XorShift64_Pool<> rand_pool(42 + ippl::Comm->rank());

        if (params_.distribution == "uniform") {
            Kokkos::parallel_for("init_uniform", n_local,
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d)
                        R_view(i)[d] = gen.drand() * 2.0 * M_PI;
                    rand_pool.free_state(gen);
                });
        } else if (params_.distribution == "clustered") {
            Kokkos::parallel_for("init_clustered", n_local,
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        double u1 = gen.drand();
                        double u2 = gen.drand();
                        double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-10))
                                   * Kokkos::cos(2.0 * M_PI * u2);
                        R_view(i)[d] = M_PI + 0.3 * z;
                        while (R_view(i)[d] < 0)          R_view(i)[d] += 2.0 * M_PI;
                        while (R_view(i)[d] >= 2.0 * M_PI) R_view(i)[d] -= 2.0 * M_PI;
                    }
                    rand_pool.free_state(gen);
                });
        }

        // Initialise particle source values to 1
        auto Q_view = Q_.getView();
        Kokkos::parallel_for("init_values", n_local,
            KOKKOS_LAMBDA(const size_t i) {
                Q_view(i) = one();                       // works for real and complex
            });

        *grid_ = one();                                  // works for real and complex
        Kokkos::fence();
    }

    void cleanup() {
        bunch_.reset();
        playout_.reset();
        grid_.reset();
        mesh_.reset();
        layout_.reset();
    }

    void write_csv(const std::vector<ThroughputMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_throughput.csv";
        std::ofstream out(filename);

        // value_type column added
        out << "kernel,operation,distribution,value_type,width,tolerance,n_particles,n_grid,rho,"
            << "total_mean_ms,total_stddev_ms,total_min_ms,total_max_ms,total_median_ms,"
            << "kernel_mean_ms,kernel_stddev_ms,"
            << "sort_mean_ms,sort_stddev_ms,"
            << "throughput_Mpts_per_s,time_per_pt_ns\n";

        for (const auto& m : results) {
            out << m.kernel_name << ","
                << m.operation << ","
                << m.distribution << ","
                << m.value_type << ","                   // NEW column
                << m.kernel_width << ","
                << std::scientific << std::setprecision(1) << m.tolerance << ","
                << m.n_particles << ","
                << m.n_grid << ","
                << std::fixed << std::setprecision(1) << m.rho << ","
                << std::setprecision(4) << m.total_stats.mean_ms << ","
                << m.total_stats.stddev_ms << ","
                << m.total_stats.min_ms << ","
                << m.total_stats.max_ms << ","
                << m.total_stats.median_ms << ","
                << m.kernel_only_stats.mean_ms << ","
                << m.kernel_only_stats.stddev_ms << ","
                << m.sort_stats.mean_ms << ","
                << m.sort_stats.stddev_ms << ","
                << std::setprecision(2) << m.throughput_Mpts_per_sec() << ","
                << std::setprecision(2) << m.time_per_point_ns() << "\n";
        }

        out.close();
        std::cout << "\nWrote results to: " << filename << "\n";
    }

    void write_raw_csv(const std::vector<ThroughputMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_raw.csv";
        std::ofstream out(filename);

        out << "kernel,operation,distribution,value_type,width,run,time_sec\n";  // NEW column

        for (const auto& m : results) {
            for (size_t i = 0; i < m.total_times_sec.size(); ++i) {
                out << m.kernel_name << ","
                    << m.operation << ","
                    << m.distribution << ","
                    << m.value_type << ","               // NEW column
                    << m.kernel_width << ","
                    << i << ","
                    << std::scientific << std::setprecision(9) << m.total_times_sec[i] << "\n";
            }
        }

        out.close();
        std::cout << "Wrote raw measurements to: " << filename << "\n";
    }

    void print_summary(const std::vector<ThroughputMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        int target_w = kernel_.width();

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "        Results Summary (w=" << target_w << ", "
                  << params_.distribution << " distribution, "
                  << value_type_str() << ")\n";              // NEW: show value type
        std::cout << "================================================================\n";

        auto print_section = [&](const std::string& op, const std::string& baseline_name) {
            std::cout << "\n" << (op == "scatter" ? "SCATTER (type-1 spreading)" : "GATHER (type-2 interpolation)") << ":\n";
            std::cout << std::left  << std::setw(16) << "Kernel"
                      << std::right << std::setw(12) << "Total (ms)"
                      << std::setw(12) << "Stddev"
                      << std::setw(14) << "Mpts/s"
                      << std::setw(12) << "ns/pt" << "\n";
            std::cout << std::string(66, '-') << "\n";

            double baseline = 0.0;
            for (const auto& m : results) {
                if (m.operation != op || m.kernel_width != target_w) continue;
                if (m.kernel_name == baseline_name) baseline = m.total_stats.mean_ms;
                double speedup = (baseline > 0) ? baseline / m.total_stats.mean_ms : 1.0;
                std::cout << std::left  << std::setw(16) << m.kernel_name
                          << std::right << std::fixed
                          << std::setw(12) << std::setprecision(3) << m.total_stats.mean_ms
                          << std::setw(12) << std::setprecision(3) << m.total_stats.stddev_ms
                          << std::setw(14) << std::setprecision(1) << m.throughput_Mpts_per_sec()
                          << std::setw(12) << std::setprecision(2) << m.time_per_point_ns()
                          << " (" << std::setprecision(2) << speedup << "x)\n";
            }
        };

        print_section("scatter", "Atomic");
        print_section("gather",  "Direct");

        std::cout << "\n";

        if (!params_.ncu_mode) {
            std::cout << "For roofline analysis, run with Nsight Compute:\n";
            std::cout << "  ncu --set roofline ./BenchmarkRoofline --ncu-mode\n\n";
        }
    }

    BenchParams params_;
    ippl::nufft::ESKernel<real_type> kernel_;

    ippl::Vector<std::size_t, Dim> n_grid_;
    ippl::Vector<real_type, Dim>   origin_;
    ippl::Vector<real_type, Dim>   hx_;

    std::unique_ptr<ippl::FieldLayout<Dim>> layout_;
    std::unique_ptr<Mesh_t>    mesh_;
    std::unique_ptr<Field_t>   grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t>   bunch_;

    // Particle attributes — value_type is real or complex depending on template arg
    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R_;
    ippl::ParticleAttrib<value_type>                   Q_;        // was always complex
    ippl::ParticleAttrib<value_type>                   Q_result_; // was always complex
};

// ============================================================================
// Main — dispatch to the right template instantiation
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        auto params = parse_bench_args(argc, argv);

        if (params.use_real) {
            // Pure real benchmark
            ThroughputBenchmark<Kokkos::DefaultExecutionSpace, double> benchmark(params);
            benchmark.run();
        } else {
            // Complex benchmark (original behaviour)
            ThroughputBenchmark<Kokkos::DefaultExecutionSpace,
                                Kokkos::complex<double>> benchmark(params);
            benchmark.run();
        }
    }

    ippl::finalize();
    return EXIT_SUCCESS;
}