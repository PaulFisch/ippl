/**
 * @file BenchmarkTileSweep.cpp
 * @brief Performance benchmark sweeping tile size and kernel width
 *
 * Generates data for:
 *   Plot 1: Performance vs tile size (fixed kernel width)
 *   Plot 2: Performance vs kernel width (fixed tile size)
 *   Plot 3: Heatmap of performance across tile size × kernel width
 *
 * Usage: ./BenchmarkTileSweep [options]
 *   --grid N           Grid size per dimension (default: 256)
 *   --rho R            Particles per grid point (default: 10)
 *   --warmup N         Number of warmup runs (default: 3)
 *   --runs N           Number of benchmark runs (default: 10)
 *   --output FILE      Output CSV file prefix (default: tile_sweep)
 *   --min-tile T       Minimum tile size per dimension (default: 1)
 *   --max-tile T       Maximum tile size per dimension (default: 8)
 *   --min-width W      Minimum kernel width (default: 2)
 *   --max-width W      Maximum kernel width (default: 12)
 *   --dist D           Particle distribution: uniform, clustered (default: uniform)
 *   --ncu-mode         Single run mode for Nsight Compute profiling
 *   --real             Use real-valued field and particles instead of complex
 *   --optimize         Run simulated-annealing optimiser for rectangular tile sizes
 *   --sa-steps N       SA: number of annealing steps (default: 200)
 *   --sa-t0 T          SA: initial temperature (default: 5.0)
 *   --sa-alpha A       SA: cooling factor per step (default: 0.97)
 *   -v, --verbose      Verbose output
 */

#include "Ippl.h"
#include <Kokkos_Random.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

using namespace ippl;

// ============================================================================
// Manual Timer
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
        auto end = clock_type::now();
        return std::chrono::duration<double>(end - start_time_).count();
    }

private:
    time_point start_time_;
};

// ============================================================================
// Benchmark Parameters
// ============================================================================

struct BenchParams {
    int n_grid        = 128;
    double rho        = 1.0;
    int warmup_runs   = 3;
    int benchmark_runs = 5;
    std::string output_prefix = "tile_sweep";
    std::string distribution  = "uniform";
    bool verbose   = false;
    bool ncu_mode  = false;
    bool use_real  = false;   // NEW: real-valued field/particles
    bool optimize  = false;   // NEW: run SA optimiser

    // Sweep ranges
    int min_tile_size    = 1;
    int max_tile_size    = 8;
    int min_kernel_width = 2;
    int max_kernel_width = 8;

    // Simulated-annealing parameters
    int    sa_steps = 200;    // total number of annealing steps
    double sa_t0    = 5.0;    // initial temperature
    double sa_alpha = 0.97;   // geometric cooling factor

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
        } else if (arg == "--warmup" && i + 1 < argc) {
            params.warmup_runs = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            params.benchmark_runs = std::atoi(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            params.output_prefix = argv[++i];
        } else if (arg == "--dist" && i + 1 < argc) {
            params.distribution = argv[++i];
        } else if (arg == "--min-tile" && i + 1 < argc) {
            params.min_tile_size = std::atoi(argv[++i]);
        } else if (arg == "--max-tile" && i + 1 < argc) {
            params.max_tile_size = std::atoi(argv[++i]);
        } else if (arg == "--min-width" && i + 1 < argc) {
            params.min_kernel_width = std::atoi(argv[++i]);
        } else if (arg == "--max-width" && i + 1 < argc) {
            params.max_kernel_width = std::atoi(argv[++i]);
        } else if (arg == "--ncu-mode") {
            params.ncu_mode        = true;
            params.warmup_runs     = 1;
            params.benchmark_runs  = 1;
        } else if (arg == "--real") {          // NEW
            params.use_real = true;
        } else if (arg == "--optimize") {      // NEW
            params.optimize = true;
        } else if (arg == "--sa-steps" && i + 1 < argc) {
            params.sa_steps = std::atoi(argv[++i]);
        } else if (arg == "--sa-t0" && i + 1 < argc) {
            params.sa_t0 = std::atof(argv[++i]);
        } else if (arg == "--sa-alpha" && i + 1 < argc) {
            params.sa_alpha = std::atof(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        }
    }
    return params;
}

// ============================================================================
// Statistics
// ============================================================================

struct TimingStats {
    double mean_ms   = 0;
    double stddev_ms = 0;
    double min_ms    = 0;
    double max_ms    = 0;
    double median_ms = 0;
    size_t count     = 0;
};

TimingStats compute_stats(const std::vector<double>& times_sec) {
    TimingStats stats{};
    stats.count = times_sec.size();
    if (stats.count == 0) return stats;

    std::vector<double> ms(stats.count);
    for (size_t i = 0; i < stats.count; ++i) ms[i] = times_sec[i] * 1000.0;

    double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
    stats.mean_ms = sum / stats.count;

    double sq = 0.0;
    for (double t : ms) sq += (t - stats.mean_ms) * (t - stats.mean_ms);
    stats.stddev_ms = (stats.count > 1) ? std::sqrt(sq / (stats.count - 1)) : 0.0;

    stats.min_ms = *std::min_element(ms.begin(), ms.end());
    stats.max_ms = *std::max_element(ms.begin(), ms.end());

    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    stats.median_ms = (stats.count % 2 == 0)
        ? (s[stats.count/2 - 1] + s[stats.count/2]) / 2.0
        : s[stats.count/2];

    return stats;
}

// ============================================================================
// Benchmark Result
// ============================================================================

struct BenchmarkResult {
    std::string method;
    std::string distribution;
    std::string value_type;        // NEW: "real" or "complex"
    // Per-dimension tile sizes (rectangular support)
    std::array<int,3> tile_sizes = {1,1,1};
    int tile_size = 1;             // kept for backward-compat (uniform case)
    int kernel_width;
    size_t n_particles;
    size_t n_grid;
    double rho;
    bool from_optimizer = false;   // NEW: true if produced by SA

    TimingStats stats;
    std::vector<double> times_sec;

    double throughput_Mpts_per_sec() const {
        return (n_particles / (stats.mean_ms * 1e-3)) / 1e6;
    }
    double time_per_point_ns() const {
        return (stats.mean_ms * 1e6) / n_particles;
    }
};

// ============================================================================
// SA Optimizer Result
// ============================================================================

struct SAResult {
    std::string method;
    std::string value_type;
    int kernel_width;
    std::array<int,3> best_tile;
    double best_throughput_Mpts;
    double best_time_ms;
    int evaluations;
    // history: (step, tile_x, tile_y, tile_z, throughput)
    std::vector<std::tuple<int,int,int,int,double>> history;
};

// ============================================================================
// Core Benchmark Engine  (templated on ExecSpace and ValueT)
// ============================================================================

template <typename ExecSpace, typename ValueT>
class TileSweepBenchmark {
public:
    static constexpr unsigned Dim = 3;
    using real_type   = double;
    using value_type  = ValueT;
    using MemSpace    = typename ExecSpace::memory_space;

    using Mesh_t      = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t = typename Mesh_t::DefaultCentering;
    using Field_t     = ippl::Field<value_type, Dim, Mesh_t, Centering_t>;
    using PLayout_t   = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t     = ippl::ParticleBase<PLayout_t>;

    static constexpr bool is_complex = !std::is_same_v<ValueT, real_type>;
    static const char* value_type_str() { return is_complex ? "complex" : "real"; }
    static value_type zero() { return value_type(0); }
    static value_type one()  { return value_type(1); }

    // ------------------------------------------------------------------
    // Shared-memory capacity query
    //
    // Returns the maximum dynamically-allocatable shared memory per block
    // on the current default device.  Falls back to 1 GiB on backends
    // that don't have real shmem constraints (Serial, OpenMP, Threads).
    // ------------------------------------------------------------------
    static size_t device_shmem_bytes() {
#if defined(KOKKOS_ENABLE_CUDA)
        int dev = 0;
        cudaGetDevice(&dev);
        // Try the larger "optin" limit first (requires cudaFuncSetAttribute).
        int bytes = 0;
        cudaDeviceGetAttribute(&bytes,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (bytes <= 0)
            cudaDeviceGetAttribute(&bytes,
                cudaDevAttrMaxSharedMemoryPerBlock, dev);
        return static_cast<size_t>(std::max(bytes, 0));
#elif defined(KOKKOS_ENABLE_HIP)
        int dev = 0;
        hipGetDevice(&dev);
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, dev);
        return prop.sharedMemPerBlock;
#else
        return static_cast<size_t>(1) << 30;  // 1 GiB sentinel: never rejects
#endif
    }

    // ------------------------------------------------------------------
    // Scratch-size formulae mirroring the actual kernel implementations.
    //
    // OutputFocused (Grid-Parallel):
    //   scratch = (IsComplex?2:1) * htot * sizeof(double)   // field tile
    //             + Dim * W * sizeof(double)                  // kernel weights
    //             + Dim * sizeof(int)                         // int offsets
    //   htot = prod_d(tile[d] + W)
    //
    // Tiled (Sorted Spread):
    //   scratch = (IsComplex?2:1) * htot * sizeof(double)
    //   htot = prod_d(tile[d] + W)
    // ------------------------------------------------------------------
    static size_t scratch_size_output_focused(const std::array<int,3>& tile, int W) {
        size_t htot = 1;
        for (unsigned d = 0; d < Dim; ++d)
            htot *= static_cast<size_t>(tile[d] + W);
        return (is_complex ? 2u : 1u) * htot * sizeof(real_type)
               + static_cast<size_t>(Dim) * static_cast<size_t>(W) * sizeof(real_type)
               + static_cast<size_t>(Dim) * sizeof(int);
    }

    static size_t scratch_size_tiled(const std::array<int,3>& tile, int W) {
        size_t htot = 1;
        for (unsigned d = 0; d < Dim; ++d)
            htot *= static_cast<size_t>(tile[d] + W);
        return (is_complex ? 2u : 1u) * htot * sizeof(real_type);
    }

    // Returns true if the tile+kernel combination fits in device shared memory.
    bool fits_in_shmem(const std::string& method,
                       const std::array<int,3>& tile,
                       int W) const {
        size_t required = (method == "Tiled")
                          ? scratch_size_tiled(tile, W)
                          : scratch_size_output_focused(tile, W);
        return required <= device_shmem_bytes();
    }

    TileSweepBenchmark(const BenchParams& params)
        : params_(params) {}

    // ------------------------------------------------------------------
    // Top-level entry point
    // ------------------------------------------------------------------
    void run() {
        print_header();

        std::vector<BenchmarkResult> results;
        std::vector<SAResult>        sa_results;

        // Build list of kernel widths to sweep
        std::vector<int> widths;
        for (int w = params_.min_kernel_width; w <= params_.max_kernel_width; ++w)
            widths.push_back(w);

        std::vector<int> tile_sizes_1d;
        for (int t = params_.min_tile_size; t <= params_.max_tile_size; ++t)
            tile_sizes_1d.push_back(t);

        const int total_configs = widths.size() * tile_sizes_1d.size() * 2;
        int current_config = 0;

        for (int width : widths) {
            double tol = std::pow(10.0, -(width - 1));
            ippl::NUFFT::ESKernel<real_type> kernel(tol);
            int actual_width = kernel.width();

            if (actual_width != width && ippl::Comm->rank() == 0)
                std::cout << "Note: Requested width " << width
                          << ", got " << actual_width << " (tol=" << tol << ")\n";

            int nghost = actual_width / 2 + 1;
            setup_domain(nghost);
            initialize(kernel, nghost);
            size_t n_particles = bunch_->getLocalNum();

            // ---- Uniform tile sweep ----------------------------------------
            for (int t : tile_sizes_1d) {
                ++current_config;
                if (ippl::Comm->rank() == 0) {
                    std::cout << "\r[sweep " << current_config << "/" << total_configs << "] "
                              << "width=" << actual_width << ", tile=" << t
                              << "          " << std::flush;
                }

                {
                    const std::array<int,3> tile_arr = {t, t, t};
                    auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
                    cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
                    cfg.set_tile_size(t);
                    if (fits_in_shmem("Tiled", tile_arr, actual_width)) {
                        results.push_back(benchmark_scatter("Tiled", cfg, kernel, n_particles,
                                                            tile_arr, false));
                    } else {
                        if (params_.verbose && ippl::Comm->rank() == 0)
                            std::cout << "  [shmem-skip] Tiled tile=" << t
                                      << " width=" << actual_width << "\n";
                    }
                }
                ++current_config;
                {
                    const std::array<int,3> tile_arr = {t, t, t};
                    auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
                    cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
                    cfg.set_tile_size(t);
                    if (fits_in_shmem("OutputFocused", tile_arr, actual_width)) {
                        results.push_back(benchmark_scatter("OutputFocused", cfg, kernel, n_particles,
                                                            tile_arr, false));
                    } else {
                        if (params_.verbose && ippl::Comm->rank() == 0)
                            std::cout << "  [shmem-skip] OutputFocused tile=" << t
                                      << " width=" << actual_width << "\n";
                    }
                }
            }

            // ---- Simulated-annealing optimiser --------------------------------
            if (params_.optimize) {
                if (ippl::Comm->rank() == 0)
                    std::cout << "\n  [SA] Optimising tile sizes for width=" << actual_width << "...\n";

                for (const std::string& method : {"Tiled", "OutputFocused"}) {
                    auto sa = run_sa(method, kernel, n_particles);
                    sa_results.push_back(sa);

                    // Store the best SA point as a result row (flagged)
                    auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
                    if (method == "Tiled")
                        cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
                    else
                        cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
                    cfg.tile_size = {sa.best_tile[0], sa.best_tile[1], sa.best_tile[2]};
                    auto r = benchmark_scatter(method, cfg, kernel, n_particles,
                                               sa.best_tile, true);
                    results.push_back(r);
                }
            }

            cleanup();
        }

        if (ippl::Comm->rank() == 0) std::cout << "\n";

        // Output
        write_full_csv(results);
        write_heatmap_csv(results, "Tiled");
        write_heatmap_csv(results, "OutputFocused");
        write_optimal_csv(results);
        if (params_.optimize) {
            write_sa_csv(sa_results);
            write_sa_history_csv(sa_results);
        }
        print_summary(results, sa_results);
    }

    // ------------------------------------------------------------------
    // Single-configuration scatter benchmark
    // ------------------------------------------------------------------
    BenchmarkResult benchmark_scatter(const std::string& method,
                                      const ippl::Interpolation::ScatterConfig<Dim>& cfg,
                                      const ippl::NUFFT::ESKernel<real_type>& kernel,
                                      size_t n_particles,
                                      std::array<int,3> tile_arr,
                                      bool from_optimizer) {
        BenchmarkResult r;
        r.method         = method;
        r.distribution   = params_.distribution;
        r.value_type     = value_type_str();
        r.tile_sizes     = tile_arr;
        r.tile_size      = tile_arr[0];   // uniform representation for heatmap
        r.kernel_width   = kernel.width();
        r.n_particles    = n_particles;
        r.n_grid         = params_.n_grid;
        r.rho            = params_.rho;
        r.from_optimizer = from_optimizer;

        try {
            ManualTimer timer;

            for (int i = 0; i < params_.warmup_runs; ++i) {
                *grid_ = zero();
                Q_.scatter_kernel(*grid_, R_, kernel, cfg);
                grid_->accumulateHalo();
            }
            Kokkos::fence();

            std::vector<double> times;
            times.reserve(params_.benchmark_runs);
            for (int i = 0; i < params_.benchmark_runs; ++i) {
                *grid_ = zero();
                Kokkos::fence();
                timer.start();
                Q_.scatter_kernel(*grid_, R_, kernel, cfg);
                grid_->accumulateHalo();
                times.push_back(timer.stop());
            }

            r.times_sec = times;
            r.stats     = compute_stats(times);

        } catch (const std::runtime_error& e) {
            if (ippl::Comm->rank() == 0 && params_.verbose)
                std::cout << "\n    [SKIP] " << method
                          << " tile=(" << tile_arr[0] << "," << tile_arr[1] << "," << tile_arr[2]
                          << ") width=" << kernel.width() << ": " << e.what() << "\n";

            const double nan = std::numeric_limits<double>::quiet_NaN();
            r.stats.mean_ms = r.stats.stddev_ms = r.stats.min_ms
                            = r.stats.max_ms    = r.stats.median_ms = nan;
            r.stats.count   = 0;
        }
        return r;
    }

    // ------------------------------------------------------------------
    // Simulated Annealing over integer tile sizes (per dimension)
    // ------------------------------------------------------------------
    //
    // State space:  tile ∈ [min_tile, max_tile]^3  (integers, per dimension)
    // Objective:    maximise throughput (Mpts/s)
    // Moves:        randomly perturb one dimension by ±1 (clamped to bounds)
    // Budget:       sa_steps = number of *distinct kernel evaluations*.
    //               Rejected proposals and boundary-clamped no-ops do NOT
    //               consume budget or advance the cooling schedule — only a
    //               real GPU measurement counts as a step.  This means the
    //               user's --sa-steps budget is never wasted on duplicates.
    // Cache:        previously-evaluated configs are looked up in a map so
    //               that revisiting a known tile (common when the chain
    //               bounces around a local basin) costs zero kernel time.
    //               Cached hits still participate in Metropolis acceptance
    //               and DO advance the cooling counter.
    // Schedule:     geometric cooling T_k = T0 * alpha^k, where T0 is
    //               AUTO-CALIBRATED to the observed throughput scale so that
    //               the initial acceptance rate for a ~5% regression is ~50%.
    // Restart:      after half the evaluation budget, restart from the
    //               best-seen point with T reset to T0/4 ("iterated SA").
    // ------------------------------------------------------------------
    SAResult run_sa(const std::string& method,
                    const ippl::NUFFT::ESKernel<real_type>& kernel,
                    size_t n_particles) {

        SAResult sa;
        sa.method       = method;
        sa.value_type   = value_type_str();
        sa.kernel_width = kernel.width();
        sa.evaluations  = 0;

        const int lo = params_.min_tile_size;
        const int hi = params_.max_tile_size;

        // Seed includes method hash so two methods at the same width differ.
        std::size_t method_hash = std::hash<std::string>{}(method);
        std::mt19937 rng(static_cast<uint32_t>(12345
                         + kernel.width() * 1000
                         + (method_hash & 0xFFFF)));

        std::uniform_int_distribution<int>     dim_dist(0, Dim - 1);
        std::uniform_int_distribution<int>     delta_dist(0, 1);  // 0→-1, 1→+1
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        // ---- cache: tile → throughput (avoids re-running known configs) ----
        // Key: flat index  x*(R^2) + y*R + z  where R = hi - lo + 1
        const int R = hi - lo + 1;
        auto tile_key = [&](const std::array<int,Dim>& t) -> int {
            return (t[0]-lo)*R*R + (t[1]-lo)*R + (t[2]-lo);
        };
        std::unordered_map<int, double> cache;

        // ---- evaluate-or-lookup helper -------------------------------------
        // Returns throughput, sets *was_cached=true if no kernel was run.
        // Returns 0.0 immediately (cached as such) if the tile exceeds the
        // device shared-memory limit — avoids a hang on OOM launches.
        const int kernel_W = kernel.width();
        auto evaluate = [&](const std::array<int,Dim>& tile, bool* was_cached) -> double {
            int key = tile_key(tile);
            auto it = cache.find(key);
            if (it != cache.end()) {
                if (was_cached) *was_cached = true;
                return it->second;
            }
            if (was_cached) *was_cached = false;

            // Pre-flight shmem check: cache and return 0 without launching.
            if (!fits_in_shmem(method, tile, kernel_W)) {
                if (params_.verbose && ippl::Comm->rank() == 0)
                    std::cout << "    SA [shmem-skip] tile=("
                              << tile[0] << "," << tile[1] << "," << tile[2]
                              << ") exceeds device shmem\n";
                cache[key] = 0.0;
                ++sa.evaluations;   // counts as a used evaluation
                return 0.0;
            }

            auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = (method == "Tiled")
                ? ippl::Interpolation::ScatterMethod::Tiled
                : ippl::Interpolation::ScatterMethod::OutputFocused;
            cfg.tile_size = {tile[0], tile[1], tile[2]};

            auto r = benchmark_scatter(method, cfg, kernel, n_particles, tile, true);
            ++sa.evaluations;
            double tp = std::isnan(r.stats.mean_ms) ? 0.0 : r.throughput_Mpts_per_sec();
            cache[key] = tp;
            return tp;
        };

        // ---- initial point: midpoint of search space -----------------------
        std::array<int,Dim> current;
        current.fill((lo + hi) / 2);
        bool dummy;
        double current_tp = evaluate(current, &dummy);

        std::array<int,Dim> best    = current;
        double              best_tp = current_tp;

        // ---- auto-calibrate T0 ---------------------------------------------
        // Target: exp(-0.05 * tp0 / T0) = 0.5  →  T0 = 0.05*tp0 / ln2
        double T0 = (current_tp > 0.0)
                    ? 0.05 * current_tp / std::log(2.0)
                    : params_.sa_t0;
        if (std::abs(params_.sa_t0 - 5.0) > 1e-9)
            T0 = params_.sa_t0;    // explicit user override

        double T = T0;

        // ---- alpha: span 3 orders of magnitude over the evaluation budget --
        // IMPORTANT: the while-loop cools T on every non-clamped *proposal*,
        // which is strictly >= sa_steps (real evaluations), because cached hits
        // and rejected moves also advance the cooling counter.  If alpha is set
        // to reach T0*1e-3 after only sa_steps steps, T will hit zero long
        // before the budget is exhausted, freezing the chain in an infinite loop.
        //
        // Fix: use a gentler alpha so T reaches T0*1e-3 after an expected number
        // of proposals, estimated as sa_steps * expected_proposals_per_eval.
        // We conservatively assume ~3 proposals per real evaluation on average
        // (accounts for ~50% rejection rate + some cache hits + boundary bounces).
        // A hard floor T_min = T0*1e-4 additionally guarantees termination even
        // if proposals/eval is higher than expected: once T == T_min, the chain
        // still runs but effectively does greedy local search, and the while()
        // condition on sa.evaluations ensures it always terminates.
        double alpha = params_.sa_alpha;
        if (std::abs(alpha - 0.97) < 1e-9 && params_.sa_steps > 0) {
            constexpr double expected_proposals_per_eval = 3.0;
            double effective_steps = params_.sa_steps * expected_proposals_per_eval;
            alpha = std::pow(1e-3, 1.0 / effective_steps);
        }
        const double T_min = T0 * 1e-2;   // floor: ~1% regression still ~1% accepted

        const int restart_eval = params_.sa_steps / 2;  // restart after this many evals
        bool restarted = false;                          // fire exactly once

        if (params_.verbose && ippl::Comm->rank() == 0) {
            std::cout << "    SA init: tp0=" << std::fixed << std::setprecision(1) << current_tp
                      << "  T0=" << std::setprecision(3) << T0
                      << "  alpha=" << std::setprecision(6) << alpha << "\n";
        }

        // ---- main annealing loop -------------------------------------------
        // Loop until we have consumed sa_steps real evaluations.
        // Proposals that are boundary-clamped identical to current are
        // discarded without touching the step counter or temperature.
        // Proposals that hit the cache count as a step (and cool T) but
        // don't increment sa.evaluations.
        int step = 0;
        while (sa.evaluations < params_.sa_steps) {

            // Mid-run restart after half the *evaluation* budget — fires once only
            if (!restarted && sa.evaluations >= restart_eval) {
                restarted  = true;
                current    = best;
                current_tp = best_tp;
                T          = T0 / 4.0;
                if (params_.verbose && ippl::Comm->rank() == 0)
                    std::cout << "    SA restart at eval " << sa.evaluations
                              << "  best=(" << best[0] << "," << best[1] << "," << best[2] << ")"
                              << "  T reset to " << std::setprecision(4) << T << "\n";
            }

            // Generate candidate neighbour
            std::array<int,Dim> candidate = current;
            int d   = dim_dist(rng);
            int dir = (delta_dist(rng) == 0) ? -1 : +1;
            candidate[d] = std::clamp(current[d] + dir, lo, hi);

            // Discard boundary-clamped no-ops without consuming any budget
            if (candidate == current) continue;

            bool was_cached = false;
            double candidate_tp = evaluate(candidate, &was_cached);

            // Metropolis acceptance — guard exp() against underflow
            double delta = candidate_tp - current_tp;
            if (delta > 0.0 || unif(rng) < std::exp(std::max(delta / T, -500.0))) {
                current    = candidate;
                current_tp = candidate_tp;
            }

            if (current_tp > best_tp) {
                best    = current;
                best_tp = current_tp;
            }

            // Record history entry and cool — once per real proposal
            // (whether accepted or not, whether cached or not)
            sa.history.emplace_back(step, current[0], current[1], current[2], current_tp);
            T = std::max(T * alpha, T_min);   // clamp: never let T reach zero
            ++step;

            if (params_.verbose && ippl::Comm->rank() == 0) {
                std::cout << "    SA eval " << std::setw(4) << sa.evaluations
                          << " step " << std::setw(4) << step
                          << (was_cached ? "C" : " ")
                          << "  T=" << std::fixed << std::setprecision(4) << T
                          << "  tile=(" << current[0] << "," << current[1] << "," << current[2] << ")"
                          << "  tp=" << std::setprecision(1) << current_tp
                          << "  best=(" << best[0] << "," << best[1] << "," << best[2] << ")"
                          << "  best_tp=" << best_tp << "\n";
            }
        }

        sa.best_tile = best;

        // ---- final re-measurement of best config with full statistics -------
        {
            auto cfg = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = (method == "Tiled")
                ? ippl::Interpolation::ScatterMethod::Tiled
                : ippl::Interpolation::ScatterMethod::OutputFocused;
            cfg.tile_size = {best[0], best[1], best[2]};
            auto r = benchmark_scatter(method, cfg, kernel, n_particles, best, true);
            sa.best_time_ms         = r.stats.mean_ms;
            sa.best_throughput_Mpts = r.throughput_Mpts_per_sec();
        }

        if (ippl::Comm->rank() == 0) {
            std::cout << "  [SA] " << method << " w=" << kernel.width()
                      << "  best tile=(" << best[0] << "," << best[1] << "," << best[2] << ")"
                      << "  throughput=" << std::fixed << std::setprecision(1)
                      << sa.best_throughput_Mpts << " Mpts/s"
                      << "  (" << sa.evaluations << " evals)\n";
        }

        return sa;
    }

    // ------------------------------------------------------------------
    // Domain / particle setup / teardown
    // ------------------------------------------------------------------

    void setup_domain(int /*nghost*/) {
        for (unsigned d = 0; d < Dim; ++d) n_grid_[d] = params_.n_grid;

        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d) domain[d] = ippl::Index(n_grid_[d]);

        std::array<bool, Dim> isParallel; isParallel.fill(true);
        layout_ = std::make_unique<ippl::FieldLayout<Dim>>(
            MPI_COMM_WORLD, domain, isParallel, true);

        for (unsigned d = 0; d < Dim; ++d) {
            origin_[d] = 0.0;
            hx_[d]     = 2.0 * M_PI / static_cast<real_type>(n_grid_[d]);
        }
        mesh_ = std::make_unique<Mesh_t>(domain, hx_, origin_);
    }

    void initialize(const ippl::NUFFT::ESKernel<real_type>& /*kernel*/, int nghost) {
        grid_    = std::make_unique<Field_t>(*mesh_, *layout_, nghost);
        playout_ = std::make_unique<PLayout_t>(*layout_, *mesh_);
        bunch_   = std::make_unique<Bunch_t>(*playout_);

        bunch_->addAttribute(R_);
        bunch_->addAttribute(Q_);
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
                        double u1 = gen.drand(), u2 = gen.drand();
                        double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-10))
                                   * Kokkos::cos(2.0 * M_PI * u2);
                        R_view(i)[d] = M_PI + 0.3 * z;
                        while (R_view(i)[d] < 0)           R_view(i)[d] += 2.0 * M_PI;
                        while (R_view(i)[d] >= 2.0 * M_PI) R_view(i)[d] -= 2.0 * M_PI;
                    }
                    rand_pool.free_state(gen);
                });
        }

        auto Q_view = Q_.getView();
        Kokkos::parallel_for("init_values", n_local,
            KOKKOS_LAMBDA(const size_t i) { Q_view(i) = one(); });

        Kokkos::fence();
    }

    void cleanup() {
        bunch_.reset();
        playout_.reset();
        grid_.reset();
        mesh_.reset();
        layout_.reset();
    }

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------

    void print_header() {
        if (ippl::Comm->rank() != 0) return;
        std::cout << "\n"
                  << "================================================================\n"
                  << "     Tile Size × Kernel Width Sweep Benchmark\n"
                  << "================================================================\n"
                  << "Grid size:       " << params_.n_grid << "^3\n"
                  << "Particles/grid:  " << params_.rho << "\n"
                  << "Total particles: " << params_.n_particles() << "\n"
                  << "Distribution:    " << params_.distribution << "\n"
                  << "Value type:      " << value_type_str() << "\n"     // NEW
                  << "Tile sizes:      " << params_.min_tile_size
                                         << " - " << params_.max_tile_size << "\n"
                  << "Kernel widths:   " << params_.min_kernel_width
                                         << " - " << params_.max_kernel_width << "\n"
                  << "Warmup runs:     " << params_.warmup_runs << "\n"
                  << "Benchmark runs:  " << params_.benchmark_runs << "\n";
        if (params_.optimize)
            std::cout << "SA optimiser:    enabled  (steps=" << params_.sa_steps
                      << ", T0=" << params_.sa_t0
                      << ", alpha=" << params_.sa_alpha << ")\n";
        std::cout << "================================================================\n\n";
    }

    void write_full_csv(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_full.csv";
        std::ofstream out(filename);

        // New columns: value_type, tile_x, tile_y, tile_z, from_optimizer
        out << "method,distribution,value_type,"
            << "tile_x,tile_y,tile_z,kernel_width,n_particles,n_grid,rho,"
            << "mean_ms,stddev_ms,min_ms,max_ms,median_ms,"
            << "throughput_Mpts_s,time_per_pt_ns,from_optimizer,status\n";

        for (const auto& r : results) {
            out << r.method << ","
                << r.distribution << ","
                << r.value_type << ","
                << r.tile_sizes[0] << ","
                << r.tile_sizes[1] << ","
                << r.tile_sizes[2] << ","
                << r.kernel_width << ","
                << r.n_particles << ","
                << r.n_grid << ","
                << std::fixed << std::setprecision(1) << r.rho << ",";

            if (std::isnan(r.stats.mean_ms)) {
                out << "nan,nan,nan,nan,nan,nan,nan,"
                    << (r.from_optimizer ? "1" : "0") << ",failed\n";
            } else {
                out << std::setprecision(4) << r.stats.mean_ms << ","
                    << r.stats.stddev_ms << ","
                    << r.stats.min_ms << ","
                    << r.stats.max_ms << ","
                    << r.stats.median_ms << ","
                    << std::setprecision(2) << r.throughput_Mpts_per_sec() << ","
                    << r.time_per_point_ns() << ","
                    << (r.from_optimizer ? "1" : "0") << ",ok\n";
            }
        }
        out.close();
        std::cout << "Wrote full results to: " << filename << "\n";
    }

    void write_heatmap_csv(const std::vector<BenchmarkResult>& results,
                           const std::string& method) {
        if (ippl::Comm->rank() != 0) return;

        // Heatmap only uses uniform-tile results (tile_x == tile_y == tile_z)
        std::string filename = params_.output_prefix + "_heatmap_" + method + ".csv";
        std::ofstream out(filename);

        std::vector<int> widths, tiles;
        for (const auto& r : results) {
            if (r.method == method && !r.from_optimizer) {
                if (std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end())
                    widths.push_back(r.kernel_width);
                if (std::find(tiles.begin(), tiles.end(), r.tile_sizes[0]) == tiles.end())
                    tiles.push_back(r.tile_sizes[0]);
            }
        }
        std::sort(widths.begin(), widths.end());
        std::sort(tiles.begin(), tiles.end());

        out << "tile_size";
        for (int w : widths) out << ",width_" << w;
        out << "\n";

        for (int t : tiles) {
            out << t;
            for (int w : widths) {
                bool found = false;
                for (const auto& r : results) {
                    if (r.method == method && !r.from_optimizer
                            && r.tile_sizes[0] == t && r.kernel_width == w) {
                        out << (std::isnan(r.stats.mean_ms)
                                    ? ",nan"
                                    : "," + std::to_string(r.throughput_Mpts_per_sec()));
                        found = true;
                        break;
                    }
                }
                if (!found) out << ",nan";
            }
            out << "\n";
        }
        out.close();
        std::cout << "Wrote heatmap for " << method << " to: " << filename << "\n";
    }

    void write_optimal_csv(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_optimal.csv";
        std::ofstream out(filename);
        out << "method,kernel_width,optimal_tile_size,throughput_Mpts_s,time_ms\n";

        std::vector<std::string> methods = {"Tiled", "OutputFocused"};
        std::vector<int> widths;
        for (const auto& r : results) {
            if (std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end())
                widths.push_back(r.kernel_width);
        }
        std::sort(widths.begin(), widths.end());

        for (const auto& method : methods) {
            for (int w : widths) {
                const BenchmarkResult* best = nullptr;
                double best_tp = 0.0;
                for (const auto& r : results) {
                    if (r.method == method && r.kernel_width == w
                            && !r.from_optimizer && !std::isnan(r.stats.mean_ms)) {
                        double tp = r.throughput_Mpts_per_sec();
                        if (tp > best_tp) { best_tp = tp; best = &r; }
                    }
                }
                if (best)
                    out << method << "," << w << "," << best->tile_sizes[0] << ","
                        << std::fixed << std::setprecision(2) << best->throughput_Mpts_per_sec()
                        << "," << std::setprecision(4) << best->stats.mean_ms << "\n";
                else
                    out << method << "," << w << ",nan,nan,nan\n";
            }
        }
        out.close();
        std::cout << "Wrote optimal configurations to: " << filename << "\n";
    }

    void write_sa_csv(const std::vector<SAResult>& sa_results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_sa_optimal.csv";
        std::ofstream out(filename);
        out << "method,value_type,kernel_width,"
            << "best_tile_x,best_tile_y,best_tile_z,"
            << "throughput_Mpts_s,time_ms,evaluations\n";

        for (const auto& sa : sa_results) {
            out << sa.method << ","
                << sa.value_type << ","
                << sa.kernel_width << ","
                << sa.best_tile[0] << ","
                << sa.best_tile[1] << ","
                << sa.best_tile[2] << ","
                << std::fixed << std::setprecision(2) << sa.best_throughput_Mpts << ","
                << std::setprecision(4) << sa.best_time_ms << ","
                << sa.evaluations << "\n";
        }
        out.close();
        std::cout << "Wrote SA optimal results to: " << filename << "\n";
    }

    void write_sa_history_csv(const std::vector<SAResult>& sa_results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_sa_history.csv";
        std::ofstream out(filename);
        out << "method,value_type,kernel_width,step,tile_x,tile_y,tile_z,throughput_Mpts_s\n";

        for (const auto& sa : sa_results) {
            for (const auto& [step, tx, ty, tz, tp] : sa.history) {
                out << sa.method << ","
                    << sa.value_type << ","
                    << sa.kernel_width << ","
                    << step << ","
                    << tx << "," << ty << "," << tz << ","
                    << std::fixed << std::setprecision(2) << tp << "\n";
            }
        }
        out.close();
        std::cout << "Wrote SA convergence history to: " << filename << "\n";
    }

    void print_summary(const std::vector<BenchmarkResult>& results,
                       const std::vector<SAResult>& sa_results) {
        if (ippl::Comm->rank() != 0) return;

        int failed = 0;
        for (const auto& r : results)
            if (std::isnan(r.stats.mean_ms)) ++failed;

        std::cout << "\n"
                  << "================================================================\n"
                  << "                    Results Summary (" << value_type_str() << ")\n"
                  << "================================================================\n";
        if (failed > 0)
            std::cout << "\nNote: " << failed << " configuration(s) failed.\n";

        // Per-method optimal uniform-tile table
        std::vector<std::string> methods = {"Tiled", "OutputFocused"};
        std::vector<int> widths;
        for (const auto& r : results) {
            if (!r.from_optimizer
                    && std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end())
                widths.push_back(r.kernel_width);
        }
        std::sort(widths.begin(), widths.end());

        for (const auto& method : methods) {
            std::cout << "\n" << method << " — optimal uniform tile by kernel width:\n"
                      << std::string(60, '-') << "\n"
                      << std::left << std::setw(8) << "Width"
                      << std::right << std::setw(12) << "Best Tile"
                      << std::setw(14) << "Mpts/s"
                      << std::setw(12) << "Time (ms)" << "\n"
                      << std::string(60, '-') << "\n";

            for (int w : widths) {
                const BenchmarkResult* best = nullptr;
                double best_tp = 0.0;
                for (const auto& r : results) {
                    if (r.method == method && r.kernel_width == w
                            && !r.from_optimizer && !std::isnan(r.stats.mean_ms)) {
                        double tp = r.throughput_Mpts_per_sec();
                        if (tp > best_tp) { best_tp = tp; best = &r; }
                    }
                }
                if (best)
                    std::cout << std::left << std::setw(8) << w
                              << std::right << std::setw(12) << best->tile_sizes[0]
                              << std::fixed << std::setprecision(1)
                              << std::setw(14) << best->throughput_Mpts_per_sec()
                              << std::setprecision(3)
                              << std::setw(12) << best->stats.mean_ms << "\n";
                else
                    std::cout << std::left << std::setw(8) << w
                              << std::right << std::setw(38) << "all failed\n";
            }
        }

        // SA results table (if available)
        if (!sa_results.empty()) {
            std::cout << "\n"
                      << "================================================================\n"
                      << "        SA-Optimised Rectangular Tile Sizes\n"
                      << "================================================================\n"
                      << std::left  << std::setw(16) << "Method"
                      << std::right << std::setw(8)  << "Width"
                      << std::setw(22) << "Best tile (x,y,z)"
                      << std::setw(14) << "Mpts/s"
                      << std::setw(10) << "Evals" << "\n"
                      << std::string(70, '-') << "\n";

            for (const auto& sa : sa_results) {
                std::ostringstream tile_str;
                tile_str << "(" << sa.best_tile[0] << ","
                               << sa.best_tile[1] << ","
                               << sa.best_tile[2] << ")";
                std::cout << std::left  << std::setw(16) << sa.method
                          << std::right << std::setw(8)  << sa.kernel_width
                          << std::setw(22) << tile_str.str()
                          << std::fixed << std::setprecision(1)
                          << std::setw(14) << sa.best_throughput_Mpts
                          << std::setw(10) << sa.evaluations << "\n";
            }
        }

        std::cout << "\n";
    }

private:
    BenchParams params_;

    ippl::Vector<std::size_t, Dim> n_grid_;
    ippl::Vector<real_type, Dim>   origin_;
    ippl::Vector<real_type, Dim>   hx_;

    std::unique_ptr<ippl::FieldLayout<Dim>> layout_;
    std::unique_ptr<Mesh_t>    mesh_;
    std::unique_ptr<Field_t>   grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t>   bunch_;

    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R_;
    ippl::ParticleAttrib<value_type>                   Q_;   // real or complex
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        auto params = parse_bench_args(argc, argv);

        if (params.use_real) {
            TileSweepBenchmark<Kokkos::DefaultExecutionSpace, double> bench(params);
            bench.run();
        } else {
            TileSweepBenchmark<Kokkos::DefaultExecutionSpace,
                               Kokkos::complex<double>> bench(params);
            bench.run();
        }
    }

    ippl::finalize();
    return EXIT_SUCCESS;
}