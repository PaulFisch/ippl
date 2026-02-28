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
 *   --optimize         Run Bayesian Optimisation for rectangular tile sizes
 *   --bo-budget N      BO: total number of kernel evaluations (default: 60)
 *   --bo-xi X          BO: EI exploration parameter xi (default: 0.01)
 *   --bo-ucb-prob P    BO: probability of using UCB vs EI acquisition (default: 0.3)
 *   -v, --verbose      Verbose output
 *
 */

#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
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
    int n_grid                = 128;
    double rho                = 1.0;
    int warmup_runs           = 3;
    int benchmark_runs        = 5;
    std::string output_prefix = "tile_sweep";
    std::string distribution  = "uniform";
    bool verbose              = false;
    bool ncu_mode             = false;
    bool use_real             = false;
    bool optimize             = false;

    // Sweep ranges
    int min_tile_size    = 1;
    int max_tile_size    = 8;
    int min_kernel_width = 2;
    int max_kernel_width = 8;

    // Bayesian Optimisation parameters
    // (sa_steps kept as alias for bo_budget for CLI backward compatibility)
    int bo_budget      = 60;    // total number of kernel evaluations
    double bo_xi       = 0.01;  // EI exploration parameter
    double bo_ucb_prob = 0.30;  // fraction of steps that use UCB instead of EI

    // Legacy SA parameters — accepted but ignored (kept for backward compat)
    double sa_t0    = 5.0;
    double sa_alpha = 0.97;

    size_t n_particles() const { return static_cast<size_t>(rho * n_grid * n_grid * n_grid); }
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
            params.ncu_mode       = true;
            params.warmup_runs    = 1;
            params.benchmark_runs = 1;
        } else if (arg == "--real") {
            params.use_real = true;
        } else if (arg == "--optimize") {
            params.optimize = true;
        } else if ((arg == "--bo-budget" || arg == "--sa-steps") && i + 1 < argc) {
            params.bo_budget = std::atoi(argv[++i]);
        } else if (arg == "--bo-xi" && i + 1 < argc) {
            params.bo_xi = std::atof(argv[++i]);
        } else if (arg == "--bo-ucb-prob" && i + 1 < argc) {
            params.bo_ucb_prob = std::atof(argv[++i]);
        } else if (arg == "--sa-t0" && i + 1 < argc) {
            params.sa_t0 = std::atof(argv[++i]);  // accepted, ignored
        } else if (arg == "--sa-alpha" && i + 1 < argc) {
            params.sa_alpha = std::atof(argv[++i]);  // accepted, ignored
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
    if (stats.count == 0)
        return stats;

    std::vector<double> ms(stats.count);
    for (size_t i = 0; i < stats.count; ++i)
        ms[i] = times_sec[i] * 1000.0;

    double sum    = std::accumulate(ms.begin(), ms.end(), 0.0);
    stats.mean_ms = sum / stats.count;

    double sq = 0.0;
    for (double t : ms)
        sq += (t - stats.mean_ms) * (t - stats.mean_ms);
    stats.stddev_ms = (stats.count > 1) ? std::sqrt(sq / (stats.count - 1)) : 0.0;

    stats.min_ms = *std::min_element(ms.begin(), ms.end());
    stats.max_ms = *std::max_element(ms.begin(), ms.end());

    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    stats.median_ms = (stats.count % 2 == 0) ? (s[stats.count / 2 - 1] + s[stats.count / 2]) / 2.0
                                             : s[stats.count / 2];

    return stats;
}

// ============================================================================
// Benchmark Result
// ============================================================================

struct BenchmarkResult {
    std::string method;
    std::string distribution;
    std::string value_type;
    std::array<int, 3> tile_sizes = {1, 1, 1};
    int tile_size                 = 1;
    int kernel_width;
    size_t n_particles;
    size_t n_grid;
    double rho;
    bool from_optimizer = false;

    TimingStats stats;
    std::vector<double> times_sec;

    double throughput_Mpts_per_sec() const { return (n_particles / (stats.mean_ms * 1e-3)) / 1e6; }
    double time_per_point_ns() const { return (stats.mean_ms * 1e6) / n_particles; }
};

// ============================================================================
// Optimiser Result  (named SAResult for API compatibility)
// ============================================================================

struct SAResult {
    std::string method;
    std::string value_type;
    int kernel_width;
    std::array<int, 3> best_tile;
    double best_throughput_Mpts;
    double best_time_ms;
    int evaluations;
    std::vector<std::tuple<int, int, int, int, double>> history;
};

// ============================================================================
// Minimal self-contained Gaussian Process (RBF kernel, no external deps)
// ============================================================================
//
// Represents observations on a 3-D integer lattice.
// Points are normalised to [0,1]^3 internally.
// Hyperparameters (length-scale l, noise sigma_n) are selected by a quick
// grid search over log marginal likelihood.
//
// Predictions return (posterior mean, posterior variance).
// Acquisition functions: Expected Improvement (EI) and UCB.
// ============================================================================

struct GPModel {
    std::vector<std::array<int, 3>> X;
    std::vector<double> y;
    double y_mean = 0.0, y_std = 1.0;
    std::vector<double> y_norm;

    double length_scale = 0.4;
    double sigma_n      = 0.1;

    // Cholesky factor of K + sigma_n^2 I  (row-major lower triangular)
    std::vector<std::vector<double>> L;
    std::vector<double> alpha;  // K^{-1} y_norm

    int lo, hi;

    GPModel(int lo_, int hi_) : lo(lo_), hi(hi_) {}

    std::array<double, 3> normalise(const std::array<int, 3>& t) const {
        double range = std::max(hi - lo, 1);
        return {(t[0] - lo) / range, (t[1] - lo) / range, (t[2] - lo) / range};
    }

    double sq_dist(const std::array<double, 3>& a, const std::array<double, 3>& b) const {
        double s = 0;
        for (int d = 0; d < 3; ++d)
            s += (a[d] - b[d]) * (a[d] - b[d]);
        return s;
    }

    double kernel_val(const std::array<double, 3>& a, const std::array<double, 3>& b,
                      double l) const {
        return std::exp(-sq_dist(a, b) / (2.0 * l * l));
    }

    // ------------------------------------------------------------------
    // Fit: normalise observations, select hyperparameters by MLE grid search,
    //      compute Cholesky decomposition and alpha = K^{-1} y_norm.
    // ------------------------------------------------------------------
    void fit() {
        int n = (int)y.size();
        if (n == 0)
            return;

        // Normalise y using statistics of *valid* (non-zero) observations
        std::vector<double> valid;
        for (double v : y)
            if (v > 0)
                valid.push_back(v);
        if (valid.empty()) {
            y_mean = 0;
            y_std  = 1;
        } else {
            y_mean = std::accumulate(valid.begin(), valid.end(), 0.0) / valid.size();
            double var = 0;
            for (double v : valid)
                var += (v - y_mean) * (v - y_mean);
            y_std = std::sqrt(var / valid.size() + 1e-12);
        }
        y_norm.resize(n);
        for (int i = 0; i < n; ++i)
            y_norm[i] = (y[i] - y_mean) / y_std;

        // Hyperparameter grid search
        double best_lml = -1e300;
        for (double l : {0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0}) {
            for (double sn : {0.005, 0.02, 0.08, 0.2, 0.5}) {
                double lml = log_marginal_likelihood(l, sn);
                if (lml > best_lml) {
                    best_lml   = lml;
                    length_scale = l;
                    sigma_n      = sn;
                }
            }
        }

        build_cholesky(length_scale, sigma_n, L);
        solve_alpha();
    }

    // Returns log p(y | X, l, sn).  Builds its own Cholesky internally.
    double log_marginal_likelihood(double l, double sn) const {
        int n = (int)X.size();
        std::vector<std::vector<double>> Lc(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            auto xi = normalise(X[i]);
            for (int j = 0; j <= i; ++j) {
                auto xj = normalise(X[j]);
                double s = kernel_val(xi, xj, l);
                if (i == j)
                    s += sn * sn + 1e-8;
                for (int k = 0; k < j; ++k)
                    s -= Lc[i][k] * Lc[j][k];
                if (i == j) {
                    if (s <= 0)
                        return -1e300;
                    Lc[i][j] = std::sqrt(s);
                } else {
                    Lc[i][j] = s / (Lc[j][j] > 1e-12 ? Lc[j][j] : 1e-12);
                }
            }
        }
        // Solve L v = y_norm, then L^T alpha = v
        std::vector<double> v(n), alph(n);
        for (int i = 0; i < n; ++i) {
            double s = y_norm[i];
            for (int j = 0; j < i; ++j)
                s -= Lc[i][j] * v[j];
            v[i] = s / (Lc[i][i] > 1e-12 ? Lc[i][i] : 1e-12);
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = v[i];
            for (int j = i + 1; j < n; ++j)
                s -= Lc[j][i] * alph[j];
            alph[i] = s / (Lc[i][i] > 1e-12 ? Lc[i][i] : 1e-12);
        }
        double fit_term = 0;
        for (int i = 0; i < n; ++i)
            fit_term += y_norm[i] * alph[i];
        double log_det = 0;
        for (int i = 0; i < n; ++i)
            log_det += std::log(std::max(Lc[i][i], 1e-300));
        return -0.5 * fit_term - log_det - 0.5 * n * std::log(2 * M_PI);
    }

    void build_cholesky(double l, double sn, std::vector<std::vector<double>>& Lout) const {
        int n = (int)X.size();
        Lout.assign(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            auto xi = normalise(X[i]);
            for (int j = 0; j <= i; ++j) {
                auto xj = normalise(X[j]);
                double s = kernel_val(xi, xj, l);
                if (i == j)
                    s += sn * sn + 1e-8;
                for (int k = 0; k < j; ++k)
                    s -= Lout[i][k] * Lout[j][k];
                if (i == j) {
                    for (int k = 0; k < i; ++k)
                        s -= Lout[i][k] * Lout[i][k];
                    Lout[i][j] = (s > 1e-16) ? std::sqrt(s) : 1e-8;
                } else {
                    Lout[i][j] = s / (Lout[j][j] > 1e-12 ? Lout[j][j] : 1e-12);
                }
            }
        }
    }

    void solve_alpha() {
        int n = (int)X.size();
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) {
            double s = y_norm[i];
            for (int j = 0; j < i; ++j)
                s -= L[i][j] * v[j];
            v[i] = s / (L[i][i] > 1e-12 ? L[i][i] : 1e-12);
        }
        alpha.resize(n);
        for (int i = n - 1; i >= 0; --i) {
            double s = v[i];
            for (int j = i + 1; j < n; ++j)
                s -= L[j][i] * alpha[j];
            alpha[i] = s / (L[i][i] > 1e-12 ? L[i][i] : 1e-12);
        }
    }

    // Posterior (mean, variance) at point t — both in original (unnormalised) scale
    std::pair<double, double> predict(const std::array<int, 3>& t) const {
        int n = (int)X.size();
        if (n == 0)
            return {y_mean, y_std * y_std};

        auto xt = normalise(t);
        std::vector<double> k_star(n);
        for (int i = 0; i < n; ++i)
            k_star[i] = kernel_val(normalise(X[i]), xt, length_scale);

        double mu_norm = 0;
        for (int i = 0; i < n; ++i)
            mu_norm += k_star[i] * alpha[i];

        // Posterior variance: k** - k*^T K^{-1} k*  = 1 - ||L^{-1}k*||^2
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) {
            double s = k_star[i];
            for (int j = 0; j < i; ++j)
                s -= L[i][j] * v[j];
            v[i] = s / (L[i][i] > 1e-12 ? L[i][i] : 1e-12);
        }
        double var_norm = 1.0;
        for (int i = 0; i < n; ++i)
            var_norm -= v[i] * v[i];
        var_norm = std::max(var_norm, 1e-10);

        return {mu_norm * y_std + y_mean, var_norm * y_std * y_std};
    }

    // Expected Improvement: integrates over the predictive distribution
    // how much we expect to beat f_best by.
    // xi > 0 encourages exploration (larger xi → more exploration).
    double expected_improvement(const std::array<int, 3>& t, double f_best,
                                double xi = 0.01) const {
        auto [mu, var] = predict(t);
        double sigma   = std::sqrt(std::max(var, 0.0));
        if (sigma < 1e-10)
            return std::max(0.0, mu - f_best);
        double improvement = mu - f_best - xi * y_std;
        double Z           = improvement / sigma;
        double Phi         = 0.5 * (1.0 + std::erf(Z / std::sqrt(2.0)));
        double phi         = std::exp(-0.5 * Z * Z) / std::sqrt(2.0 * M_PI);
        return std::max(0.0, improvement * Phi + sigma * phi);
    }

    // Upper Confidence Bound: mu + beta * sigma  (optimistic under uncertainty)
    double ucb(const std::array<int, 3>& t, double beta = 2.0) const {
        auto [mu, var] = predict(t);
        return mu + beta * std::sqrt(std::max(var, 0.0));
    }

    void add_observation(const std::array<int, 3>& tile, double tp) {
        X.push_back(tile);
        y.push_back(tp);
    }
};

// ============================================================================
// Core Benchmark Engine  (templated on ExecSpace and ValueT)
// ============================================================================

template <typename ExecSpace, typename ValueT>
class TileSweepBenchmark {
public:
    static constexpr unsigned Dim = 3;
    using real_type               = double;
    using value_type              = ValueT;
    using MemSpace                = typename ExecSpace::memory_space;

    using Mesh_t      = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t = typename Mesh_t::DefaultCentering;
    using Field_t     = ippl::Field<value_type, Dim, Mesh_t, Centering_t>;
    using PLayout_t   = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t     = ippl::ParticleBase<PLayout_t>;

    static constexpr bool is_complex = !std::is_same_v<ValueT, real_type>;
    static const char* value_type_str() { return is_complex ? "complex" : "real"; }
    static constexpr KOKKOS_INLINE_FUNCTION value_type zero() { return value_type(0); }
    static constexpr KOKKOS_INLINE_FUNCTION value_type one() { return value_type(1); }

    static size_t device_shmem_bytes() {
#if defined(KOKKOS_ENABLE_CUDA)
        int dev = 0;
        cudaGetDevice(&dev);
        int bytes = 0;
        cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        return static_cast<size_t>(std::max(bytes, 0));
#elif defined(KOKKOS_ENABLE_HIP)
        int dev = 0;
        hipGetDevice(&dev);
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, dev);
        return prop.sharedMemPerBlock;
#else
        return static_cast<size_t>(1) << 30;
#endif
    }

    using DummyFieldView = typename Field_t::view_type;
    using DummyPosView   = typename ippl::ParticleAttrib<ippl::Vector<real_type, Dim>>::view_type;
    using DummyValView   = typename ippl::ParticleAttrib<value_type>::view_type;

    template <int W>
    using TiledTypes = ippl::Interpolation::detail::ScatterTypes<
        Dim, real_type,
        ippl::NUFFT::ESKernel<real_type>,
        DummyFieldView, DummyPosView, DummyValView>;

    template <int W>
    using GPTypes = TiledTypes<W>;

    using SortedPolicy = ippl::Interpolation::detail::SortedPolicy;

    template <int W>
    static size_t required_shmem_tiled(const ippl::Vector<int, Dim>& tv) {
        return ippl::Interpolation::detail::TiledScatter<
            W, TiledTypes<W>, SortedPolicy>::template compute_scratch_size<is_complex>(tv);
    }

    template <int W>
    static size_t required_shmem_gp(const ippl::Vector<int, Dim>& tv) {
        return ippl::Interpolation::detail::GridParallelScatter<
            W, GPTypes<W>, SortedPolicy>::template compute_scratch_size<is_complex>(tv);
    }

    static size_t required_shmem(const std::string& method, const std::array<int, 3>& tile,
                                 int W) {
        ippl::Vector<int, Dim> tv;
        for (unsigned d = 0; d < Dim; ++d)
            tv[d] = tile[d];

        size_t result = std::numeric_limits<size_t>::max();
        ippl::Interpolation::WidthDispatcher<1, 14>::dispatch(W, [&]<int Wc>() {
            if (method == "Tiled")
                result = required_shmem_tiled<Wc>(tv);
            else
                result = required_shmem_gp<Wc>(tv);
        });
        return result;
    }

    bool fits_in_shmem(const std::string& method, const std::array<int, 3>& tile, int W) const {
        const size_t required  = required_shmem(method, tile, W);
        const size_t available = device_shmem_bytes();
        if (params_.verbose && ippl::Comm->rank() == 0)
            std::cout << "  [shmem] " << method << " tile=(" << tile[0] << "," << tile[1] << ","
                      << tile[2] << ")"
                      << " W=" << W << " required=" << required << " available=" << available
                      << (required <= available ? " OK" : " SKIP") << "\n";
        return required <= available;
    }

    TileSweepBenchmark(const BenchParams& params)
        : params_(params) {}

    // ------------------------------------------------------------------
    // Top-level entry point
    // ------------------------------------------------------------------
    void run() {
        print_header();

        std::vector<BenchmarkResult> results;
        std::vector<SAResult> sa_results;

        std::vector<int> widths;
        for (int w = params_.min_kernel_width; w <= params_.max_kernel_width; ++w)
            widths.push_back(w);

        std::vector<int> tile_sizes_1d;
        for (int t = params_.min_tile_size; t <= params_.max_tile_size; ++t)
            tile_sizes_1d.push_back(t);

        const int total_configs = widths.size() * tile_sizes_1d.size() * 2;
        int current_config      = 0;

        for (int width : widths) {
            double tol = std::pow(10.0, -(width - 1));
            ippl::NUFFT::ESKernel<real_type> kernel(tol);
            int actual_width = kernel.width();

            if (actual_width != width && ippl::Comm->rank() == 0)
                std::cout << "Note: Requested width " << width << ", got " << actual_width
                          << " (tol=" << tol << ")\n";

            int nghost = actual_width / 2 + 1;
            setup_domain(nghost);
            initialize(kernel, nghost);
            size_t n_particles = bunch_->getLocalNum();

            // ---- Uniform tile sweep ----------------------------------------
            for (int t : tile_sizes_1d) {
                ++current_config;
                if (ippl::Comm->rank() == 0) {
                    std::cout << "\r[sweep " << current_config << "/" << total_configs << "] "
                              << "width=" << actual_width << ", tile=" << t << "          "
                              << std::flush;
                }

                {
                    const std::array<int, 3> tile_arr = {t, t, t};
                    auto cfg   = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
                    cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
                    cfg.set_tile_size(t);
                    if (fits_in_shmem("Tiled", tile_arr, actual_width)) {
                        results.push_back(
                            benchmark_scatter("Tiled", cfg, kernel, n_particles, tile_arr, false));
                    } else if (params_.verbose && ippl::Comm->rank() == 0) {
                        std::cout << "  [shmem-skip] Tiled tile=" << t
                                  << " width=" << actual_width << "\n";
                    }
                }
                ++current_config;
                {
                    const std::array<int, 3> tile_arr = {t, t, t};
                    auto cfg   = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
                    cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
                    cfg.set_tile_size(t);
                    if (fits_in_shmem("OutputFocused", tile_arr, actual_width)) {
                        results.push_back(benchmark_scatter("OutputFocused", cfg, kernel,
                                                            n_particles, tile_arr, false));
                    } else if (params_.verbose && ippl::Comm->rank() == 0) {
                        std::cout << "  [shmem-skip] OutputFocused tile=" << t
                                  << " width=" << actual_width << "\n";
                    }
                }
            }

            // ---- Bayesian Optimisation -------------------------------------
            if (params_.optimize) {
                if (ippl::Comm->rank() == 0)
                    std::cout << "\n  [BO] Optimising tile sizes for width=" << actual_width
                              << " (budget=" << params_.bo_budget << " evals)...\n";

                for (const std::string& method : {"Tiled", "OutputFocused"}) {
                    auto bo = run_bo(method, kernel, n_particles);
                    sa_results.push_back(bo);

                    // Store the best BO point as a flagged result row
                    if (fits_in_shmem(method, bo.best_tile, actual_width)) {
                        auto cfg =
                            ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
                        if (method == "Tiled")
                            cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
                        else
                            cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
                        cfg.tile_size = {bo.best_tile[0], bo.best_tile[1], bo.best_tile[2]};
                        auto r = benchmark_scatter(method, cfg, kernel, n_particles, bo.best_tile,
                                                   true);
                        results.push_back(r);
                    }
                }
            }

            cleanup();
        }

        if (ippl::Comm->rank() == 0)
            std::cout << "\n";

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
                                      size_t n_particles, std::array<int, 3> tile_arr,
                                      bool from_optimizer) {
        BenchmarkResult r;
        r.method         = method;
        r.distribution   = params_.distribution;
        r.value_type     = value_type_str();
        r.tile_sizes     = tile_arr;
        r.tile_size      = tile_arr[0];
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
                std::cout << "\n    [SKIP] " << method << " tile=(" << tile_arr[0] << ","
                          << tile_arr[1] << "," << tile_arr[2] << ") width=" << kernel.width()
                          << ": " << e.what() << "\n";

            const double nan     = std::numeric_limits<double>::quiet_NaN();
            r.stats.mean_ms      = r.stats.stddev_ms = r.stats.min_ms = r.stats.max_ms =
                r.stats.median_ms                                      = nan;
            r.stats.count                                              = 0;
        }
        return r;
    }

    // ============================================================================
    // run_bo: Bayesian Optimisation over integer tile sizes
    // ============================================================================
    //
    // Algorithm overview:
    //
    //   Phase 1 — Latin Hypercube Sampling initialisation  (~20% of budget)
    //     Distribute initial evaluations across the space to avoid clustering
    //     and give the GP a good prior fit before acquisition optimisation starts.
    //
    //   Phase 2 — Bayesian Optimisation loop  (~75% of budget)
    //     At each step:
    //       1. Fit GP to all observations (hyperparameter MLE grid search)
    //       2. Maximise acquisition over candidate set:
    //          - Space ≤ 5000 pts:  exhaustive sweep of integer lattice
    //          - Larger spaces:     2000 random candidates + neighbourhood of best
    //       3. Evaluate the chosen candidate (kernel benchmark)
    //       4. Update GP
    //     Acquisition alternates between EI (exploitation) and UCB (exploration)
    //     with UCB probability decreasing from 0.5 to 0.1 as progress increases.
    //
    //   Phase 3 — Local neighbourhood polish  (≤ 6 extra evaluations)
    //     ±1 grid search around the best point found.  Corrects for GP inaccuracy
    //     in high-gradient regions.
    //
    // Key properties:
    //   - OOM configs are cached with throughput=0 and NEVER re-proposed.
    //   - Termination is strictly on sa.evaluations (real kernel runs), so
    //     there is NO risk of an infinite loop regardless of cache hit rate.
    //   - For small spaces the search is nearly exhaustive within the budget.
    //   - For large spaces (e.g. tile ∈ [1,30]^3 = 27,000 pts) the GP
    //     surrogate focuses measurements on promising regions.
    // ============================================================================
    SAResult run_bo(const std::string& method, const ippl::NUFFT::ESKernel<real_type>& kernel,
                    size_t n_particles) {
        SAResult bo;
        bo.method       = method;
        bo.value_type   = value_type_str();
        bo.kernel_width = kernel.width();
        bo.evaluations  = 0;

        const int lo        = params_.min_tile_size;
        const int hi        = params_.max_tile_size;
        const int R_range   = hi - lo + 1;
        const long space_vol = (long)R_range * R_range * R_range;

        // Constraint and throughput caches
        const int kernel_W = kernel.width();
        std::unordered_map<int, bool>   feasible_cache;
        std::unordered_map<int, double> tp_cache;

        auto tile_key = [&](const std::array<int, 3>& t) -> int {
            return (t[0] - lo) * R_range * R_range + (t[1] - lo) * R_range + (t[2] - lo);
        };

        auto is_feasible = [&](const std::array<int, 3>& t) -> bool {
            int key = tile_key(t);
            auto it = feasible_cache.find(key);
            if (it != feasible_cache.end())
                return it->second;
            bool ok           = fits_in_shmem(method, t, kernel_W);
            feasible_cache[key] = ok;
            return ok;
        };

        // evaluate(): run kernel benchmark or return cached value.
        // Always terminates — no looping.
        auto evaluate = [&](const std::array<int, 3>& tile) -> double {
            int key = tile_key(tile);
            auto it = tp_cache.find(key);
            if (it != tp_cache.end())
                return it->second;  // cached — no increment of bo.evaluations

            if (!is_feasible(tile)) {
                // OOM: record 0, count as evaluation to prevent endless OOM probing
                tp_cache[key] = 0.0;
                ++bo.evaluations;
                if (params_.verbose && ippl::Comm->rank() == 0)
                    std::cout << "    BO [OOM] tile=(" << tile[0] << "," << tile[1] << ","
                              << tile[2] << ")\n";
                return 0.0;
            }

            auto cfg   = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = (method == "Tiled") ? ippl::Interpolation::ScatterMethod::Tiled
                                             : ippl::Interpolation::ScatterMethod::OutputFocused;
            cfg.tile_size = {tile[0], tile[1], tile[2]};
            auto r        = benchmark_scatter(method, cfg, kernel, n_particles, tile, true);
            ++bo.evaluations;
            double tp    = std::isnan(r.stats.mean_ms) ? 0.0 : r.throughput_Mpts_per_sec();
            tp_cache[key] = tp;

            if (params_.verbose && ippl::Comm->rank() == 0)
                std::cout << "    BO eval " << std::setw(4) << bo.evaluations << "  tile=("
                          << tile[0] << "," << tile[1] << "," << tile[2] << ")"
                          << "  tp=" << std::fixed << std::setprecision(1) << tp << " Mpts/s\n";
            return tp;
        };

        // Seeded RNG — different per method so two methods explore differently
        std::size_t method_hash = std::hash<std::string>{}(method);
        std::mt19937 rng(static_cast<uint32_t>(98765 + kernel.width() * 1000
                                               + (method_hash & 0xFFFF)));
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        std::uniform_int_distribution<int>     coord_dist(lo, hi);

        const int total_budget = params_.bo_budget;
        const int n_init       = std::max(4, total_budget / 5);  // 20% for LHS
        const int n_bo_steps   = total_budget - n_init;

        GPModel gp(lo, hi);

        std::array<int, 3> best_tile;
        best_tile.fill((lo + hi) / 2);
        double best_tp = 0.0;

        // ------------------------------------------------------------------
        // Phase 1: Latin Hypercube Sampling
        // ------------------------------------------------------------------
        if (ippl::Comm->rank() == 0)
            std::cout << "    BO [init] LHS initialisation (" << n_init << " evals)\n";

        {
            std::vector<int> px(n_init), py(n_init), pz(n_init);
            std::iota(px.begin(), px.end(), 0);
            std::iota(py.begin(), py.end(), 0);
            std::iota(pz.begin(), pz.end(), 0);
            std::shuffle(px.begin(), px.end(), rng);
            std::shuffle(py.begin(), py.end(), rng);
            std::shuffle(pz.begin(), pz.end(), rng);

            for (int i = 0; i < n_init && bo.evaluations < total_budget; ++i) {
                auto cell = [&](int c) -> int {
                    double frac = (c + unif(rng)) / n_init;
                    return std::clamp(lo + (int)std::round(frac * (hi - lo)), lo, hi);
                };
                std::array<int, 3> tile = {cell(px[i]), cell(py[i]), cell(pz[i])};
                double tp               = evaluate(tile);
                gp.add_observation(tile, tp);
                bo.history.emplace_back(bo.evaluations, tile[0], tile[1], tile[2], tp);
                if (tp > best_tp) { best_tp = tp; best_tile = tile; }
            }
        }

        // Pre-build full candidate list for small spaces
        std::vector<std::array<int, 3>> all_candidates;
        if (space_vol <= 5000) {
            all_candidates.reserve(space_vol);
            for (int x = lo; x <= hi; ++x)
                for (int y = lo; y <= hi; ++y)
                    for (int z = lo; z <= hi; ++z)
                        all_candidates.push_back({x, y, z});
        }

        // ------------------------------------------------------------------
        // Phase 2: Bayesian Optimisation loop
        // ------------------------------------------------------------------
        if (ippl::Comm->rank() == 0)
            std::cout << "    BO [opt] acquisition loop (" << n_bo_steps << " evals)\n";

        int bo_step = 0;
        while (bo.evaluations < total_budget) {
            gp.fit();

            // Decreasing UCB probability: start exploratory, end exploitative
            double progress = (double)bo_step / std::max(n_bo_steps, 1);
            double ucb_prob = params_.bo_ucb_prob * (1.0 - 0.7 * progress);  // e.g. 0.30→0.09
            double beta_ucb = 2.0 * (1.0 - progress) + 0.3;                  // 2.0→0.3
            bool use_ucb    = (unif(rng) < ucb_prob);

            std::array<int, 3> next_tile = best_tile;
            double best_acq              = -1e300;

            auto score = [&](const std::array<int, 3>& c) {
                if (tp_cache.count(tile_key(c)))
                    return;  // already evaluated — skip
                double acq = use_ucb ? gp.ucb(c, beta_ucb)
                                     : gp.expected_improvement(c, best_tp, params_.bo_xi);
                if (acq > best_acq) { best_acq = acq; next_tile = c; }
            };

            if (!all_candidates.empty()) {
                for (auto& c : all_candidates)
                    score(c);
            } else {
                // Random candidates
                for (int s = 0; s < 2000; ++s) {
                    std::array<int, 3> c = {coord_dist(rng), coord_dist(rng), coord_dist(rng)};
                    score(c);
                }
                // Local neighbourhood of best (ensures descent in well-explored regions)
                for (int d = 0; d < 3; ++d)
                    for (int delta : {-3, -2, -1, +1, +2, +3}) {
                        std::array<int, 3> c = best_tile;
                        c[d] = std::clamp(best_tile[d] + delta, lo, hi);
                        score(c);
                    }
            }

            // If all candidates exhausted, stop early
            if (best_acq <= -1e200) {
                if (ippl::Comm->rank() == 0)
                    std::cout << "    BO [done] space fully explored after " << bo.evaluations
                              << " evaluations\n";
                break;
            }

            double tp = evaluate(next_tile);
            gp.add_observation(next_tile, tp);
            bo.history.emplace_back(bo.evaluations, next_tile[0], next_tile[1], next_tile[2], tp);
            if (tp > best_tp) { best_tp = tp; best_tile = next_tile; }

            ++bo_step;
        }

        // ------------------------------------------------------------------
        // Phase 3: Local neighbourhood polish
        // Greedy ±1 hill-climb from best — corrects GP inaccuracy near optimum.
        // Budget limit: use at most 6 extra evaluations (cost is negligible).
        // ------------------------------------------------------------------
        if (ippl::Comm->rank() == 0)
            std::cout << "    BO [polish] hill-climb from best=(" << best_tile[0] << ","
                      << best_tile[1] << "," << best_tile[2] << ")\n";
        {
            bool improved = true;
            int polish_evals = 0;
            while (improved && polish_evals < 12) {
                improved = false;
                for (int d = 0; d < 3 && !improved; ++d) {
                    for (int delta : {-1, +1}) {
                        std::array<int, 3> c = best_tile;
                        c[d] = std::clamp(best_tile[d] + delta, lo, hi);
                        int key = tile_key(c);
                        double tp;
                        if (tp_cache.count(key)) {
                            tp = tp_cache[key];
                        } else {
                            tp = evaluate(c);
                            ++polish_evals;
                            bo.history.emplace_back(bo.evaluations, c[0], c[1], c[2], tp);
                        }
                        if (tp > best_tp) {
                            best_tp   = tp;
                            best_tile = c;
                            improved  = true;
                            break;
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Final re-measurement of best config with full statistics
        // ------------------------------------------------------------------
        bo.best_tile = best_tile;
        {
            auto cfg   = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = (method == "Tiled") ? ippl::Interpolation::ScatterMethod::Tiled
                                             : ippl::Interpolation::ScatterMethod::OutputFocused;
            cfg.tile_size = {best_tile[0], best_tile[1], best_tile[2]};
            if (is_feasible(best_tile)) {
                auto r              = benchmark_scatter(method, cfg, kernel, n_particles, best_tile, true);
                bo.best_time_ms         = r.stats.mean_ms;
                bo.best_throughput_Mpts = r.throughput_Mpts_per_sec();
            } else {
                bo.best_time_ms         = std::numeric_limits<double>::quiet_NaN();
                bo.best_throughput_Mpts = 0.0;
            }
        }

        if (ippl::Comm->rank() == 0) {
            std::cout << "  [BO] " << method << " w=" << kernel.width() << "  best tile=("
                      << best_tile[0] << "," << best_tile[1] << "," << best_tile[2] << ")"
                      << "  throughput=" << std::fixed << std::setprecision(1)
                      << bo.best_throughput_Mpts << " Mpts/s"
                      << "  (" << bo.evaluations << " evals)\n";
        }

        return bo;
    }

    // ------------------------------------------------------------------
    // Domain / particle setup / teardown
    // ------------------------------------------------------------------

    void setup_domain(int /*nghost*/) {
        for (unsigned d = 0; d < Dim; ++d)
            n_grid_[d] = params_.n_grid;

        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d)
            domain[d] = ippl::Index(n_grid_[d]);

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        layout_ =
            std::make_unique<ippl::FieldLayout<Dim>>(MPI_COMM_WORLD, domain, isParallel, true);

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
            Kokkos::parallel_for(
                "init_uniform", n_local, KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d)
                        R_view(i)[d] = gen.drand() * 2.0 * M_PI;
                    rand_pool.free_state(gen);
                });
        } else if (params_.distribution == "clustered") {
            Kokkos::parallel_for(
                "init_clustered", n_local, KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        double u1 = gen.drand(), u2 = gen.drand();
                        double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-10))
                                   * Kokkos::cos(2.0 * M_PI * u2);
                        R_view(i)[d] = M_PI + 0.3 * z;
                        while (R_view(i)[d] < 0)
                            R_view(i)[d] += 2.0 * M_PI;
                        while (R_view(i)[d] >= 2.0 * M_PI)
                            R_view(i)[d] -= 2.0 * M_PI;
                    }
                    rand_pool.free_state(gen);
                });
        }

        auto Q_view = Q_.getView();
        Kokkos::parallel_for(
            "init_values", n_local, KOKKOS_LAMBDA(const size_t i) { Q_view(i) = one(); });

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
        if (ippl::Comm->rank() != 0)
            return;
        std::cout << "\n"
                  << "================================================================\n"
                  << "     Tile Size × Kernel Width Sweep Benchmark\n"
                  << "================================================================\n"
                  << "Grid size:       " << params_.n_grid << "^3\n"
                  << "Particles/grid:  " << params_.rho << "\n"
                  << "Total particles: " << params_.n_particles() << "\n"
                  << "Distribution:    " << params_.distribution << "\n"
                  << "Value type:      " << value_type_str() << "\n"
                  << "Tile sizes:      " << params_.min_tile_size << " - " << params_.max_tile_size
                  << "\n"
                  << "Kernel widths:   " << params_.min_kernel_width << " - "
                  << params_.max_kernel_width << "\n"
                  << "Warmup runs:     " << params_.warmup_runs << "\n"
                  << "Benchmark runs:  " << params_.benchmark_runs << "\n";
        if (params_.optimize)
            std::cout << "BO optimiser:    enabled  (budget=" << params_.bo_budget
                      << ", xi=" << params_.bo_xi
                      << ", ucb_prob=" << params_.bo_ucb_prob << ")\n";
        std::cout << "================================================================\n\n";
    }

    void write_full_csv(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0)
            return;

        std::string filename = params_.output_prefix + "_full.csv";
        std::ofstream out(filename);

        out << "method,distribution,value_type,"
            << "tile_x,tile_y,tile_z,kernel_width,n_particles,n_grid,rho,"
            << "mean_ms,stddev_ms,min_ms,max_ms,median_ms,"
            << "throughput_Mpts_s,time_per_pt_ns,from_optimizer,status\n";

        for (const auto& r : results) {
            out << r.method << "," << r.distribution << "," << r.value_type << ","
                << r.tile_sizes[0] << "," << r.tile_sizes[1] << "," << r.tile_sizes[2] << ","
                << r.kernel_width << "," << r.n_particles << "," << r.n_grid << "," << std::fixed
                << std::setprecision(1) << r.rho << ",";

            if (std::isnan(r.stats.mean_ms)) {
                out << "nan,nan,nan,nan,nan,nan,nan," << (r.from_optimizer ? "1" : "0")
                    << ",failed\n";
            } else {
                out << std::setprecision(4) << r.stats.mean_ms << "," << r.stats.stddev_ms << ","
                    << r.stats.min_ms << "," << r.stats.max_ms << "," << r.stats.median_ms << ","
                    << std::setprecision(2) << r.throughput_Mpts_per_sec() << ","
                    << r.time_per_point_ns() << "," << (r.from_optimizer ? "1" : "0") << ",ok\n";
            }
        }
        out.close();
        std::cout << "Wrote full results to: " << filename << "\n";
    }

    void write_heatmap_csv(const std::vector<BenchmarkResult>& results,
                           const std::string& method) {
        if (ippl::Comm->rank() != 0)
            return;

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
        for (int w : widths)
            out << ",width_" << w;
        out << "\n";

        for (int t : tiles) {
            out << t;
            for (int w : widths) {
                bool found = false;
                for (const auto& r : results) {
                    if (r.method == method && !r.from_optimizer && r.tile_sizes[0] == t
                        && r.kernel_width == w) {
                        out << (std::isnan(r.stats.mean_ms)
                                    ? ",nan"
                                    : "," + std::to_string(r.throughput_Mpts_per_sec()));
                        found = true;
                        break;
                    }
                }
                if (!found)
                    out << ",nan";
            }
            out << "\n";
        }
        out.close();
        std::cout << "Wrote heatmap for " << method << " to: " << filename << "\n";
    }

    void write_optimal_csv(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0)
            return;

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
                double best_tp              = 0.0;
                for (const auto& r : results) {
                    if (r.method == method && r.kernel_width == w && !r.from_optimizer
                        && !std::isnan(r.stats.mean_ms)) {
                        double tp = r.throughput_Mpts_per_sec();
                        if (tp > best_tp) { best_tp = tp; best = &r; }
                    }
                }
                if (best)
                    out << method << "," << w << "," << best->tile_sizes[0] << "," << std::fixed
                        << std::setprecision(2) << best->throughput_Mpts_per_sec() << ","
                        << std::setprecision(4) << best->stats.mean_ms << "\n";
                else
                    out << method << "," << w << ",nan,nan,nan\n";
            }
        }
        out.close();
        std::cout << "Wrote optimal configurations to: " << filename << "\n";
    }

    // write_sa_csv / write_sa_history_csv kept with original names for CSV compat
    void write_sa_csv(const std::vector<SAResult>& sa_results) {
        if (ippl::Comm->rank() != 0)
            return;

        std::string filename = params_.output_prefix + "_sa_optimal.csv";
        std::ofstream out(filename);
        out << "method,value_type,kernel_width,"
            << "best_tile_x,best_tile_y,best_tile_z,"
            << "throughput_Mpts_s,time_ms,evaluations\n";

        for (const auto& r : sa_results) {
            out << r.method << "," << r.value_type << "," << r.kernel_width << ","
                << r.best_tile[0] << "," << r.best_tile[1] << "," << r.best_tile[2] << ","
                << std::fixed << std::setprecision(2) << r.best_throughput_Mpts << ","
                << std::setprecision(4) << r.best_time_ms << "," << r.evaluations << "\n";
        }
        out.close();
        std::cout << "Wrote BO optimal results to: " << filename << "\n";
    }

    void write_sa_history_csv(const std::vector<SAResult>& sa_results) {
        if (ippl::Comm->rank() != 0)
            return;

        std::string filename = params_.output_prefix + "_sa_history.csv";
        std::ofstream out(filename);
        out << "method,value_type,kernel_width,step,tile_x,tile_y,tile_z,throughput_Mpts_s\n";

        for (const auto& r : sa_results) {
            for (const auto& [step, tx, ty, tz, tp] : r.history) {
                out << r.method << "," << r.value_type << "," << r.kernel_width << "," << step
                    << "," << tx << "," << ty << "," << tz << "," << std::fixed
                    << std::setprecision(2) << tp << "\n";
            }
        }
        out.close();
        std::cout << "Wrote BO convergence history to: " << filename << "\n";
    }

    void print_summary(const std::vector<BenchmarkResult>& results,
                       const std::vector<SAResult>& sa_results) {
        if (ippl::Comm->rank() != 0)
            return;

        int failed = 0;
        for (const auto& r : results)
            if (std::isnan(r.stats.mean_ms))
                ++failed;

        std::cout << "\n"
                  << "================================================================\n"
                  << "                    Results Summary (" << value_type_str() << ")\n"
                  << "================================================================\n";
        if (failed > 0)
            std::cout << "\nNote: " << failed << " configuration(s) failed.\n";

        std::vector<std::string> methods = {"Tiled", "OutputFocused"};
        std::vector<int> widths;
        for (const auto& r : results) {
            if (!r.from_optimizer
                && std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end())
                widths.push_back(r.kernel_width);
        }
        std::sort(widths.begin(), widths.end());

        for (const auto& method : methods) {
            std::cout << "\n"
                      << method << " — optimal uniform tile by kernel width:\n"
                      << std::string(60, '-') << "\n"
                      << std::left << std::setw(8) << "Width" << std::right << std::setw(12)
                      << "Best Tile" << std::setw(14) << "Mpts/s" << std::setw(12)
                      << "Time (ms)\n"
                      << std::string(60, '-') << "\n";

            for (int w : widths) {
                const BenchmarkResult* best = nullptr;
                double best_tp              = 0.0;
                for (const auto& r : results) {
                    if (r.method == method && r.kernel_width == w && !r.from_optimizer
                        && !std::isnan(r.stats.mean_ms)) {
                        double tp = r.throughput_Mpts_per_sec();
                        if (tp > best_tp) { best_tp = tp; best = &r; }
                    }
                }
                if (best)
                    std::cout << std::left << std::setw(8) << w << std::right << std::setw(12)
                              << best->tile_sizes[0] << std::fixed << std::setprecision(1)
                              << std::setw(14) << best->throughput_Mpts_per_sec()
                              << std::setprecision(3) << std::setw(12) << best->stats.mean_ms
                              << "\n";
                else
                    std::cout << std::left << std::setw(8) << w << std::right << std::setw(38)
                              << "all failed\n";
            }
        }

        if (!sa_results.empty()) {
            std::cout << "\n"
                      << "================================================================\n"
                      << "        BO-Optimised Rectangular Tile Sizes\n"
                      << "================================================================\n"
                      << std::left << std::setw(16) << "Method" << std::right << std::setw(8)
                      << "Width" << std::setw(22) << "Best tile (x,y,z)" << std::setw(14)
                      << "Mpts/s" << std::setw(10) << "Evals\n"
                      << std::string(70, '-') << "\n";

            for (const auto& r : sa_results) {
                std::ostringstream ts;
                ts << "(" << r.best_tile[0] << "," << r.best_tile[1] << "," << r.best_tile[2]
                   << ")";
                std::cout << std::left << std::setw(16) << r.method << std::right << std::setw(8)
                          << r.kernel_width << std::setw(22) << ts.str() << std::fixed
                          << std::setprecision(1) << std::setw(14) << r.best_throughput_Mpts
                          << std::setw(10) << r.evaluations << "\n";
            }
        }

        std::cout << "\n";
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
    ippl::ParticleAttrib<value_type> Q_;
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
            TileSweepBenchmark<Kokkos::DefaultExecutionSpace, Kokkos::complex<double>> bench(
                params);
            bench.run();
        }
    }

    ippl::finalize();
    return EXIT_SUCCESS;
}