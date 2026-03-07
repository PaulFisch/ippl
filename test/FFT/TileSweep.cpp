/**
 * @file BenchmarkTileSweep.cpp
 * @brief Performance benchmark sweeping tile size and kernel width
 *
 * Generates data for:
 *   Plot 1: Performance vs tile size (fixed kernel width)
 *   Plot 2: Performance vs kernel width (fixed tile size)
 *   Plot 3: Heatmap of performance across tile size x kernel width
 *
 * Usage: ./BenchmarkTileSweep [options]
 *   --grid N              Grid size per dimension (default: 128)
 *   --rho R               Particles per grid point (default: 1.0)
 *   --warmup N            Number of warmup runs (default: 3)
 *   --runs N              Number of benchmark runs (default: 5)
 *   --output FILE         Output CSV file prefix (default: tile_sweep)
 *   --min-tile T          Minimum tile size per dimension (default: 1)
 *   --max-tile T          Maximum tile size per dimension (default: 8)
 *   --min-width W         Minimum kernel width (default: 2)
 *   --max-width W         Maximum kernel width (default: 8)
 *   --dist D              Particle distribution: uniform, clustered (default: uniform)
 *   --ncu-mode            Single run mode for Nsight Compute profiling
 *   --real                Use real-valued field and particles instead of complex
 *   --optimize            Run Bayesian Optimisation over all config parameters
 *                         (always optimizes BOTH real and complex value types)
 *   --bo-budget N         BO: total number of kernel evaluations (default: 80)
 *   --bo-xi X             BO: EI exploration parameter xi (default: 0.01)
 *   --bo-ucb-prob P       BO: probability of using UCB vs EI (default: 0.3)
 *   --team-sizes LIST     Comma-separated list of Tiled thread counts to try (default: 8,16,32,64)
 *   --gp-team-sizes LIST  Comma-separated list of OutputFocused warp counts to try (default:
 * 1,2,3,4,8)
 *   --min-osub N          Minimum oversubscription factor (default: 1)
 *   --max-osub N          Maximum oversubscription factor (default: 8)
 *   --min-z-batches N     Minimum z-stencil batches for OutputFocused (default: 1)
 *   --max-z-batches N     Maximum z-stencil batches for OutputFocused (default: 4)
 *   -v, --verbose         Verbose output
 *
 * Note: --sa-steps is accepted as an alias for --bo-budget (backward compat).
 *
 * Team-size conventions:
 *   Tiled              : team_size = actual GPU thread count (e.g. 8, 16, 32, 64)
 *   OutputFocused (GP) : team_size = number of warps          (e.g. 1, 2, 3, 4, 8)
 *                        → real thread count = team_size * warp_size (32)
 *   is_config_valid() and the hardware thread-limit check account for this via
 *   actual_threads(method, team_size).
 *
 * CHANGES vs original
 * -------------------
 * Fix 1 (critical):  build_L() had a double-subtraction on the Cholesky diagonal.
 *   The outer `for k < j` loop (with j==i) already accumulates Σ L[i][k]² for k=0..i-1;
 *   a second inner loop then subtracted the same terms again, driving s deeply negative
 *   and causing every diagonal pivot to fall back to the 1e-8 guard.  The extra loop
 *   is removed; build_L() now matches lml_compute() exactly.
 *
 * Fix 2 (important): pre-flight infeasibility no longer charges the benchmark budget.
 *   evaluate() previously incremented bo.evaluations for cheap shmem/thread-count
 *   checks that cost nothing to compute.  Only actual kernel launches now count.
 *   A separate bo.preflight_rejections counter is kept for diagnostics.
 *
 * Fix 3 (important): duplicate GP observations are now guarded.
 *   LHS cell-rounding can map two different stratum draws to the same SearchPoint.
 *   Adding the same (x,y) pair twice makes K rank-deficient.  An observed_gp set
 *   now gates gp.add_observation() in both the LHS and acquisition phases.
 *
 * Fix 4 (minor): Tiled method uses a 5-D GP (z_batches dimension suppressed).
 *   For Tiled, lo_zb == hi_zb == 1, making dimension 5 degenerate (all normalised
 *   to 0) and wasting GP capacity.  run_bo() now dispatches to a templated helper
 *   that instantiates GPModel<5> for Tiled and GPModel<6> for OutputFocused.
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
#include <sstream>
#include <unordered_map>
#include <unordered_set>
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
        return std::chrono::duration<double>(clock_type::now() - start_time_).count();
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

    int min_tile_size    = 1;
    int max_tile_size    = 8;
    int min_kernel_width = 2;
    int max_kernel_width = 8;

    // BO parameters
    int bo_budget      = 80;
    double bo_xi       = 0.01;
    double bo_ucb_prob = 0.30;

    // GPU warp size — used to convert OutputFocused warp counts to thread counts
    // for hardware-limit checks.
    static constexpr int warp_size = 32;

    // Tiled: team_size = actual GPU thread count (8, 16, 32, 64, ...)
    std::vector<int> team_size_candidates = {8, 16, 32, 64};

    // OutputFocused / GridParallelScatter: team_size = number of warps (1, 2, 3, 4, 8, ...)
    // The real thread count seen by the hardware is warp_count * warp_size.
    std::vector<int> gp_team_size_candidates = {1, 2, 3, 4, 8};

    // oversubscription_factor range
    int min_osub = 1;
    int max_osub = 8;

    // z_batches range (for GridParallelScatter z-stencil batching)
    // 1 = no batching (default), larger values reduce shared memory pressure
    // For optimization mode, we default to max=4 to help find feasible configs for large widths
    int min_z_batches = 1;
    int max_z_batches = 4;  // default: allow up to 4 z-batches in optimization

    // Internal: set true for the complementary-type BO pass so that the inner
    // benchmark does not itself recurse into another complementary run.
    bool complementary_bo_pass = false;

    size_t n_particles() const { return static_cast<size_t>(rho * n_grid * n_grid * n_grid); }
};

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try {
            out.push_back(std::stoi(tok));
        } catch (...) {
        }
    }
    return out;
}

BenchParams parse_bench_args(int argc, char* argv[]) {
    BenchParams p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--grid" || a == "-n") && i + 1 < argc)
            p.n_grid = std::atoi(argv[++i]);
        else if (a == "--rho" && i + 1 < argc)
            p.rho = std::atof(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc)
            p.warmup_runs = std::atoi(argv[++i]);
        else if (a == "--runs" && i + 1 < argc)
            p.benchmark_runs = std::atoi(argv[++i]);
        else if ((a == "--output" || a == "-o") && i + 1 < argc)
            p.output_prefix = argv[++i];
        else if (a == "--dist" && i + 1 < argc)
            p.distribution = argv[++i];
        else if (a == "--min-tile" && i + 1 < argc)
            p.min_tile_size = std::atoi(argv[++i]);
        else if (a == "--max-tile" && i + 1 < argc)
            p.max_tile_size = std::atoi(argv[++i]);
        else if (a == "--min-width" && i + 1 < argc)
            p.min_kernel_width = std::atoi(argv[++i]);
        else if (a == "--max-width" && i + 1 < argc)
            p.max_kernel_width = std::atoi(argv[++i]);
        else if (a == "--ncu-mode") {
            p.ncu_mode       = true;
            p.warmup_runs    = 1;
            p.benchmark_runs = 1;
        } else if (a == "--real")
            p.use_real = true;
        else if (a == "--optimize")
            p.optimize = true;
        else if ((a == "--bo-budget" || a == "--sa-steps") && i + 1 < argc)
            p.bo_budget = std::atoi(argv[++i]);
        else if (a == "--bo-xi" && i + 1 < argc)
            p.bo_xi = std::atof(argv[++i]);
        else if (a == "--bo-ucb-prob" && i + 1 < argc)
            p.bo_ucb_prob = std::atof(argv[++i]);
        else if (a == "--team-sizes" && i + 1 < argc) {
            auto v = parse_int_list(argv[++i]);
            if (!v.empty())
                p.team_size_candidates = v;
        } else if (a == "--gp-team-sizes" && i + 1 < argc) {
            auto v = parse_int_list(argv[++i]);
            if (!v.empty())
                p.gp_team_size_candidates = v;
        } else if (a == "--min-osub" && i + 1 < argc)
            p.min_osub = std::atoi(argv[++i]);
        else if (a == "--max-osub" && i + 1 < argc)
            p.max_osub = std::atoi(argv[++i]);
        else if (a == "--min-z-batches" && i + 1 < argc)
            p.min_z_batches = std::atoi(argv[++i]);
        else if (a == "--max-z-batches" && i + 1 < argc)
            p.max_z_batches = std::atoi(argv[++i]);
        else if (a == "-v" || a == "--verbose")
            p.verbose = true;
        else if ((a == "--sa-t0" || a == "--sa-alpha") && i + 1 < argc)
            ++i;  // ignored
    }
    return p;
}

// ============================================================================
// Statistics
// ============================================================================

struct TimingStats {
    double mean_ms = 0, stddev_ms = 0, min_ms = 0, max_ms = 0, median_ms = 0;
    size_t count = 0;
};

TimingStats compute_stats(const std::vector<double>& times_sec) {
    TimingStats s{};
    s.count = times_sec.size();
    if (s.count == 0)
        return s;
    std::vector<double> ms(s.count);
    for (size_t i = 0; i < s.count; ++i)
        ms[i] = times_sec[i] * 1000.0;
    double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
    s.mean_ms  = sum / s.count;
    double sq  = 0;
    for (double t : ms)
        sq += (t - s.mean_ms) * (t - s.mean_ms);
    s.stddev_ms                = (s.count > 1) ? std::sqrt(sq / (s.count - 1)) : 0.0;
    s.min_ms                   = *std::min_element(ms.begin(), ms.end());
    s.max_ms                   = *std::max_element(ms.begin(), ms.end());
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    s.median_ms = (s.count % 2 == 0) ? (sorted[s.count / 2 - 1] + sorted[s.count / 2]) / 2.0
                                     : sorted[s.count / 2];
    return s;
}

// ============================================================================
// Benchmark / Optimiser Result types
// ============================================================================

struct BenchmarkResult {
    std::string method, distribution, value_type;
    std::array<int, 3> tile_sizes = {1, 1, 1};
    int tile_size = 1, team_size = 16, oversubscription_factor = 4, z_batches = 1, kernel_width = 0;
    size_t n_particles = 0, n_grid = 0;
    double rho          = 0;
    bool from_optimizer = false;
    TimingStats stats;
    std::vector<double> times_sec;
    double throughput_Mpts_per_sec() const { return (n_particles / (stats.mean_ms * 1e-3)) / 1e6; }
    double time_per_point_ns() const { return (stats.mean_ms * 1e6) / n_particles; }
};

struct BOResult {
    std::string method, value_type;
    int kernel_width             = 0;
    std::array<int, 3> best_tile = {1, 1, 1};
    int best_team_size = 16, best_oversubscription_factor = 4, best_z_batches = 1;
    double best_throughput_Mpts = 0, best_time_ms = 0;
    int evaluations          = 0;
    int preflight_rejections = 0;  // Fix 2: separate counter for cheap pre-flight checks
    // (eval_count, tx, ty, tz, team_size, osub, z_batches, throughput)
    std::vector<std::tuple<int, int, int, int, int, int, int, double>> history;
};

// ============================================================================
// N-dimensional Gaussian Process  (RBF kernel, self-contained, no deps)
// ============================================================================
//
// Template parameter N = dimensionality of the search space.
// Each dimension has independent [lo_d, hi_d] bounds for normalisation,
// ensuring the isotropic RBF kernel is geometrically meaningful across
// axes with very different scales (e.g. tile in [1,30] vs ts_idx in [0,3]).
//
// Hyperparameters (l, sigma_n) chosen by marginal likelihood grid search.
//
template <int N>
struct GPModel {
    using Point = std::array<int, N>;

    std::vector<Point> X;
    std::vector<double> y;
    double y_mean = 0, y_std = 1;
    std::vector<double> y_norm;

    double length_scale = 0.4, sigma_n = 0.1;

    std::array<int, N> lo_bounds, hi_bounds;
    std::vector<std::vector<double>> L;  // lower-triangular Cholesky
    std::vector<double> alpha;           // K^{-1} y_norm

    GPModel() {
        lo_bounds.fill(0);
        hi_bounds.fill(1);
    }

    void set_bounds(const std::array<int, N>& lo, const std::array<int, N>& hi) {
        lo_bounds = lo;
        hi_bounds = hi;
    }

    std::array<double, N> normalise(const Point& p) const {
        std::array<double, N> out;
        for (int d = 0; d < N; ++d) {
            double range = std::max(hi_bounds[d] - lo_bounds[d], 1);
            out[d]       = (p[d] - lo_bounds[d]) / range;
        }
        return out;
    }

    double sq_dist(const std::array<double, N>& a, const std::array<double, N>& b) const {
        double s = 0;
        for (int d = 0; d < N; ++d)
            s += (a[d] - b[d]) * (a[d] - b[d]);
        return s;
    }

    double kernel_val(const std::array<double, N>& a, const std::array<double, N>& b,
                      double l) const {
        return std::exp(-sq_dist(a, b) / (2.0 * l * l));
    }

    // Fit: normalise y, hyperparameter MLE grid search, build Cholesky + alpha.
    void fit() {
        int n = (int)y.size();
        if (n == 0)
            return;

        // Normalise from all observations (including penalties) so the GP
        // correctly models the boundary between feasible and infeasible regions.
        y_mean     = std::accumulate(y.begin(), y.end(), 0.0) / n;
        double var = 0;
        for (double v : y)
            var += (v - y_mean) * (v - y_mean);
        y_std = std::sqrt(var / n + 1e-12);

        y_norm.resize(n);
        for (int i = 0; i < n; ++i)
            y_norm[i] = (y[i] - y_mean) / y_std;

        // Grid search — denser l grid for higher-D spaces
        double best_lml = -1e300;
        for (double l : {0.03, 0.07, 0.15, 0.3, 0.6, 1.2, 2.5}) {
            for (double sn : {0.005, 0.02, 0.08, 0.2, 0.5}) {
                double lml = lml_compute(l, sn);
                if (lml > best_lml) {
                    best_lml     = lml;
                    length_scale = l;
                    sigma_n      = sn;
                }
            }
        }
        build_L(length_scale, sigma_n, L);
        solve_alpha();
    }

    double lml_compute(double l, double sn) const {
        int n = (int)X.size();
        std::vector<std::vector<double>> Lc(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            auto xi = normalise(X[i]);
            for (int j = 0; j <= i; ++j) {
                auto xj  = normalise(X[j]);
                double s = kernel_val(xi, xj, l);
                if (i == j)
                    s += sn * sn + 1e-8;
                for (int k = 0; k < j; ++k)
                    s -= Lc[i][k] * Lc[j][k];
                if (i == j) {
                    if (s <= 0)
                        return -1e300;
                    Lc[i][j] = std::sqrt(s);
                } else
                    Lc[i][j] = s / (Lc[j][j] > 1e-12 ? Lc[j][j] : 1e-12);
            }
        }
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
        double fit = 0, ldet = 0;
        for (int i = 0; i < n; ++i) {
            fit += y_norm[i] * alph[i];
            ldet += std::log(std::max(Lc[i][i], 1e-300));
        }
        return -0.5 * fit - ldet - 0.5 * n * std::log(2 * M_PI);
    }

    // FIX 1: Removed the erroneous second inner loop that was double-subtracting
    // the diagonal accumulation.  The outer `for k < j` (with j==i) already
    // computes Σ_{k<i} L[i][k]², matching lml_compute() exactly.
    void build_L(double l, double sn, std::vector<std::vector<double>>& Lout) const {
        int n = (int)X.size();
        Lout.assign(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            auto xi = normalise(X[i]);
            for (int j = 0; j <= i; ++j) {
                auto xj  = normalise(X[j]);
                double s = kernel_val(xi, xj, l);
                if (i == j)
                    s += sn * sn + 1e-8;
                // Accumulate Σ_{k<j} L[i][k]*L[j][k]  (for diagonal: Σ_{k<i} L[i][k]²)
                for (int k = 0; k < j; ++k)
                    s -= Lout[i][k] * Lout[j][k];
                if (i == j) {
                    // ← original code had a second identical loop here — removed (Fix 1)
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

    std::pair<double, double> predict(const Point& p) const {
        int n = (int)X.size();
        if (n == 0)
            return {y_mean, y_std * y_std};
        auto xt = normalise(p);
        std::vector<double> k(n);
        for (int i = 0; i < n; ++i)
            k[i] = kernel_val(normalise(X[i]), xt, length_scale);
        double mu_norm = 0;
        for (int i = 0; i < n; ++i)
            mu_norm += k[i] * alpha[i];
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) {
            double s = k[i];
            for (int j = 0; j < i; ++j)
                s -= L[i][j] * v[j];
            v[i] = s / (L[i][i] > 1e-12 ? L[i][i] : 1e-12);
        }
        double var = 1.0;
        for (int i = 0; i < n; ++i)
            var -= v[i] * v[i];
        var = std::max(var, 1e-10);
        return {mu_norm * y_std + y_mean, var * y_std * y_std};
    }

    double expected_improvement(const Point& p, double f_best, double xi = 0.01) const {
        auto [mu, var] = predict(p);
        double sigma   = std::sqrt(std::max(var, 0.0));
        if (sigma < 1e-10)
            return std::max(0.0, mu - f_best);
        double imp = mu - f_best - xi * std::max(y_std, 1.0), Z = imp / sigma;
        double Phi = 0.5 * (1.0 + std::erf(Z / std::sqrt(2.0)));
        double phi = std::exp(-0.5 * Z * Z) / std::sqrt(2.0 * M_PI);
        return std::max(0.0, imp * Phi + sigma * phi);
    }

    double ucb(const Point& p, double beta = 2.0) const {
        auto [mu, var] = predict(p);
        return mu + beta * std::sqrt(std::max(var, 0.0));
    }

    void add_observation(const Point& p, double tp) {
        X.push_back(p);
        y.push_back(tp);
    }
};

// ============================================================================
// SearchPoint: 6D config point (5D for Tiled — z_batches fixed to 1)
// ============================================================================
//
// Dimensions:
//   [0] tile_x       in [min_tile, max_tile]
//   [1] tile_y       in [min_tile, max_tile]
//   [2] tile_z       in [min_tile, max_tile]
//   [3] ts_idx       in [0, n_team_opts-1]
//   [4] osub         in [min_osub, max_osub]
//   [5] z_batches    in [lo_zb, hi_zb]   (always 1 for Tiled → dim suppressed in GP)
//
struct SearchPoint {
    std::array<int, 6> v = {1, 1, 1, 0, 1, 1};
    int tile_x() const { return v[0]; }
    int tile_y() const { return v[1]; }
    int tile_z() const { return v[2]; }
    int ts_idx() const { return v[3]; }
    int osub() const { return v[4]; }
    int z_batches() const { return v[5]; }
    std::array<int, 3> tile() const { return {v[0], v[1], v[2]}; }
    bool operator==(const SearchPoint& o) const { return v == o.v; }
};

struct SearchPointHash {
    std::size_t operator()(const SearchPoint& p) const noexcept {
        std::size_t h = 0;
        for (int x : p.v)
            h = h * 104729 + static_cast<std::size_t>(x + 100);
        return h;
    }
};

// ============================================================================
// Core Benchmark Engine
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

    // The other value type (real ↔ complex), used for the dual-type BO pass.
    using other_value_type =
        std::conditional_t<is_complex, real_type, Kokkos::complex<real_type>>;

    // Returns the maximum scratch memory available for a given team configuration.
    static size_t scratch_size_max_for_team(const std::string& method, int team_size) {
        using team_policy = Kokkos::TeamPolicy<ExecSpace>;
        if (method == "OutputFocused") {
            return team_policy(1, team_size, 32).scratch_size_max(0);
        } else {
            return team_policy(1, team_size).scratch_size_max(0);
        }
    }

    using DummyFieldView = typename Field_t::view_type;
    using DummyPosView   = typename ippl::ParticleAttrib<ippl::Vector<real_type, Dim>>::view_type;
    using DummyValView   = typename ippl::ParticleAttrib<value_type>::view_type;

    template <int W>
    using TiledTypes =
        ippl::Interpolation::detail::ScatterTypes<Dim, real_type, ippl::nufft::ESKernel<real_type>,
                                                  DummyFieldView, DummyPosView, DummyValView>;
    template <int W>
    using GPTypes      = TiledTypes<W>;
    using SortedPolicy = ippl::Interpolation::detail::SortedPolicy;

    template <int W, bool ForceComplex = is_complex>
    static size_t required_shmem_tiled(const ippl::Vector<int, Dim>& tv, int team_size) {
        return ippl::Interpolation::detail::TiledScatter<
            W, TiledTypes<W>, SortedPolicy>::template compute_scratch_size<ForceComplex>(tv,
                                                                                         team_size);
    }

    template <int W, bool ForceComplex = is_complex>
    static size_t required_shmem_gp(const ippl::Vector<int, Dim>& tv, int team_size_warps,
                                    int z_batches = 1) {
        return ippl::Interpolation::detail::GridParallelScatter<W, GPTypes<W>, SortedPolicy>::
            template compute_scratch_size<ForceComplex>(tv, team_size_warps, z_batches);
    }

    static size_t required_shmem(const std::string& method, const std::array<int, 3>& tile, int W,
                                 int team_size, int z_batches = 1, bool force_complex = false) {
        ippl::Vector<int, Dim> tv;
        for (unsigned d = 0; d < Dim; ++d)
            tv[d] = tile[d];
        size_t result = std::numeric_limits<size_t>::max();
        ippl::Interpolation::WidthDispatcher<1, 14>::dispatch(W, [&]<int Wc>() {
            if (force_complex) {
                result = (method == "Tiled")
                             ? required_shmem_tiled<Wc, true>(tv, team_size)
                             : required_shmem_gp<Wc, true>(tv, team_size, z_batches);
            } else {
                result = (method == "Tiled")
                             ? required_shmem_tiled<Wc>(tv, team_size)
                             : required_shmem_gp<Wc>(tv, team_size, z_batches);
            }
        });
        return result;
    }

    static int actual_threads(const std::string& method, int team_size) {
        return (method == "OutputFocused") ? team_size * BenchParams::warp_size : team_size;
    }

    static int max_team_size() {
        static int cached = -1;
        if (cached >= 0)
            return cached;
#if defined(KOKKOS_ENABLE_CUDA)
        int dev = 0;
        cudaGetDevice(&dev);
        int v = 0;
        cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerBlock, dev);
        cached = (v > 0) ? v : 1024;
#elif defined(KOKKOS_ENABLE_HIP)
        int dev = 0;
        hipGetDevice(&dev);
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, dev);
        cached = (int)prop.maxThreadsPerBlock;
#else
        cached = std::numeric_limits<int>::max();
#endif
        return cached;
    }

    bool fits_in_shmem(const std::string& method, const std::array<int, 3>& tile, int W,
                       int team_size, int z_batches = 1, bool force_complex = false) const {
        size_t req   = required_shmem(method, tile, W, team_size, z_batches, force_complex);
        size_t avail = scratch_size_max_for_team(method, team_size);
        if (params_.verbose && ippl::Comm->rank() == 0)
            std::cout << "  [shmem] " << method << " tile=(" << tile[0] << "," << tile[1] << ","
                      << tile[2] << ") W=" << W << " team=" << team_size
                      << " z_batches=" << z_batches << " req=" << req << " avail=" << avail
                      << (req <= avail ? " OK" : " SKIP") << "\n";
        return req <= avail;
    }

    bool is_config_valid(const std::string& method, const std::array<int, 3>& tile, int W,
                         int team_size, int z_batches = 1, bool force_complex = false) const {
        int threads = actual_threads(method, team_size);
        if (threads > max_team_size()) {
            if (params_.verbose && ippl::Comm->rank() == 0)
                std::cout << "  [team-skip] " << method << " team=" << team_size
                          << " → threads=" << threads << " > max=" << max_team_size() << "\n";
            return false;
        }
        if (!fits_in_shmem(method, tile, W, team_size, z_batches, false))
            return false;
        if (force_complex && !is_complex)
            if (!fits_in_shmem(method, tile, W, team_size, z_batches, true))
                return false;
        return true;
    }

    bool is_config_valid_both_types(const std::string& method, const std::array<int, 3>& tile,
                                    int W, int team_size) const {
        return is_config_valid(method, tile, W, team_size, /*force_complex=*/true);
    }

    size_t get_n_local_particles() const {
        return bunch_ ? bunch_->getLocalNum() : 0;
    }

    explicit TileSweepBenchmark(const BenchParams& params)
        : params_(params) {}

    // ------------------------------------------------------------------
    // Top-level entry point
    // ------------------------------------------------------------------
    void run() {
        print_header();
        std::vector<BenchmarkResult> results;
        std::vector<BOResult> bo_results;

        std::vector<int> widths, tile_sizes_1d;
        for (int w = params_.min_kernel_width; w <= params_.max_kernel_width; ++w)
            widths.push_back(w);
        for (int t = params_.min_tile_size; t <= params_.max_tile_size; ++t)
            tile_sizes_1d.push_back(t);

        const int total_configs = (int)widths.size() * (int)tile_sizes_1d.size() * 2;
        int current_config      = 0;

        for (int width : widths) {
            double tol = std::pow(10.0, -(width - 1));
            ippl::nufft::ESKernel<real_type> kernel(tol);
            int actual_width = kernel.width();
            if (actual_width != width && ippl::Comm->rank() == 0)
                std::cout << "Note: Requested width " << width << ", got " << actual_width << "\n";

            int nghost = actual_width / 2 + 1;
            setup_domain(nghost);
            initialize(kernel, nghost);
            size_t n_particles = bunch_->getLocalNum();

            // ---- Uniform tile sweep ----------------------------------------
            for (int t : tile_sizes_1d) {
                for (const std::string& method : {"Tiled", "OutputFocused"}) {
                    ++current_config;
                    if (ippl::Comm->rank() == 0)
                        std::cout << "\r[sweep " << current_config << "/" << total_configs << "] "
                                  << "width=" << actual_width << ", tile=" << t << ", " << method
                                  << "     " << std::flush;

                    const std::array<int, 3> ta = {t, t, t};

                    auto cfg   = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
                    cfg.method = (method == "Tiled")
                                     ? ippl::Interpolation::ScatterMethod::Tiled
                                     : ippl::Interpolation::ScatterMethod::OutputFocused;
                    cfg.set_tile_size(t);

                    if (is_config_valid(method, ta, actual_width, cfg.team_size))
                        results.push_back(
                            benchmark_scatter(method, cfg, kernel, n_particles, ta, false));
                    else if (params_.verbose && ippl::Comm->rank() == 0)
                        std::cout << "\n  [skip] " << method << " tile=" << t << "\n";
                }
            }

            // ---- Bayesian Optimisation  ------------------------------------
            if (params_.optimize) {
                if (ippl::Comm->rank() == 0)
                    std::cout << "\n  [BO] width=" << actual_width
                              << " budget=" << params_.bo_budget << "\n";

                for (const std::string& method : {"Tiled", "OutputFocused"}) {
                    auto bo = run_bo(method, kernel, n_particles);
                    bo_results.push_back(bo);
                    const auto& bt = bo.best_tile;
                    if (is_config_valid(method, bt, actual_width, bo.best_team_size,
                                        bo.best_z_batches)) {
                        auto cfg =
                            ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
                        cfg.method = (method == "Tiled")
                                         ? ippl::Interpolation::ScatterMethod::Tiled
                                         : ippl::Interpolation::ScatterMethod::OutputFocused;
                        cfg.tile_size               = {bt[0], bt[1], bt[2]};
                        cfg.team_size               = bo.best_team_size;
                        cfg.oversubscription_factor = bo.best_oversubscription_factor;
                        cfg.z_batches               = bo.best_z_batches;
                        auto r      = benchmark_scatter(method, cfg, kernel, n_particles, bt, true);
                        r.team_size = bo.best_team_size;
                        r.oversubscription_factor = bo.best_oversubscription_factor;
                        r.z_batches               = bo.best_z_batches;
                        results.push_back(r);
                    }
                }

                if (!params_.complementary_bo_pass) {
                    if (ippl::Comm->rank() == 0)
                        std::cout << "  [BO-dual] also optimising for "
                                  << (is_complex ? "real" : "complex")
                                  << " value type\n";

                    BenchParams other_params            = params_;
                    other_params.use_real               = is_complex;
                    other_params.complementary_bo_pass  = true;

                    TileSweepBenchmark<ExecSpace, other_value_type> other(other_params);
                    other.setup_domain(nghost);
                    other.initialize(kernel, nghost);
                    size_t other_np = other.get_n_local_particles();

                    for (const std::string& method : {"Tiled", "OutputFocused"}) {
                        auto bo = other.run_bo(method, kernel, other_np);
                        bo_results.push_back(bo);
                        const auto& bt = bo.best_tile;
                        if (other.is_config_valid(method, bt, actual_width, bo.best_team_size,
                                                  bo.best_z_batches)) {
                            auto cfg = ippl::Interpolation::ScatterConfig<Dim>::
                                get_default<ExecSpace>();
                            cfg.method = (method == "Tiled")
                                             ? ippl::Interpolation::ScatterMethod::Tiled
                                             : ippl::Interpolation::ScatterMethod::OutputFocused;
                            cfg.tile_size               = {bt[0], bt[1], bt[2]};
                            cfg.team_size               = bo.best_team_size;
                            cfg.oversubscription_factor = bo.best_oversubscription_factor;
                            cfg.z_batches               = bo.best_z_batches;
                            auto r = other.benchmark_scatter(method, cfg, kernel, other_np, bt,
                                                             true);
                            r.team_size               = bo.best_team_size;
                            r.oversubscription_factor = bo.best_oversubscription_factor;
                            r.z_batches               = bo.best_z_batches;
                            results.push_back(r);
                        }
                    }
                    other.cleanup();
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
            write_bo_csv(bo_results);
            write_bo_history_csv(bo_results);
        }
        print_summary(results, bo_results);
    }

    // ------------------------------------------------------------------
    // Single-configuration scatter benchmark
    // ------------------------------------------------------------------
    BenchmarkResult benchmark_scatter(const std::string& method,
                                      const ippl::Interpolation::ScatterConfig<Dim>& cfg,
                                      const ippl::nufft::ESKernel<real_type>& kernel,
                                      size_t n_particles, std::array<int, 3> tile_arr,
                                      bool from_optimizer) {
        BenchmarkResult r;
        r.method                  = method;
        r.distribution            = params_.distribution;
        r.value_type              = value_type_str();
        r.tile_sizes              = tile_arr;
        r.tile_size               = tile_arr[0];
        r.team_size               = cfg.team_size;
        r.oversubscription_factor = cfg.oversubscription_factor;
        r.z_batches               = cfg.z_batches;
        r.kernel_width            = kernel.width();
        r.n_particles             = n_particles;
        r.n_grid                  = params_.n_grid;
        r.rho                     = params_.rho;
        r.from_optimizer          = from_optimizer;

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
                std::cout << "\n    [SKIP] " << method << " " << e.what() << "\n";
            const double nan = std::numeric_limits<double>::quiet_NaN();
            r.stats.mean_ms = r.stats.stddev_ms = r.stats.min_ms = r.stats.max_ms =
                r.stats.median_ms                                = nan;
            r.stats.count                                        = 0;
        }
        return r;
    }

    // ============================================================================
    // run_bo: Bayesian Optimisation over parameter space
    //
    // Fix 2: pre-flight infeasibility no longer charges bo.evaluations.
    //        A separate preflight_rejections counter is incremented instead.
    //
    // Fix 3: a SearchPoint hash-set (observed_gp) gates gp.add_observation()
    //        so duplicate LHS points do not corrupt the covariance matrix.
    //
    // Fix 4: Tiled uses GPModel<5> (no z_batches dimension);
    //        OutputFocused uses GPModel<6>.  Dispatched via run_bo_impl<N>().
    // ============================================================================

private:
    // -----------------------------------------------------------------------
    // Shared bookkeeping passed into the templated BO implementation.
    // -----------------------------------------------------------------------
    struct BOContext {
        const std::string& method;
        const ippl::nufft::ESKernel<double>& kernel;
        size_t n_particles;
        BOResult& bo;

        // Bounds
        int lo_tile, hi_tile;
        int lo_ts, hi_ts;
        int lo_osub, hi_osub;
        int lo_zb, hi_zb;

        const std::vector<int>& ts_candidates;
        const BenchParams& params;
    };

    // Helper: look up actual team size from index
    static int ts_of(const std::vector<int>& cands, int idx) {
        return cands[std::clamp(idx, 0, (int)cands.size() - 1)];
    }

    // -----------------------------------------------------------------------
    // Templated BO core — N = GP dimensionality (5 for Tiled, 6 for OutputFocused)
    // -----------------------------------------------------------------------
    template <int N>
    BOResult run_bo_impl(BOContext& ctx) {
        static_assert(N == 5 || N == 6, "BO dimensionality must be 5 or 6");

        BOResult& bo      = ctx.bo;
        const auto& p     = ctx.params;
        const int kernel_W = ctx.kernel.width();

        // Build bounds arrays for the GP.
        // For N==5 (Tiled), z_batches is omitted; the SearchPoint still carries
        // v[5] but we never vary it (lo_zb == hi_zb == 1).
        std::array<int, N> lo_bounds_gp, hi_bounds_gp;
        lo_bounds_gp[0] = ctx.lo_tile; hi_bounds_gp[0] = ctx.hi_tile;
        lo_bounds_gp[1] = ctx.lo_tile; hi_bounds_gp[1] = ctx.hi_tile;
        lo_bounds_gp[2] = ctx.lo_tile; hi_bounds_gp[2] = ctx.hi_tile;
        lo_bounds_gp[3] = ctx.lo_ts;   hi_bounds_gp[3] = ctx.hi_ts;
        lo_bounds_gp[4] = ctx.lo_osub; hi_bounds_gp[4] = ctx.hi_osub;
        if constexpr (N == 6) {
            lo_bounds_gp[5] = ctx.lo_zb;
            hi_bounds_gp[5] = ctx.hi_zb;
        }

        GPModel<N> gp;
        gp.set_bounds(lo_bounds_gp, hi_bounds_gp);

        // Convert a SearchPoint to an N-dim GP key
        auto gp_key = [&](const SearchPoint& pt) -> typename GPModel<N>::Point {
            typename GPModel<N>::Point k;
            k[0] = pt.v[0]; k[1] = pt.v[1]; k[2] = pt.v[2];
            k[3] = pt.v[3]; k[4] = pt.v[4];
            if constexpr (N == 6)
                k[5] = pt.v[5];
            return k;
        };

        // Fix 3: track which GP keys have been observed to prevent duplicates
        std::unordered_map<SearchPoint, bool,  SearchPointHash> feasible_cache;
        std::unordered_map<SearchPoint, double, SearchPointHash> tp_cache;
        // Use a set keyed on the GP point (N-dim) to guard add_observation
        using GPKey = typename GPModel<N>::Point;
        struct GPKeyHash {
            std::size_t operator()(const GPKey& k) const noexcept {
                std::size_t h = 0;
                for (int x : k)
                    h = h * 104729 + static_cast<std::size_t>(x + 100);
                return h;
            }
        };
        std::unordered_set<GPKey, GPKeyHash> observed_gp;  // Fix 3

        auto is_feasible = [&](const SearchPoint& pt) -> bool {
            auto it = feasible_cache.find(pt);
            if (it != feasible_cache.end())
                return it->second;
            bool ok = is_config_valid(ctx.method, pt.tile(), kernel_W,
                                      ts_of(ctx.ts_candidates, pt.ts_idx()), pt.z_batches());
            feasible_cache[pt] = ok;
            return ok;
        };

        auto make_cfg = [&](const SearchPoint& pt) {
            auto cfg      = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method    = (ctx.method == "Tiled")
                                ? ippl::Interpolation::ScatterMethod::Tiled
                                : ippl::Interpolation::ScatterMethod::OutputFocused;
            cfg.tile_size = {pt.tile_x(), pt.tile_y(), pt.tile_z()};
            cfg.team_size = ts_of(ctx.ts_candidates, pt.ts_idx());
            cfg.oversubscription_factor = pt.osub();
            cfg.z_batches               = pt.z_batches();
            return cfg;
        };

        // Fix 2 + Fix 3: evaluate() only increments bo.evaluations for actual
        // kernel launches.  Pre-flight rejections use bo.preflight_rejections.
        // gp_add() guards against duplicate observations.
        static constexpr double kInfeasiblePenalty = -1.0;

        auto gp_add = [&](const SearchPoint& pt, double tp) {
            // Fix 3: only add to GP if this GP key has not been seen before
            auto gk = gp_key(pt);
            if (observed_gp.count(gk))
                return;
            observed_gp.insert(gk);
            gp.add_observation(gk, tp);
        };

        auto evaluate = [&](const SearchPoint& pt) -> double {
            auto it = tp_cache.find(pt);
            if (it != tp_cache.end())
                return it->second;

            // Fix 2: pre-flight check is free — don't charge the kernel budget
            if (!is_feasible(pt)) {
                tp_cache[pt] = kInfeasiblePenalty;
                ++bo.preflight_rejections;  // Fix 2: separate counter
                if (p.verbose && ippl::Comm->rank() == 0)
                    std::cout << "    BO [pre-OOM] tile=(" << pt.tile_x() << "," << pt.tile_y()
                              << "," << pt.tile_z()
                              << ") team=" << ts_of(ctx.ts_candidates, pt.ts_idx()) << "\n";
                return kInfeasiblePenalty;
            }

            // Actual kernel launch — counts against budget
            auto cfg = make_cfg(pt);
            auto ta  = pt.tile();
            auto r   = benchmark_scatter(ctx.method, cfg, ctx.kernel, ctx.n_particles, ta, true);
            ++bo.evaluations;  // Fix 2: only incremented here

            const bool oom = (r.stats.count == 0) || std::isnan(r.stats.mean_ms);
            double tp      = oom ? kInfeasiblePenalty : r.throughput_Mpts_per_sec();
            tp_cache[pt]   = tp;

            if (oom) {
                feasible_cache[pt] = false;
                if (p.verbose && ippl::Comm->rank() == 0)
                    std::cout << "    BO [runtime-OOM] tile=(" << pt.tile_x() << "," << pt.tile_y()
                              << "," << pt.tile_z()
                              << ") team=" << ts_of(ctx.ts_candidates, pt.ts_idx())
                              << " — adding to infeasible cache\n";
            }

            if (p.verbose && ippl::Comm->rank() == 0)
                std::cout << "    BO eval " << std::setw(4) << bo.evaluations << "  tile=("
                          << pt.tile_x() << "," << pt.tile_y() << "," << pt.tile_z() << ")"
                          << " team=" << ts_of(ctx.ts_candidates, pt.ts_idx())
                          << " (threads=" << actual_threads(ctx.method,
                                                ts_of(ctx.ts_candidates, pt.ts_idx())) << ")"
                          << " osub=" << pt.osub() << " zb=" << pt.z_batches()
                          << "  tp=" << std::fixed << std::setprecision(1) << tp << " Mpts/s\n";
            return tp;
        };

        // RNG
        std::size_t mh = std::hash<std::string>{}(ctx.method);
        std::mt19937 rng(static_cast<uint32_t>(98765 + kernel_W * 1000 + (mh & 0xFFFF)));
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        auto rand_int = [&](int lo, int hi) -> int {
            return lo + (int)(unif(rng) * (hi - lo + 1));
        };

        const int lo_tile = ctx.lo_tile, hi_tile = ctx.hi_tile;
        const int lo_ts   = ctx.lo_ts,   hi_ts   = ctx.hi_ts;
        const int lo_osub = ctx.lo_osub, hi_osub = ctx.hi_osub;
        const int lo_zb   = ctx.lo_zb,   hi_zb   = ctx.hi_zb;

        // 6-dim bounds for SearchPoint clamping (always 6 — z_batches fixed for Tiled)
        std::array<int, 6> lo6 = {lo_tile, lo_tile, lo_tile, lo_ts, lo_osub, lo_zb};
        std::array<int, 6> hi6 = {hi_tile, hi_tile, hi_tile, hi_ts, hi_osub, hi_zb};

        const int total_budget = p.bo_budget;
        const int n_init       = std::max(6, total_budget / 5);
        const int n_bo_steps   = total_budget - n_init;

        SearchPoint best_pt;
        best_pt.v = {(lo_tile + hi_tile) / 2, (lo_tile + hi_tile) / 2,
                     (lo_tile + hi_tile) / 2, (lo_ts + hi_ts) / 2,
                     (lo_osub + hi_osub) / 2,  (lo_zb + hi_zb) / 2};
        double best_tp = 0.0;

        // Phase 1: LHS
        if (ippl::Comm->rank() == 0)
            std::cout << "    BO [init] " << N << "D LHS (" << n_init << " kernel evals)\n";
        {
            std::array<std::vector<int>, 6> strata;
            for (int d = 0; d < 6; ++d) {
                strata[d].resize(n_init);
                std::iota(strata[d].begin(), strata[d].end(), 0);
                std::shuffle(strata[d].begin(), strata[d].end(), rng);
            }
            auto cell = [&](int s, int n, int lo, int hi) -> int {
                double f = (s + unif(rng)) / n;
                return std::clamp(lo + (int)std::round(f * (hi - lo)), lo, hi);
            };
            int lhs_launched = 0;
            for (int i = 0; i < n_init && lhs_launched < n_init; ++i) {
                SearchPoint pt;
                pt.v[0] = cell(strata[0][i], n_init, lo_tile, hi_tile);
                pt.v[1] = cell(strata[1][i], n_init, lo_tile, hi_tile);
                pt.v[2] = cell(strata[2][i], n_init, lo_tile, hi_tile);
                pt.v[3] = cell(strata[3][i], n_init, lo_ts,   hi_ts);
                pt.v[4] = cell(strata[4][i], n_init, lo_osub, hi_osub);
                pt.v[5] = cell(strata[5][i], n_init, lo_zb,   hi_zb);
                double tp = evaluate(pt);
                // Fix 3: guard against duplicate GP observations
                gp_add(pt, tp);
                if (tp > kInfeasiblePenalty) {
                    // Only count actual kernel launches toward the LHS budget
                    ++lhs_launched;
                }
                bo.history.emplace_back(bo.evaluations + bo.preflight_rejections,
                                        pt.tile_x(), pt.tile_y(), pt.tile_z(),
                                        ts_of(ctx.ts_candidates, pt.ts_idx()),
                                        pt.osub(), pt.z_batches(), tp);
                if (tp > best_tp) {
                    best_tp = tp;
                    best_pt = pt;
                }
            }
        }

        if (best_tp <= 0.0 && ippl::Comm->rank() == 0)
            std::cout << "    BO [warn] No feasible config in LHS — exploring aggressively.\n";

        // Phase 2: Acquisition loop
        if (ippl::Comm->rank() == 0)
            std::cout << "    BO [opt] acquisition loop (" << n_bo_steps << " kernel evals)\n";
        {
            int bo_step = 0;
            while (bo.evaluations < total_budget) {
                gp.fit();

                double progress = (double)bo_step / std::max(n_bo_steps, 1);
                double ucb_prob = p.bo_ucb_prob * (1.0 - 0.9 * progress);
                double beta_ucb = 2.0 * std::sqrt(1.0 - progress) + 0.5;
                bool use_ucb    = (unif(rng) < ucb_prob);

                SearchPoint next = best_pt;
                double best_acq  = -1e300;

                auto score = [&](const SearchPoint& c) {
                    if (tp_cache.count(c))
                        return;
                    auto gk    = gp_key(c);
                    double acq = use_ucb ? gp.ucb(gk, beta_ucb)
                                         : gp.expected_improvement(gk, best_tp, p.bo_xi);
                    if (acq > best_acq) {
                        best_acq = acq;
                        next     = c;
                    }
                };

                for (int s = 0; s < 3000; ++s) {
                    SearchPoint c;
                    c.v[0] = rand_int(lo_tile, hi_tile);
                    c.v[1] = rand_int(lo_tile, hi_tile);
                    c.v[2] = rand_int(lo_tile, hi_tile);
                    c.v[3] = rand_int(lo_ts,   hi_ts);
                    c.v[4] = rand_int(lo_osub, hi_osub);
                    c.v[5] = rand_int(lo_zb,   hi_zb);
                    score(c);
                }
                for (int d = 0; d < 6; ++d) {
                    for (int delta : {-2, -1, +1, +2}) {
                        SearchPoint c = best_pt;
                        c.v[d] = std::clamp(best_pt.v[d] + delta, lo6[d], hi6[d]);
                        score(c);
                    }
                }

                if (best_acq <= -1e200) {
                    if (ippl::Comm->rank() == 0)
                        std::cout << "    BO [done] space exhausted after " << bo.evaluations
                                  << " kernel evals\n";
                    break;
                }

                double tp = evaluate(next);
                gp_add(next, tp);  // Fix 3: guarded add
                bo.history.emplace_back(bo.evaluations + bo.preflight_rejections,
                                        next.tile_x(), next.tile_y(), next.tile_z(),
                                        ts_of(ctx.ts_candidates, next.ts_idx()),
                                        next.osub(), next.z_batches(), tp);
                if (tp > best_tp) {
                    best_tp = tp;
                    best_pt = next;
                }
                ++bo_step;
            }
        }

        // Phase 3: Hill-climb polish
        if (ippl::Comm->rank() == 0)
            std::cout << "    BO [polish] hill-climb from tile=(" << best_pt.tile_x() << ","
                      << best_pt.tile_y() << "," << best_pt.tile_z()
                      << ") team=" << ts_of(ctx.ts_candidates, best_pt.ts_idx())
                      << " (threads=" << actual_threads(ctx.method,
                             ts_of(ctx.ts_candidates, best_pt.ts_idx())) << ")"
                      << " osub=" << best_pt.osub() << " zb=" << best_pt.z_batches() << "\n";
        {
            bool improved = true;
            int budget    = 30;
            while (improved && budget > 0) {
                improved = false;
                for (int d = 0; d < 6 && !improved; ++d) {
                    for (int delta : {-1, +1}) {
                        SearchPoint c = best_pt;
                        c.v[d] = std::clamp(best_pt.v[d] + delta, lo6[d], hi6[d]);
                        if (c == best_pt)
                            continue;
                        double tp;
                        if (tp_cache.count(c)) {
                            tp = tp_cache[c];
                        } else {
                            tp = evaluate(c);
                            --budget;
                            bo.history.emplace_back(
                                bo.evaluations + bo.preflight_rejections,
                                c.tile_x(), c.tile_y(), c.tile_z(),
                                ts_of(ctx.ts_candidates, c.ts_idx()),
                                c.osub(), c.z_batches(), tp);
                        }
                        if (tp > best_tp) {
                            best_tp  = tp;
                            best_pt  = c;
                            improved = true;
                            break;
                        }
                    }
                }
            }
        }

        bo.best_tile                    = {best_pt.tile_x(), best_pt.tile_y(), best_pt.tile_z()};
        bo.best_team_size               = ts_of(ctx.ts_candidates, best_pt.ts_idx());
        bo.best_oversubscription_factor = best_pt.osub();
        bo.best_z_batches               = best_pt.z_batches();

        if (best_tp <= 0.0 && ippl::Comm->rank() == 0) {
            std::cout << "    BO [warn] No config with positive throughput for "
                      << ctx.method << " w=" << kernel_W << " type=" << value_type_str()
                      << ".  All evaluated configs were infeasible or OOM.\n"
                      << "    Consider widening --team-sizes / --gp-team-sizes or "
                         "reducing --min-tile.\n";
        }

        {
            auto cfg = make_cfg(best_pt);
            if (is_feasible(best_pt) && best_tp > 0.0) {
                auto r = benchmark_scatter(ctx.method, cfg, ctx.kernel, ctx.n_particles,
                                           best_pt.tile(), true);
                bo.best_time_ms         = r.stats.mean_ms;
                bo.best_throughput_Mpts = r.throughput_Mpts_per_sec();
            } else {
                bo.best_time_ms         = std::numeric_limits<double>::quiet_NaN();
                bo.best_throughput_Mpts = 0.0;
            }
        }

        if (ippl::Comm->rank() == 0)
            std::cout << "  [BO] " << ctx.method << " w=" << kernel_W << "  best tile=("
                      << bo.best_tile[0] << "," << bo.best_tile[1] << "," << bo.best_tile[2]
                      << ") team=" << bo.best_team_size
                      << " (threads=" << actual_threads(ctx.method, bo.best_team_size) << ")"
                      << " osub=" << bo.best_oversubscription_factor
                      << " zb=" << bo.best_z_batches << "  throughput=" << std::fixed
                      << std::setprecision(1) << bo.best_throughput_Mpts << " Mpts/s"
                      << "  (" << bo.evaluations << " kernel evals, "
                      << bo.preflight_rejections << " pre-flight skips)\n";
        return bo;
    }

public:
    // Fix 4: dispatch to the right GP dimensionality
    BOResult run_bo(const std::string& method, const ippl::nufft::ESKernel<real_type>& kernel,
                    size_t n_particles) {
        BOResult bo;
        bo.method       = method;
        bo.value_type   = value_type_str();
        bo.kernel_width = kernel.width();
        bo.evaluations  = 0;
        bo.preflight_rejections = 0;

        const int kernel_W = kernel.width();
        const int lo_tile = params_.min_tile_size, hi_tile = params_.max_tile_size;

        const std::vector<int>& ts_candidates =
            (method == "Tiled") ? params_.team_size_candidates : params_.gp_team_size_candidates;
        const int n_ts  = (int)ts_candidates.size();
        const int lo_ts = 0, hi_ts = n_ts - 1;

        const int lo_osub = params_.min_osub, hi_osub = params_.max_osub;

        // Fix 4: z_batches only varies for OutputFocused
        const int lo_zb = (method == "OutputFocused") ? params_.min_z_batches : 1;
        const int hi_zb = (method == "OutputFocused") ? params_.max_z_batches : 1;

        BOContext ctx{method, kernel, n_particles, bo,
                      lo_tile, hi_tile, lo_ts, hi_ts, lo_osub, hi_osub, lo_zb, hi_zb,
                      ts_candidates, params_};

        // Fix 4: Tiled → 5D GP (no z_batches), OutputFocused → 6D GP
        if (method == "Tiled")
            return run_bo_impl<5>(ctx);
        else
            return run_bo_impl<6>(ctx);
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

    void initialize(const ippl::nufft::ESKernel<real_type>& /*kernel*/, int nghost) {
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
                  << "    Tile x Team-Size x Oversubscription Sweep Benchmark\n"
                  << "================================================================\n"
                  << "Grid size:        " << params_.n_grid << "^3\n"
                  << "Particles/grid:   " << params_.rho << "\n"
                  << "Total particles:  " << params_.n_particles() << "\n"
                  << "Distribution:     " << params_.distribution << "\n"
                  << "Value type:       " << value_type_str() << "\n"
                  << "Tile sizes:       " << params_.min_tile_size << " - " << params_.max_tile_size
                  << "\n"
                  << "Kernel widths:    " << params_.min_kernel_width << " - "
                  << params_.max_kernel_width << "\n"
                  << "Warmup runs:      " << params_.warmup_runs << "\n"
                  << "Benchmark runs:   " << params_.benchmark_runs << "\n";
        if (params_.optimize) {
            std::cout << "BO budget:        " << params_.bo_budget << " kernel evals\n"
                      << "BO xi:            " << params_.bo_xi << "\n"
                      << "BO UCB prob:      " << params_.bo_ucb_prob << "\n"
                      << "BO value types:   real + complex (both)\n"
                      << "Tiled GP dims:    5  (tile x3, ts_idx, osub)\n"
                      << "OutputFocused GD: 6  (tile x3, ts_idx, osub, z_batches)\n"
                      << "Tiled team sizes: ";
            for (int ts : params_.team_size_candidates)
                std::cout << ts << " ";
            std::cout << "(GPU thread counts)\n"
                      << "GP warp counts:   ";
            for (int ts : params_.gp_team_size_candidates)
                std::cout << ts << " ";
            std::cout << "(× " << BenchParams::warp_size << " = GPU threads)\n"
                      << "Osub range:       [" << params_.min_osub << ", " << params_.max_osub
                      << "]\n"
                      << "Z-batches range:  [" << params_.min_z_batches << ", "
                      << params_.max_z_batches << "]\n";
        }
        std::cout << "================================================================\n\n";
    }

    void write_full_csv(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0)
            return;
        std::string fn = params_.output_prefix + "_full.csv";
        std::ofstream out(fn);
        out << "method,distribution,value_type,"
            << "tile_x,tile_y,tile_z,team_size,oversubscription_factor,z_batches,"
            << "kernel_width,n_particles,n_grid,rho,"
            << "mean_ms,stddev_ms,min_ms,max_ms,median_ms,"
            << "throughput_Mpts_s,time_per_pt_ns,from_optimizer,status\n";
        for (const auto& r : results) {
            out << r.method << "," << r.distribution << "," << r.value_type << ","
                << r.tile_sizes[0] << "," << r.tile_sizes[1] << "," << r.tile_sizes[2] << ","
                << r.team_size << "," << r.oversubscription_factor << "," << r.z_batches << ","
                << r.kernel_width << ","
                << r.n_particles << "," << r.n_grid << "," << std::fixed << std::setprecision(1)
                << r.rho << ",";
            if (std::isnan(r.stats.mean_ms))
                out << "nan,nan,nan,nan,nan,nan,nan," << (r.from_optimizer ? "1" : "0")
                    << ",failed\n";
            else
                out << std::setprecision(4) << r.stats.mean_ms << "," << r.stats.stddev_ms << ","
                    << r.stats.min_ms << "," << r.stats.max_ms << "," << r.stats.median_ms << ","
                    << std::setprecision(2) << r.throughput_Mpts_per_sec() << ","
                    << r.time_per_point_ns() << "," << (r.from_optimizer ? "1" : "0") << ",ok\n";
        }
        out.close();
        std::cout << "Wrote full results to: " << fn << "\n";
    }

    void write_heatmap_csv(const std::vector<BenchmarkResult>& results, const std::string& method) {
        if (ippl::Comm->rank() != 0)
            return;
        std::string fn = params_.output_prefix + "_heatmap_" + method + ".csv";
        std::ofstream out(fn);
        std::vector<int> widths, tiles;
        for (const auto& r : results)
            if (r.method == method && !r.from_optimizer) {
                if (std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end())
                    widths.push_back(r.kernel_width);
                if (std::find(tiles.begin(), tiles.end(), r.tile_sizes[0]) == tiles.end())
                    tiles.push_back(r.tile_sizes[0]);
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
                for (const auto& r : results)
                    if (r.method == method && !r.from_optimizer && r.tile_sizes[0] == t
                        && r.kernel_width == w) {
                        out << (std::isnan(r.stats.mean_ms)
                                    ? ",nan"
                                    : "," + std::to_string(r.throughput_Mpts_per_sec()));
                        found = true;
                        break;
                    }
                if (!found)
                    out << ",nan";
            }
            out << "\n";
        }
        out.close();
        std::cout << "Wrote heatmap for " << method << " to: " << fn << "\n";
    }

    void write_optimal_csv(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0)
            return;
        std::string fn = params_.output_prefix + "_optimal.csv";
        std::ofstream out(fn);
        out << "method,kernel_width,optimal_tile_size,throughput_Mpts_s,time_ms\n";
        for (const std::string& method : {"Tiled", "OutputFocused"}) {
            std::vector<int> widths;
            for (const auto& r : results)
                if (std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end())
                    widths.push_back(r.kernel_width);
            std::sort(widths.begin(), widths.end());
            for (int w : widths) {
                const BenchmarkResult* best = nullptr;
                double bt                   = 0;
                for (const auto& r : results)
                    if (r.method == method && r.kernel_width == w && !r.from_optimizer
                        && !std::isnan(r.stats.mean_ms)) {
                        double tp = r.throughput_Mpts_per_sec();
                        if (tp > bt) {
                            bt   = tp;
                            best = &r;
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
        std::cout << "Wrote optimal configurations to: " << fn << "\n";
    }

    void write_bo_csv(const std::vector<BOResult>& rs) {
        if (ippl::Comm->rank() != 0)
            return;
        std::string fn = params_.output_prefix + "_sa_optimal.csv";
        std::ofstream out(fn);
        out << "method,value_type,kernel_width,"
            << "best_tile_x,best_tile_y,best_tile_z,"
            << "best_team_size,best_oversubscription_factor,best_z_batches,"
            << "throughput_Mpts_s,time_ms,kernel_evaluations,preflight_rejections\n";
        for (const auto& r : rs)
            out << r.method << "," << r.value_type << "," << r.kernel_width << ","
                << r.best_tile[0] << "," << r.best_tile[1] << "," << r.best_tile[2] << ","
                << r.best_team_size << "," << r.best_oversubscription_factor << ","
                << r.best_z_batches << "," << std::fixed << std::setprecision(2)
                << r.best_throughput_Mpts << "," << std::setprecision(4) << r.best_time_ms << ","
                << r.evaluations << "," << r.preflight_rejections << "\n";
        out.close();
        std::cout << "Wrote BO optimal results to: " << fn << "\n";
    }

    void write_bo_history_csv(const std::vector<BOResult>& rs) {
        if (ippl::Comm->rank() != 0)
            return;
        std::string fn = params_.output_prefix + "_sa_history.csv";
        std::ofstream out(fn);
        out << "method,value_type,kernel_width,step,tile_x,tile_y,tile_z,team_size,"
               "oversubscription_factor,z_batches,throughput_Mpts_s\n";
        for (const auto& r : rs)
            for (const auto& [step, tx, ty, tz, ts, osub, zb, tp] : r.history)
                out << r.method << "," << r.value_type << "," << r.kernel_width << "," << step
                    << "," << tx << "," << ty << "," << tz << "," << ts << "," << osub << "," << zb
                    << "," << std::fixed << std::setprecision(2) << tp << "\n";
        out.close();
        std::cout << "Wrote BO convergence history to: " << fn << "\n";
    }

    void print_summary(const std::vector<BenchmarkResult>& results,
                       const std::vector<BOResult>& bo_results) {
        if (ippl::Comm->rank() != 0)
            return;
        int failed = 0;
        for (const auto& r : results)
            if (std::isnan(r.stats.mean_ms))
                ++failed;
        std::cout << "\n"
                  << "================================================================\n"
                  << "                Results Summary (" << value_type_str() << ")\n"
                  << "================================================================\n";
        if (failed > 0)
            std::cout << "\nNote: " << failed << " configuration(s) failed.\n";

        std::vector<int> widths;
        for (const auto& r : results)
            if (!r.from_optimizer
                && std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end())
                widths.push_back(r.kernel_width);
        std::sort(widths.begin(), widths.end());

        for (const std::string& method : {"Tiled", "OutputFocused"}) {
            std::cout << "\n"
                      << method << " -- optimal uniform tile (default team/osub):\n"
                      << std::string(60, '-') << "\n"
                      << std::left << std::setw(8) << "Width" << std::right << std::setw(12)
                      << "Best Tile" << std::setw(14) << "Mpts/s" << std::setw(12) << "Time (ms)\n"
                      << std::string(60, '-') << "\n";
            for (int w : widths) {
                const BenchmarkResult* best = nullptr;
                double bt                   = 0;
                for (const auto& r : results)
                    if (r.method == method && r.kernel_width == w && !r.from_optimizer
                        && !std::isnan(r.stats.mean_ms)) {
                        double tp = r.throughput_Mpts_per_sec();
                        if (tp > bt) {
                            bt   = tp;
                            best = &r;
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

        if (!bo_results.empty()) {
            std::cout << "\n"
                      << "================================================================\n"
                      << "           BO-Optimised Full Configuration (real + complex)\n"
                      << "================================================================\n"
                      << std::left << std::setw(16) << "Method" << std::right << std::setw(8)
                      << "Width" << std::setw(10) << "VType" << std::setw(18) << "Tile (x,y,z)"
                      << std::setw(8) << "Team" << std::setw(8) << "Threads" << std::setw(6)
                      << "Osub" << std::setw(8) << "ZBatch" << std::setw(12) << "Mpts/s"
                      << std::setw(8) << "KernEvals\n"
                      << std::string(102, '-') << "\n";
            for (const auto& r : bo_results) {
                std::ostringstream ts;
                ts << "(" << r.best_tile[0] << "," << r.best_tile[1] << "," << r.best_tile[2]
                   << ")";
                int threads = actual_threads(r.method, r.best_team_size);
                std::cout << std::left << std::setw(16) << r.method << std::right << std::setw(8)
                          << r.kernel_width << std::setw(10) << r.value_type << std::setw(18)
                          << ts.str() << std::setw(8) << r.best_team_size << std::setw(8)
                          << threads << std::setw(6) << r.best_oversubscription_factor
                          << std::setw(8) << r.best_z_batches << std::fixed << std::setprecision(1)
                          << std::setw(12) << r.best_throughput_Mpts << std::setw(8)
                          << r.evaluations << "\n";
            }
        }
        std::cout << "\n";
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