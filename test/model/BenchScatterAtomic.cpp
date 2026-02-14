#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <numeric>
#include <string>
#include <vector>

using namespace ippl;

// --------------------- Manual timer (GPU fenced) ---------------------
class ManualTimer {
public:
    using clock_type = std::chrono::high_resolution_clock;
    void start() {
        Kokkos::fence();
        t0_ = clock_type::now();
    }
    double stop() {
        Kokkos::fence();
        auto t1 = clock_type::now();
        return std::chrono::duration<double>(t1 - t0_).count();
    }

private:
    clock_type::time_point t0_;
};

// --------------------- CLI ---------------------
struct BenchParams {
    int grid           = 256;
    double rho         = 10.0;
    double tol         = 1e-6;
    int warmup         = 5;
    int runs           = 20;
    std::string dist   = "uniform";  // uniform|clustered
    std::string output = "scatter_es_roofline.csv";
    bool verbose       = false;
    bool ncu_mode      = false;

    // microbench controls (new)
    int micro_repeats               = 5;
    int micro_iters                 = 8;
    int micro_points_per_thread     = 1;  // per thread per iter (keep 1 for clean modeling)
    std::vector<size_t> micro_bins  = {1ul << 10, 1ul << 12, 1ul << 14,
                                       1ul << 16, 1ul << 18, 1ul << 20};
    double micro_hot_frac_uniform   = 1.0;   // uniform: all bins used
    double micro_hot_frac_clustered = 0.01;  // clustered: only 1% bins used (contention dial)

    size_t n_particles_global() const {
        return static_cast<size_t>(rho * double(grid) * double(grid) * double(grid));
    }
};

static BenchParams parse_args(int argc, char** argv) {
    BenchParams p;
    for (int i = 1; i < argc; ++i) {
        auto get = [&](const char* k) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", k);
                std::exit(1);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--grid"))
            p.grid = std::atoi(get("--grid"));
        else if (!std::strcmp(argv[i], "--rho"))
            p.rho = std::atof(get("--rho"));
        else if (!std::strcmp(argv[i], "--tol"))
            p.tol = std::atof(get("--tol"));
        else if (!std::strcmp(argv[i], "--warmup"))
            p.warmup = std::atoi(get("--warmup"));
        else if (!std::strcmp(argv[i], "--runs"))
            p.runs = std::atoi(get("--runs"));
        else if (!std::strcmp(argv[i], "--dist"))
            p.dist = get("--dist");
        else if (!std::strcmp(argv[i], "--output"))
            p.output = get("--output");
        else if (!std::strcmp(argv[i], "--verbose") || !std::strcmp(argv[i], "-v"))
            p.verbose = true;
        else if (!std::strcmp(argv[i], "--ncu-mode")) {
            p.ncu_mode = true;
            p.warmup   = 1;
            p.runs     = 1;
        } else if (!std::strcmp(argv[i], "--help")) {
            std::cout << "Usage: BenchmarkScatterESRoofline [options]\n"
                         "  --grid N\n"
                         "  --rho R\n"
                         "  --tol T\n"
                         "  --warmup W\n"
                         "  --runs K\n"
                         "  --dist uniform|clustered\n"
                         "  --output FILE\n"
                         "  --ncu-mode\n"
                         "  -v,--verbose\n";
            std::exit(0);
        }
    }
    return p;
}

// --------------------- Stats ---------------------
struct Stats {
    double mean_ms = 0, std_ms = 0, min_ms = 0, max_ms = 0, median_ms = 0;
    size_t n = 0;
};
static Stats stats_ms(const std::vector<double>& sec) {
    Stats s;
    s.n = sec.size();
    if (!s.n)
        return s;
    std::vector<double> ms(s.n);
    for (size_t i = 0; i < s.n; ++i)
        ms[i] = sec[i] * 1000.0;
    double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
    s.mean_ms  = sum / double(s.n);
    double sq  = 0.0;
    for (double x : ms)
        sq += (x - s.mean_ms) * (x - s.mean_ms);
    s.std_ms      = (s.n > 1) ? std::sqrt(sq / double(s.n - 1)) : 0.0;
    auto [mn, mx] = std::minmax_element(ms.begin(), ms.end());
    s.min_ms      = *mn;
    s.max_ms      = *mx;
    std::sort(ms.begin(), ms.end());
    s.median_ms = (s.n % 2 == 0) ? 0.5 * (ms[s.n / 2 - 1] + ms[s.n / 2]) : ms[s.n / 2];
    return s;
}

// ============================================================================
// NEW: random-index atomic microbench (bins + hot fraction), 1x and 2x atomics
// ============================================================================

struct MicroPoint {
    size_t bins      = 0;
    double hot_frac  = 1.0;
    double ops_per_s = 0.0;  // effective "updates/s"
};

static inline size_t clamp_sz(size_t x, size_t lo, size_t hi) {
    return std::max(lo, std::min(hi, x));
}

// log-log interpolate throughput vs bins (monotone-ish). Falls back to nearest.
static double interp_loglog(const std::vector<MicroPoint>& pts, size_t bins_query,
                            double hot_frac_query) {
    // filter by hot_frac (exact match buckets for simplicity)
    std::vector<MicroPoint> f;
    for (auto& p : pts)
        if (std::abs(p.hot_frac - hot_frac_query) < 1e-15)
            f.push_back(p);
    if (f.empty())
        return 0.0;

    std::sort(f.begin(), f.end(), [](auto& a, auto& b) {
        return a.bins < b.bins;
    });

    if (bins_query <= f.front().bins)
        return f.front().ops_per_s;
    if (bins_query >= f.back().bins)
        return f.back().ops_per_s;

    for (size_t i = 0; i + 1 < f.size(); ++i) {
        auto b0 = f[i].bins, b1 = f[i + 1].bins;
        if (bins_query >= b0 && bins_query <= b1) {
            double x0 = std::log(double(b0)), x1 = std::log(double(b1)),
                   x  = std::log(double(bins_query));
            double y0 = std::log(std::max(f[i].ops_per_s, 1e-30));
            double y1 = std::log(std::max(f[i + 1].ops_per_s, 1e-30));
            double t  = (x - x0) / (x1 - x0);
            return std::exp(y0 + t * (y1 - y0));
        }
    }
    return f.back().ops_per_s;
}

template <class ExecSpace>
static double micro_atomic_random_ops_per_s(size_t nThreads, size_t bins, double hot_frac,
                                            int iters, int repeats, bool two_atomics) {
    // build random indices in [0, hot_bins)
    size_t hot_bins = clamp_sz(size_t(std::llround(double(bins) * hot_frac)), 1, bins);

    Kokkos::View<uint32_t*, ExecSpace> idx("idx", nThreads);
    Kokkos::View<float*, ExecSpace> grid("grid", bins);
    Kokkos::deep_copy(grid, 0.0f);

    using Pool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
    Pool pool(12345 + ippl::Comm->rank() * 17);

    Kokkos::parallel_for(
        "init_idx", Kokkos::RangePolicy<ExecSpace>(0, nThreads), KOKKOS_LAMBDA(const int i) {
            auto gen = pool.get_state();
            idx(i)   = (uint32_t)(gen.urand() % (uint32_t)hot_bins);
            pool.free_state(gen);
        });
    Kokkos::fence();

    auto run_once = [&]() {
        Kokkos::parallel_for(
            "atomic_random", Kokkos::RangePolicy<ExecSpace>(0, nThreads),
            KOKKOS_LAMBDA(const int i) {
                uint32_t j = idx(i);
                for (int k = 0; k < iters; ++k) {
                    Kokkos::atomic_add(&grid(j), 1.0f);
                    if (two_atomics)
                        Kokkos::atomic_add(&grid(j), 1.0f);  // 2nd atomic to mimic complex split
                }
            });
        Kokkos::fence();
    };

    // warmup
    run_once();

    ManualTimer t;
    double best = 1e100;
    for (int r = 0; r < repeats; ++r) {
        t.start();
        run_once();
        double dt = t.stop();
        best      = std::min(best, dt);
    }

    const double ops = double(nThreads) * double(iters) * (two_atomics ? 2.0 : 1.0);
    return ops / best;
}

// ============================================================================
// NEW: parameter fit (alpha,beta) for B_eff = alpha*N^3/w^3, R_pred = beta*R_micro(B_eff)
// ============================================================================

struct FitParams {
    double alpha = 1.0;  // mapping scale
    double beta  = 1.0;  // overhead factor
};

static FitParams fit_alpha_beta_single(double N3_over_w3, double measured_updates_per_s,
                                       const std::vector<MicroPoint>& micro_pts, double hot_frac) {
    // We grid-search alpha in log space (cheap, robust). Then solve beta in closed form.
    // alpha controls B_eff; beta = measured / micro(B_eff)
    const double a_min = 1e-6;
    const double a_max =
        1.0;  // alpha>1 usually doesn’t make sense (B_eff would exceed N^3 before clamp)
    const int steps = 80;

    FitParams best{};
    double best_relerr = 1e100;

    for (int i = 0; i < steps; ++i) {
        double ta    = double(i) / double(steps - 1);
        double alpha = std::exp(std::log(a_min) + ta * (std::log(a_max) - std::log(a_min)));

        size_t B_eff = (size_t)std::llround(alpha * N3_over_w3);
        if (B_eff < 1)
            B_eff = 1;

        double micro = interp_loglog(micro_pts, B_eff, hot_frac);
        if (micro <= 0.0)
            continue;

        double beta = measured_updates_per_s / micro;
        // constrain beta to a sane range
        beta = std::max(0.0, std::min(1.0, beta));

        double pred = beta * micro;
        double relerr =
            std::abs(pred - measured_updates_per_s) / std::max(measured_updates_per_s, 1e-30);

        if (relerr < best_relerr) {
            best_relerr = relerr;
            best.alpha  = alpha;
            best.beta   = beta;
        }
    }
    return best;
}

// --------------------- Microbench: atomic no contention (unchanged) ---------------------
template <class ExecSpace>
double atomic_nocont_ops_per_sec(size_t nThreads, int iters, int repeats) {
    Kokkos::View<float*, ExecSpace> v("v", nThreads);
    Kokkos::deep_copy(v, 0.0f);

    auto run_once = [&]() {
        Kokkos::parallel_for(
            "atomic_nocont", Kokkos::RangePolicy<ExecSpace>(0, nThreads),
            KOKKOS_LAMBDA(const int i) {
                for (int k = 0; k < iters; ++k)
                    Kokkos::atomic_add(&v(i), 1.0f);
            });
        Kokkos::fence();
    };
    run_once();
    ManualTimer t;
    double best = 1e100;
    for (int r = 0; r < repeats; ++r) {
        t.start();
        run_once();
        best = std::min(best, t.stop());
    }
    return (double(nThreads) * double(iters)) / best;
}

// --------------------- Benchmark core ---------------------
template <typename ExecSpace>
class Bench {
public:
    static constexpr unsigned Dim = 3;
    using real_type               = double;
    using complex_type            = Kokkos::complex<real_type>;

    using Mesh_t      = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t = typename Mesh_t::DefaultCentering;
    using Field_t     = ippl::Field<complex_type, Dim, Mesh_t, Centering_t>;
    using Layout_t    = ippl::FieldLayout<Dim>;
    using PLayout_t   = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t     = ippl::ParticleBase<PLayout_t>;

    explicit Bench(const BenchParams& p)
        : p_(p)
        , kernel_(p.tol) {}

    struct Result {
        std::vector<double> total_times_s;
        Stats total_stats;
    };

    int run() {
        if (ippl::Comm->rank() == 0)
            print_header();

        const int w      = kernel_.width();
        const int nghost = w / 2 + 1;

        setup_domain(nghost);
        init_particles();
        init_grid(nghost);

        // variants
        struct Variant {
            const char* name;
            ippl::Interpolation::ScatterConfig<Dim> cfg;
        };
        std::vector<Variant> variants;

        {
            auto cfg   = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
            variants.push_back({"Atomic", cfg});
        }

        {
            auto cfg   = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
            cfg.tile_size.fill(4);
            variants.push_back({"Tiled", cfg});
        }

        {
            auto cfg      = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method    = ippl::Interpolation::ScatterMethod::OutputFocused;
            int tile_size = 2;
            std::array<int, 10> tile_sizes = {1, 1, 4, 3, 3, 2, 4, 2, 2, 2};
            if (w < 10)
                tile_size = tile_sizes[w];
            cfg.tile_size.fill(tile_size);
            variants.push_back({"GridParallel", cfg});
        }

        // ideal nocont baseline
        double atomic_nocont_rate =
            atomic_nocont_ops_per_sec<ExecSpace>(n_local_, p_.micro_iters, p_.micro_repeats);
        double atomic_nocont_rate_global = 0.0;
        MPI_Allreduce(&atomic_nocont_rate, &atomic_nocont_rate_global, 1, MPI_DOUBLE, MPI_SUM,
                      ippl::Comm->getCommunicator());

        // NEW: run microbench curve(s) once per run
        std::vector<MicroPoint> micro_uniform_1x, micro_clustered_1x;
        std::vector<MicroPoint> micro_uniform_2x, micro_clustered_2x;

        run_microbench_curves(micro_uniform_1x, micro_clustered_1x, /*two_atomics=*/false);
        run_microbench_curves(micro_uniform_2x, micro_clustered_2x, /*two_atomics=*/true);

        if (ippl::Comm->rank() == 0)
            write_csv_header();

        // Precompute N^3/w^3 for mapping
        const double N3         = double(p_.grid) * double(p_.grid) * double(p_.grid);
        const double w3         = double(w) * double(w) * double(w);
        const double N3_over_w3 = N3 / w3;

        for (auto& v : variants) {
            auto res = bench_scatter_variant(v.name, v.cfg);

            double updates_local  = double(n_local_) * w3;
            double updates_global = 0.0;
            MPI_Allreduce(&updates_local, &updates_global, 1, MPI_DOUBLE, MPI_SUM,
                          ippl::Comm->getCommunicator());

            double t_mean_s        = res.total_stats.mean_ms * 1e-3;
            double t_mean_s_global = 0.0;
            MPI_Allreduce(&t_mean_s, &t_mean_s_global, 1, MPI_DOUBLE, MPI_MAX,
                          ippl::Comm->getCommunicator());

            double scatter_updates_per_s = updates_global / t_mean_s_global;
            double util                  = scatter_updates_per_s / atomic_nocont_rate_global;

            // NEW: fit + predict using microbench curve for this dist
            const bool is_clustered = (p_.dist != "uniform");
            const double hot_frac =
                is_clustered ? p_.micro_hot_frac_clustered : p_.micro_hot_frac_uniform;

            // Use 2x atomics curve by default since scatter updates complex (more realistic)
            const auto& curve_2x = is_clustered ? micro_clustered_2x : micro_uniform_2x;

            FitParams fp =
                fit_alpha_beta_single(N3_over_w3, scatter_updates_per_s, curve_2x, hot_frac);

            size_t B_eff = (size_t)std::llround(fp.alpha * N3_over_w3);
            if (B_eff < 1)
                B_eff = 1;

            double micro_at_B         = interp_loglog(curve_2x, B_eff, hot_frac);
            double pred_updates_per_s = fp.beta * micro_at_B;
            double rel_err            = std::abs(pred_updates_per_s - scatter_updates_per_s)
                             / std::max(scatter_updates_per_s, 1e-30);

            if (ippl::Comm->rank() == 0) {
                append_csv_row(v.name, res, scatter_updates_per_s, atomic_nocont_rate_global, util,
                               /*model*/ fp.alpha, fp.beta, (double)B_eff, micro_at_B,
                               pred_updates_per_s, rel_err);
                if (p_.verbose) {
                    std::cout << "Variant " << v.name << " meas=" << scatter_updates_per_s
                              << " pred=" << pred_updates_per_s << " relerr=" << rel_err
                              << " alpha=" << fp.alpha << " beta=" << fp.beta << " B_eff=" << B_eff
                              << "\n";
                }
            }
        }

        cleanup();
        return 0;
    }

    void print_header() {
        std::cout << "\n============================================================\n";
        std::cout << " Scatter benchmark (ESKernel) + microbench+fit model\n";
        std::cout << "============================================================\n";
        std::cout << "grid N           : " << p_.grid << "\n";
        std::cout << "rho              : " << p_.rho << "\n";
        std::cout << "particles global : " << p_.n_particles_global() << "\n";
        std::cout << "dist             : " << p_.dist << "\n";
        std::cout << "tol              : " << p_.tol << "\n";
        std::cout << "ESKernel width w : " << kernel_.width() << "\n";
        std::cout << "warmup/runs      : " << p_.warmup << "/" << p_.runs << "\n";
        if (p_.ncu_mode)
            std::cout << "NCU mode         : ON (single run)\n";
        std::cout << "micro bins       : ";
        for (auto b : p_.micro_bins)
            std::cout << b << " ";
        std::cout << "\n";
        std::cout << "micro iters      : " << p_.micro_iters << " repeats=" << p_.micro_repeats
                  << "\n";
        std::cout << "============================================================\n\n";
    }

    void setup_domain(int nghost) {
        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d)
            domain[d] = ippl::Index(p_.grid);

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        layout_ = std::make_unique<Layout_t>(MPI_COMM_WORLD, domain, isParallel, true, nghost);

        for (unsigned d = 0; d < Dim; ++d) {
            origin_[d] = 0.0;
            hx_[d]     = 2.0 * M_PI / double(p_.grid);
        }
        mesh_    = std::make_unique<Mesh_t>(domain, hx_, origin_);
        playout_ = std::make_unique<PLayout_t>(*layout_, *mesh_);

        bunch_ = std::make_unique<Bunch_t>(*playout_);
        bunch_->addAttribute(R_);
        bunch_->addAttribute(Q_);
        bunch_->setParticleBC(ippl::BC::PERIODIC);

        const size_t n_global = p_.n_particles_global();
        const int nr          = ippl::Comm->size();
        const int r           = ippl::Comm->rank();
        const size_t base     = n_global / size_t(nr);
        const size_t rem      = n_global % size_t(nr);
        n_local_              = base + (size_t(r) < rem ? 1 : 0);

        bunch_->create(n_local_);
    }

    void init_particles() {
        auto Rv = R_.getView();
        auto Qv = Q_.getView();
        Kokkos::Random_XorShift64_Pool<> pool(42 + ippl::Comm->rank() * 101);

        if (p_.dist == "uniform") {
            Kokkos::parallel_for(
                "init_R_uniform", Kokkos::RangePolicy<ExecSpace>(0, n_local_),
                KOKKOS_LAMBDA(const int i) {
                    auto gen = pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d)
                        Rv(i)[d] = gen.drand() * 2.0 * M_PI;
                    Qv(i) = complex_type(1.0, 0.0);
                    pool.free_state(gen);
                });
        } else {
            Kokkos::parallel_for(
                "init_R_clustered", Kokkos::RangePolicy<ExecSpace>(0, n_local_),
                KOKKOS_LAMBDA(const int i) {
                    auto gen = pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        double u1 = gen.drand(), u2 = gen.drand();
                        double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-12))
                                   * Kokkos::cos(2.0 * M_PI * u2);
                        double x = M_PI + 0.3 * z;
                        while (x < 0)
                            x += 2.0 * M_PI;
                        while (x >= 2.0 * M_PI)
                            x -= 2.0 * M_PI;
                        Rv(i)[d] = x;
                    }
                    Qv(i) = complex_type(1.0, 0.0);
                    pool.free_state(gen);
                });
        }
        Kokkos::fence();
    }

    void init_grid(int nghost) {
        grid_  = std::make_unique<Field_t>(*mesh_, *layout_, nghost);
        *grid_ = complex_type(0.0, 0.0);
        Kokkos::fence();
    }

    Result bench_scatter_variant(const std::string& name,
                                 const ippl::Interpolation::ScatterConfig<Dim>& cfg) {
        if (ippl::Comm->rank() == 0 && p_.verbose)
            std::cout << "Benchmark scatter: " << name << "\n";
        ManualTimer t;

        for (int i = 0; i < p_.warmup; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            Q_.scatter_kernel(*grid_, R_, kernel_, cfg);
            grid_->accumulateHalo();
        }
        Kokkos::fence();

        std::vector<double> times;
        times.reserve(p_.runs);
        for (int i = 0; i < p_.runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            Kokkos::fence();
            t.start();
            Q_.scatter_kernel(*grid_, R_, kernel_, cfg);
            grid_->accumulateHalo();
            double dt = t.stop();
            times.push_back(dt);
        }

        Result r;
        r.total_times_s = std::move(times);
        r.total_stats   = stats_ms(r.total_times_s);
        return r;
    }

    // NEW: microbench curves for uniform + clustered (hot fraction)
    void run_microbench_curves(std::vector<MicroPoint>& out_uniform,
                               std::vector<MicroPoint>& out_clustered, bool two_atomics) {
        // Choose a thread count representative of scatter (n_local_ threads)
        const size_t nThreads = n_local_;
        // note: bins can be huge; keep micro bins manageable but wide enough to see curve
        for (size_t B : p_.micro_bins) {
            // uniform: hot_frac=1.0
            double r_u = micro_atomic_random_ops_per_s<ExecSpace>(
                nThreads, B, p_.micro_hot_frac_uniform, p_.micro_iters, p_.micro_repeats,
                two_atomics);

            // clustered: hot_frac small -> contention
            double r_c = micro_atomic_random_ops_per_s<ExecSpace>(
                nThreads, B, p_.micro_hot_frac_clustered, p_.micro_iters, p_.micro_repeats,
                two_atomics);

            // Reduce across MPI as sum of per-rank rates
            double ru_g = 0.0, rc_g = 0.0;
            MPI_Allreduce(&r_u, &ru_g, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
            MPI_Allreduce(&r_c, &rc_g, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());

            if (ippl::Comm->rank() == 0 && p_.verbose) {
                std::cout << "[micro " << (two_atomics ? "2x" : "1x") << "] bins=" << B
                          << " uniform=" << ru_g << " ops/s"
                          << " clustered=" << rc_g << " ops/s\n";
            }

            out_uniform.push_back(MicroPoint{B, p_.micro_hot_frac_uniform, ru_g});
            out_clustered.push_back(MicroPoint{B, p_.micro_hot_frac_clustered, rc_g});
        }
    }

    void write_csv_header() {
        std::ofstream out(p_.output);
        out << "kernel_variant,dist,grid,rho,particles_global,particles_local,"
               "tol,width,warmup,runs,"
               "mean_ms,std_ms,median_ms,min_ms,max_ms,"
               "scatter_updates_per_s,atomic_nocont_ops_per_s,util,"
               "model_alpha,model_beta,model_Beff,model_micro_at_Beff,model_pred_updates_per_s,"
               "model_rel_err\n";
    }

    void append_csv_row(const std::string& variant, const Result& r, double scatter_updates_per_s,
                        double atomic_nocont_ops_per_s_global, double util, double model_alpha,
                        double model_beta, double model_Beff, double model_micro_at_Beff,
                        double model_pred_updates_per_s, double model_rel_err) {
        std::ofstream out(p_.output, std::ios::app);

        out << variant << "," << p_.dist << "," << p_.grid << "," << std::fixed
            << std::setprecision(3) << p_.rho << "," << p_.n_particles_global() << "," << n_local_
            << "," << std::scientific << std::setprecision(2) << p_.tol << "," << kernel_.width()
            << "," << p_.warmup << "," << p_.runs << "," << std::fixed << std::setprecision(4)
            << r.total_stats.mean_ms << "," << r.total_stats.std_ms << ","
            << r.total_stats.median_ms << "," << r.total_stats.min_ms << "," << r.total_stats.max_ms
            << "," << std::scientific << std::setprecision(6) << scatter_updates_per_s << ","
            << atomic_nocont_ops_per_s_global << "," << std::fixed << std::setprecision(6) << util
            << "," << std::scientific << std::setprecision(6) << model_alpha << "," << model_beta
            << "," << model_Beff << "," << model_micro_at_Beff << "," << model_pred_updates_per_s
            << "," << std::fixed << std::setprecision(6) << model_rel_err << "\n";
    }

    void cleanup() {
        bunch_.reset();
        playout_.reset();
        grid_.reset();
        mesh_.reset();
        layout_.reset();
    }

    BenchParams p_;
    ippl::NUFFT::ESKernel<real_type> kernel_;

    ippl::Vector<real_type, Dim> origin_{};
    ippl::Vector<real_type, Dim> hx_{};

    std::unique_ptr<Layout_t> layout_;
    std::unique_ptr<Mesh_t> mesh_;
    std::unique_ptr<Field_t> grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t> bunch_;
    size_t n_local_ = 0;

    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R_;
    ippl::ParticleAttrib<complex_type> Q_;
};

// --------------------- main ---------------------
int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    int rc = 1;
    {
        BenchParams p = parse_args(argc, argv);
        Bench<Kokkos::DefaultExecutionSpace> b(p);
        rc = b.run();
    }
    ippl::finalize();
    return rc;
}
