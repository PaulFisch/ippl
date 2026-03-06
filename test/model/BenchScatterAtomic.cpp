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
    std::string dtype  = "complex";  // complex|real
    bool verbose       = false;
    bool ncu_mode      = false;

    int micro_repeats               = 5;
    int micro_iters                 = 8;
    std::vector<size_t> micro_bins  = {1ul << 10, 1ul << 12, 1ul << 14,
                                       1ul << 16, 1ul << 18, 1ul << 20};
    double micro_hot_frac_uniform   = 1.0;
    double micro_hot_frac_clustered = 0.01;

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
        else if (!std::strcmp(argv[i], "--dtype"))
            p.dtype = get("--dtype");
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
                         "  --dtype real|complex   (default complex)\n"
                         "  --output FILE\n"
                         "  --ncu-mode\n"
                         "  -v,--verbose\n";
            std::exit(0);
        }
    }
    if (p.dtype != "real" && p.dtype != "complex") {
        std::fprintf(stderr, "Invalid --dtype %s (use real|complex)\n", p.dtype.c_str());
        std::exit(1);
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

    double sq = 0.0;
    for (double x : ms)
        sq += (x - s.mean_ms) * (x - s.mean_ms);
    s.std_ms = (s.n > 1) ? std::sqrt(sq / double(s.n - 1)) : 0.0;

    auto [mn, mx] = std::minmax_element(ms.begin(), ms.end());
    s.min_ms      = *mn;
    s.max_ms      = *mx;

    std::sort(ms.begin(), ms.end());
    s.median_ms = (s.n % 2 == 0) ? 0.5 * (ms[s.n / 2 - 1] + ms[s.n / 2]) : ms[s.n / 2];
    return s;
}

// ============================================================================
// Microbench helpers
// ============================================================================

static inline size_t clamp_sz(size_t x, size_t lo, size_t hi) {
    return std::max(lo, std::min(hi, x));
}

template <class ExecSpace>
static double micro_atomic_random_ops_per_s(size_t nThreads, size_t bins, double hot_frac,
                                            int iters, int repeats, int atomics_per_iter) {
    size_t hot_bins = clamp_sz(size_t(std::llround(double(bins) * hot_frac)), 1, bins);

    Kokkos::View<uint32_t*, ExecSpace> idx("idx", nThreads);
    Kokkos::View<double*, ExecSpace> grid(
        "grid", bins);  // use double so "real" matches; for complex we scale atomics_per_iter=2
    Kokkos::deep_copy(grid, 0.0);

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
                    // emulate 1x (real) or 2x (complex split)
                    for (int a = 0; a < atomics_per_iter; ++a) {
                        Kokkos::atomic_add(&grid(j), 1.0);
                    }
                }
            });
        Kokkos::fence();
    };

    run_once();  // warmup

    ManualTimer t;
    double best = 1e100;
    for (int r = 0; r < repeats; ++r) {
        t.start();
        run_once();
        best = std::min(best, t.stop());
    }

    const double ops = double(nThreads) * double(iters) * double(atomics_per_iter);
    return ops / best;
}

template <class ExecSpace>
static double atomic_nocont_ops_per_sec(size_t nThreads, int iters, int repeats,
                                        int atomics_per_iter) {
    Kokkos::View<double*, ExecSpace> v("v", nThreads);
    Kokkos::deep_copy(v, 0.0);

    auto run_once = [&]() {
        Kokkos::parallel_for(
            "atomic_nocont", Kokkos::RangePolicy<ExecSpace>(0, nThreads),
            KOKKOS_LAMBDA(const int i) {
                for (int k = 0; k < iters; ++k) {
                    for (int a = 0; a < atomics_per_iter; ++a) {
                        Kokkos::atomic_add(&v(i), 1.0);
                    }
                }
            });
        Kokkos::fence();
    };

    run_once();  // warmup
    ManualTimer t;
    double best = 1e100;
    for (int r = 0; r < repeats; ++r) {
        t.start();
        run_once();
        best = std::min(best, t.stop());
    }
    return (double(nThreads) * double(iters) * double(atomics_per_iter)) / best;
}

// ============================================================================
// Benchmark core (templated on ValueType)
// ============================================================================

template <typename ExecSpace, typename ValueType>
class Bench {
public:
    static constexpr unsigned Dim = 3;
    using real_type               = double;

    using Mesh_t      = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t = typename Mesh_t::DefaultCentering;
    using Field_t     = ippl::Field<ValueType, Dim, Mesh_t, Centering_t>;
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

        // Only keep configs you currently use
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

        // how many atomics per "grid update" for this dtype?
        // real: 1 atomic; complex: typically 2 atomics (real+imag)
        constexpr bool is_complex  = std::is_same_v<ValueType, Kokkos::complex<real_type>>;
        const int atomics_per_iter = is_complex ? 2 : 1;

        // baseline nocont
        double atomic_nocont_rate = atomic_nocont_ops_per_sec<ExecSpace>(
            n_local_, p_.micro_iters, p_.micro_repeats, atomics_per_iter);
        double atomic_nocont_rate_global = 0.0;
        MPI_Allreduce(&atomic_nocont_rate, &atomic_nocont_rate_global, 1, MPI_DOUBLE, MPI_SUM,
                      ippl::Comm->getCommunicator());

        // optional microbench curves (kept; not writing them now)
        if (ippl::Comm->rank() == 0)
            write_csv_header();

        for (auto& v : variants) {
            auto res = bench_scatter_variant(v.name, v.cfg);

            const double w3 = double(w) * double(w) * double(w);

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

            if (ippl::Comm->rank() == 0) {
                append_csv_row(v.name, res, scatter_updates_per_s, atomic_nocont_rate_global, util,
                               atomics_per_iter);
                if (p_.verbose) {
                    std::cout << "Variant " << v.name << " mean_ms=" << res.total_stats.mean_ms
                              << " updates/s=" << scatter_updates_per_s
                              << " nocont_ops/s=" << atomic_nocont_rate_global << " util=" << util
                              << "\n";
                }
            }
        }

        cleanup();
        return 0;
    }

    KOKKOS_INLINE_FUNCTION static ValueType one_value() {
        if constexpr (std::is_same_v<ValueType, double>) {
            return 1.0;
        } else {
            return ValueType(1.0, 0.0);
        }
    }
    KOKKOS_INLINE_FUNCTION static ValueType zero_value() {
        if constexpr (std::is_same_v<ValueType, double>) {
            return 0.0;
        } else {
            return ValueType(0.0, 0.0);
        }
    }

    void print_header() {
        std::cout << "\n============================================================\n";
        std::cout << " Scatter benchmark (ESKernel): " << p_.dtype << " -> " << p_.dtype << "\n";
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
                    Qv(i) = one_value();
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
                    Qv(i) = one_value();
                    pool.free_state(gen);
                });
        }
        Kokkos::fence();
    }

    void init_grid(int nghost) {
        grid_  = std::make_unique<Field_t>(*mesh_, *layout_, nghost);
        *grid_ = zero_value();
        Kokkos::fence();
    }

    Result bench_scatter_variant(const std::string& name,
                                 const ippl::Interpolation::ScatterConfig<Dim>& cfg) {
        if (ippl::Comm->rank() == 0 && p_.verbose)
            std::cout << "Benchmark scatter: " << name << "\n";
        ManualTimer t;

        for (int i = 0; i < p_.warmup; ++i) {
            *grid_ = zero_value();
            Q_.scatter_kernel(*grid_, R_, kernel_, cfg);
            grid_->accumulateHalo();
        }
        Kokkos::fence();

        std::vector<double> times;
        times.reserve(p_.runs);
        for (int i = 0; i < p_.runs; ++i) {
            *grid_ = zero_value();
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

    void write_csv_header() {
        std::ofstream out(p_.output);
        out << "kernel_variant,dist,dtype,grid,rho,particles_global,particles_local,tol,width,"
               "warmup,runs,"
               "mean_ms,std_ms,median_ms,min_ms,max_ms,"
               "scatter_updates_per_s,atomic_nocont_ops_per_s,util,atomics_per_update\n";
    }

    void append_csv_row(const std::string& variant, const Result& r, double scatter_updates_per_s,
                        double atomic_nocont_ops_per_s_global, double util,
                        int atomics_per_update) {
        std::ofstream out(p_.output, std::ios::app);

        out << variant << "," << p_.dist << "," << p_.dtype << "," << p_.grid << "," << std::fixed
            << std::setprecision(3) << p_.rho << "," << p_.n_particles_global() << "," << n_local_
            << "," << std::scientific << std::setprecision(2) << p_.tol << "," << kernel_.width()
            << "," << p_.warmup << "," << p_.runs << "," << std::fixed << std::setprecision(4)
            << r.total_stats.mean_ms << "," << r.total_stats.std_ms << ","
            << r.total_stats.median_ms << "," << r.total_stats.min_ms << "," << r.total_stats.max_ms
            << "," << std::scientific << std::setprecision(6) << scatter_updates_per_s << ","
            << atomic_nocont_ops_per_s_global << "," << std::fixed << std::setprecision(6) << util
            << "," << atomics_per_update << "\n";
    }

    void cleanup() {
        bunch_.reset();
        playout_.reset();
        grid_.reset();
        mesh_.reset();
        layout_.reset();
    }

private:
    BenchParams p_;
    ippl::nufft::ESKernel<real_type> kernel_;

    ippl::Vector<real_type, Dim> origin_{};
    ippl::Vector<real_type, Dim> hx_{};

    std::unique_ptr<Layout_t> layout_;
    std::unique_ptr<Mesh_t> mesh_;
    std::unique_ptr<Field_t> grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t> bunch_;
    size_t n_local_ = 0;

    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R_;
    ippl::ParticleAttrib<ValueType> Q_;
};

// --------------------- main ---------------------
int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    int rc = 1;
    {
        BenchParams p = parse_args(argc, argv);

        if (p.dtype == "real") {
            Bench<Kokkos::DefaultExecutionSpace, double> b(p);
            rc = b.run();
        } else {
            Bench<Kokkos::DefaultExecutionSpace, Kokkos::complex<double>> b(p);
            rc = b.run();
        }
    }
    ippl::finalize();
    return rc;
}
