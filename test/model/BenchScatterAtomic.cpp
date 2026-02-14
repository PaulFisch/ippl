/**
 * BenchmarkScatterESRoofline.cpp
 *
 * Scatter throughput benchmark using ippl::NUFFT::ESKernel and density-based particle count.
 *
 * Measures:
 *  - Scatter throughput (updates/s) for ScatterMethod variants (Atomic/Tiled/OutputFocused)
 *  - Kokkos atomicAdd baseline with no contention (ops/s) as an "ideal atomics roofline"
 *
 * Usage:
 *   mpirun -n 1 ./BenchmarkScatterESRoofline --grid 256 --rho 10 --tol 1e-6 --runs 20 --warmup 5
 * --output out.csv
 *
 * Notes:
 *  - Total particles = rho * grid^3 (global), distributed across ranks.
 *  - Updates = n_particles_local * (kernel.width()^3)   (per-rank)
 *    Global updates/s computed by MPI reduction.
 */

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
    int grid           = 256;   // N
    double rho         = 10.0;  // particles per grid point (global density)
    double tol         = 1e-6;  // ESKernel tolerance -> width
    int warmup         = 5;
    int runs           = 20;
    std::string dist   = "uniform";  // uniform|clustered
    std::string output = "scatter_es_roofline.csv";
    bool verbose       = false;
    bool ncu_mode      = false;

    size_t n_particles_global() const {
        // rho * N^3
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
                         "  --grid N        grid size per dimension (default 256)\n"
                         "  --rho R         particles per grid point (default 10)\n"
                         "  --tol T         ESKernel tolerance (default 1e-6)\n"
                         "  --warmup W      warmup runs (default 5)\n"
                         "  --runs K        timed runs (default 20)\n"
                         "  --dist D        uniform|clustered (default uniform)\n"
                         "  --output FILE   output CSV (default scatter_es_roofline.csv)\n"
                         "  --ncu-mode      single-run for Nsight Compute\n"
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
    if (s.n == 0)
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

// --------------------- Microbench: atomic no contention ---------------------
template <class ExecSpace>
double atomic_nocont_ops_per_sec(size_t nThreads, int iters, int repeats) {
    Kokkos::View<float*, ExecSpace> v("v", nThreads);
    Kokkos::deep_copy(v, 0.0f);

    auto run_once = [&]() {
        Kokkos::parallel_for(
            "atomic_nocont", Kokkos::RangePolicy<ExecSpace>(0, nThreads),
            KOKKOS_LAMBDA(const int i) {
                for (int k = 0; k < iters; ++k) {
                    Kokkos::atomic_add(&v(i), 1.0f);
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

    const double ops = double(nThreads) * double(iters);
    return ops / best;
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

    int run() {
        if (ippl::Comm->rank() == 0)
            print_header();

        const int w      = kernel_.width();
        const int nghost = w / 2 + 1;

        setup_domain(nghost);
        init_particles();
        init_grid(nghost);

        // Scatter configs to test
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
            auto cfg   = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
            // heuristic from your snippet (safe fallback)
            int tile_size                  = 2;
            std::array<int, 10> tile_sizes = {1, 1, 4, 3, 3, 2, 4, 2, 2, 2};
            if (w < 10)
                tile_size = tile_sizes[w];
            cfg.tile_size.fill(tile_size);
            variants.push_back({"GridParallel", cfg});
        }

        // Baseline “ideal atomic” (no contention)
        // Use same logical parallelism as local particles; a few iters to stabilize.
        const int microIters      = 8;
        double atomic_nocont_rate = atomic_nocont_ops_per_sec<ExecSpace>(n_local_, microIters, 5);

        // MPI-reduce atomic roofline as a *sum of per-rank rates* (approx for identical GPUs).
        double atomic_nocont_rate_global = 0.0;
        MPI_Allreduce(&atomic_nocont_rate, &atomic_nocont_rate_global, 1, MPI_DOUBLE, MPI_SUM,
                      ippl::Comm->getCommunicator());

        // Run variants and write CSV
        if (ippl::Comm->rank() == 0)
            write_csv_header();

        for (auto& v : variants) {
            auto res = bench_scatter_variant(v.name, v.cfg, nghost);

            // Global particle + update accounting
            double updates_local  = double(n_local_) * std::pow(double(kernel_.width()), 3.0);
            double updates_global = 0.0;
            MPI_Allreduce(&updates_local, &updates_global, 1, MPI_DOUBLE, MPI_SUM,
                          ippl::Comm->getCommunicator());

            double t_mean_s = res.total_stats.mean_ms * 1e-3;
            // The timing is already global-synchronized by fences, but each rank timed its own
            // work. We use the *max* mean across ranks to be conservative.
            double t_mean_s_global = 0.0;
            MPI_Allreduce(&t_mean_s, &t_mean_s_global, 1, MPI_DOUBLE, MPI_MAX,
                          ippl::Comm->getCommunicator());

            double scatter_updates_per_s = updates_global / t_mean_s_global;
            double util                  = scatter_updates_per_s / atomic_nocont_rate_global;

            if (ippl::Comm->rank() == 0) {
                append_csv_row(v.name, res, scatter_updates_per_s, atomic_nocont_rate_global, util);
                if (p_.verbose) {
                    std::cout << "Variant " << v.name << " mean_ms=" << res.total_stats.mean_ms
                              << " updates/s=" << scatter_updates_per_s
                              << " atomic_nocont_ops/s=" << atomic_nocont_rate_global
                              << " util=" << util << "\n";
                }
            }
        }

        cleanup();
        return 0;
    }

    struct Result {
        std::vector<double> total_times_s;
        Stats total_stats;
    };

    void print_header() {
        std::cout << "\n============================================================\n";
        std::cout << " Scatter benchmark (ESKernel) + atomic roofline (nocont)\n";
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

        // Global particles split across ranks
        const size_t n_global = p_.n_particles_global();
        const int nr          = ippl::Comm->size();
        const int r           = ippl::Comm->rank();

        const size_t base = n_global / size_t(nr);
        const size_t rem  = n_global % size_t(nr);
        n_local_          = base + (size_t(r) < rem ? 1 : 0);

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
                    for (unsigned d = 0; d < Dim; ++d) {
                        Rv(i)[d] = gen.drand() * 2.0 * M_PI;
                    }
                    Qv(i) = complex_type(1.0, 0.0);
                    pool.free_state(gen);
                });
        } else {  // clustered
            Kokkos::parallel_for(
                "init_R_clustered", Kokkos::RangePolicy<ExecSpace>(0, n_local_),
                KOKKOS_LAMBDA(const int i) {
                    auto gen = pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        double u1 = gen.drand();
                        double u2 = gen.drand();
                        double z  = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-12))
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
                                 const ippl::Interpolation::ScatterConfig<Dim>& cfg,
                                 int /*nghost*/) {
        if (ippl::Comm->rank() == 0 && p_.verbose) {
            std::cout << "Benchmark scatter: " << name << "\n";
        }

        ManualTimer t;

        // warmup
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

    void write_csv_header() {
        // overwrite each run (simple)
        std::ofstream out(p_.output);
        out << "kernel_variant,dist,grid,rho,particles_global,particles_local,"
               "tol,width,warmup,runs,"
               "mean_ms,std_ms,median_ms,min_ms,max_ms,"
               "scatter_updates_per_s,atomic_nocont_ops_per_s,util\n";
    }

    void append_csv_row(const std::string& variant, const Result& r, double scatter_updates_per_s,
                        double atomic_nocont_ops_per_s_global, double util) {
        std::ofstream out(p_.output, std::ios::app);

        out << variant << "," << p_.dist << "," << p_.grid << "," << std::fixed
            << std::setprecision(3) << p_.rho << "," << p_.n_particles_global() << "," << n_local_
            << "," << std::scientific << std::setprecision(2) << p_.tol << "," << kernel_.width()
            << "," << p_.warmup << "," << p_.runs << "," << std::fixed << std::setprecision(4)
            << r.total_stats.mean_ms << "," << r.total_stats.std_ms << ","
            << r.total_stats.median_ms << "," << r.total_stats.min_ms << "," << r.total_stats.max_ms
            << "," << std::scientific << std::setprecision(6) << scatter_updates_per_s << ","
            << atomic_nocont_ops_per_s_global << "," << std::fixed << std::setprecision(6) << util
            << "\n";
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
