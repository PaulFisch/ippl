// BenchmarkScatterAtomic.cpp
#include "Ippl.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "Interpolation/Scatter/Scatter.h"
#include "Interpolation/Scatter/ScatterConfig.h"
#include "Interpolation/Kernels.h"

// --------------------- tiny CLI ---------------------
struct Args {
    int dim = 3;
    int grid = 128;                 // per-dimension grid size (uniform)
    int nParticles = 2'000'000;     // per rank
    int repeats = 10;
    bool sort = true;
    std::string method = "Atomic";  // Atomic|Tiled|OutputFocused
    std::string kernel = "Cubic";   // NGP|Linear|Quadratic|Cubic
    std::string csv = "scatter_bench.csv";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto get = [&](const char* k) -> const char* {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << k << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--dim")) a.dim = std::atoi(get("--dim"));
        else if (!std::strcmp(argv[i], "--grid")) a.grid = std::atoi(get("--grid"));
        else if (!std::strcmp(argv[i], "--n")) a.nParticles = std::atoi(get("--n"));
        else if (!std::strcmp(argv[i], "--repeats")) a.repeats = std::atoi(get("--repeats"));
        else if (!std::strcmp(argv[i], "--sort")) a.sort = std::atoi(get("--sort")) != 0;
        else if (!std::strcmp(argv[i], "--method")) a.method = get("--method");
        else if (!std::strcmp(argv[i], "--kernel")) a.kernel = get("--kernel");
        else if (!std::strcmp(argv[i], "--csv")) a.csv = get("--csv");
        else if (!std::strcmp(argv[i], "--help")) {
            std::cout <<
              "Usage: BenchmarkScatterAtomic [options]\n"
              "  --dim 2|3\n"
              "  --grid N              (per dim)\n"
              "  --n N                 particles per rank\n"
              "  --repeats N\n"
              "  --sort 0|1\n"
              "  --method Atomic|Tiled|OutputFocused\n"
              "  --kernel NGP|Linear|Quadratic|Cubic\n"
              "  --csv path\n";
            std::exit(0);
        }
    }
    return a;
}

// --------------------- timing helper ---------------------
template <class F>
double time_best_seconds(int repeats, F&& f) {
    double best = 1e100;
    for (int r = 0; r < repeats; ++r) {
        ippl::fence();
        double t0 = MPI_Wtime();
        f();
        ippl::fence();
        double t1 = MPI_Wtime();
        best = std::min(best, t1 - t0);
    }
    return best;
}

// --------------------- Kokkos microbench ---------------------
struct MicroBenchRes {
    double seconds = 0.0;
    double ops = 0.0;
    double gops = 0.0;
};

template <class ExecSpace>
MicroBenchRes bench_atomic_nocont(std::size_t nThreads, int iters, int repeats) {
    Kokkos::View<float*, ExecSpace> v("v", nThreads);
    Kokkos::deep_copy(v, 0.0f);

    auto best = time_best_seconds(repeats, [&](){
        Kokkos::parallel_for("atomic_nocont",
            Kokkos::RangePolicy<ExecSpace>(0, nThreads),
            KOKKOS_LAMBDA(const int i){
                for (int k = 0; k < iters; ++k) {
                    Kokkos::atomic_add(&v(i), 1.0f);
                }
            });
        Kokkos::fence();
    });

    MicroBenchRes r;
    r.seconds = best;
    r.ops = double(nThreads) * double(iters);
    r.gops = (r.ops / r.seconds) / 1e9;
    return r;
}

template <class ExecSpace>
MicroBenchRes bench_atomic_cont(std::size_t nThreads, int iters, int repeats) {
    Kokkos::View<float*, ExecSpace> v("v", 1);
    Kokkos::deep_copy(v, 0.0f);

    auto best = time_best_seconds(repeats, [&](){
        Kokkos::parallel_for("atomic_cont",
            Kokkos::RangePolicy<ExecSpace>(0, nThreads),
            KOKKOS_LAMBDA(const int){
                for (int k = 0; k < iters; ++k) {
                    Kokkos::atomic_add(&v(0), 1.0f);
                }
            });
        Kokkos::fence();
    });

    MicroBenchRes r;
    r.seconds = best;
    r.ops = double(nThreads) * double(iters);
    r.gops = (r.ops / r.seconds) / 1e9;
    return r;
}

template <class ExecSpace>
MicroBenchRes bench_write_nocont(std::size_t nThreads, int iters, int repeats) {
    Kokkos::View<float*, ExecSpace> v("v", nThreads);
    Kokkos::deep_copy(v, 0.0f);

    auto best = time_best_seconds(repeats, [&](){
        Kokkos::parallel_for("write_nocont",
            Kokkos::RangePolicy<ExecSpace>(0, nThreads),
            KOKKOS_LAMBDA(const int i){
                float x = 0.0f;
                for (int k = 0; k < iters; ++k) {
                    x += 1.0f;
                    v(i) = x;
                }
            });
        Kokkos::fence();
    });

    MicroBenchRes r;
    r.seconds = best;
    r.ops = double(nThreads) * double(iters);
    r.gops = (r.ops / r.seconds) / 1e9;
    return r;
}

// --------------------- scatter runner (templated on Dim + Kernel) ---------------------
template <typename T, typename ExecSpace, unsigned Dim, typename Kernel>
int run_scatter(const Args& a) {
    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using layout_type    = ippl::FieldLayout<Dim>;
    using playout_type   = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;
    using field_type     = ippl::Field<T, Dim, mesh_type, typename mesh_type::DefaultCentering, ExecSpace>::uniform_type;

    // domain/grid
    ippl::Vector<T, Dim> origin, extent, hx;
    ippl::Vector<std::size_t, Dim> gridSize;
    const T L = T(1.0);

    for (unsigned d = 0; d < Dim; ++d) {
        origin[d] = 0.0;
        extent[d] = L;
        gridSize[d] = std::size_t(a.grid);
        hx[d] = extent[d] / T(gridSize[d]);
    }

    Kernel kernel;

    int nghost = kernel.width() / 2 + 1;

    std::array<ippl::Index, Dim> domains;
    std::array<bool, Dim> isParallel;
    isParallel.fill(true);
    for (unsigned d = 0; d < Dim; ++d) domains[d] = ippl::Index(gridSize[d]);

    auto owned  = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
    auto layout = std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel, true, nghost);
    auto mesh   = std::make_shared<mesh_type>(owned, hx, origin);

    auto playout = std::make_shared<playout_type>(*layout, *mesh);

    // Minimal bunch: positions + weights
    struct Bunch : public ippl::ParticleBase<playout_type> {
        ippl::ParticleAttrib<T> weight;
        explicit Bunch(playout_type& L) : ippl::ParticleBase<playout_type>(L) { this->addAttribute(weight); }
    };

    auto bunch = std::make_shared<Bunch>(*playout);
    bunch->create(a.nParticles);

    // random particle positions and weights
    auto R_view = bunch->R.getView();
    auto w_view = bunch->weight.getView();
    using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
    RandPool pool(1234 + ippl::Comm->rank() * 101);

    auto origin_local = origin;
    auto extent_local = extent;

    Kokkos::parallel_for("init_particles", Kokkos::RangePolicy<ExecSpace>(0, a.nParticles),
        KOKKOS_LAMBDA(const int i){
            auto gen = pool.get_state();
            ippl::Vector<T, Dim> pos;
            for (unsigned d = 0; d < Dim; ++d) {
                pos[d] = origin_local[d] + gen.drand() * extent_local[d];
            }
            R_view(i) = pos;
            w_view(i) = T(1.0);
            pool.free_state(gen);
        });
    Kokkos::fence();
    bunch->update();
    ippl::fence();

    field_type field(*mesh, *layout, nghost);
    field = T(0);

    // scatter config
    ippl::Interpolation::ScatterConfig<Dim> cfg;
    cfg.sort = a.sort;
    if (a.method == "Atomic") cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    else if (a.method == "Tiled") cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
    else if (a.method == "OutputFocused") cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
    else {
        if (ippl::Comm->rank() == 0) std::cerr << "Unknown --method " << a.method << "\n";
        return 2;
    }

    auto scatter = ippl::Scatter(kernel, cfg);

    // warmup
    scatter(field, bunch->R, bunch->weight);
    ippl::fence();

    // time
    double t_best = time_best_seconds(a.repeats, [&](){
        field = T(0);
        scatter(field, bunch->R, bunch->weight);
    });

    // Estimate "updates" = particles * stencil_points.
    // This is the modeling quantity you likely care about for histogram/stencil throughput.
    const double stencil_points = std::pow(double(kernel.width()), double(Dim));
    const double updates = double(a.nParticles) * stencil_points;
    const double updates_per_s = updates / t_best;

    // microbench baseline (Kokkos atomics)
    // Use same logical parallelism as particle count; do a few iters to stabilize.
    constexpr int microIters = 8;

    auto m_nocont = bench_atomic_nocont<ExecSpace>(std::size_t(a.nParticles), microIters, 5);
    auto m_cont   = bench_atomic_cont<ExecSpace>(std::size_t(a.nParticles), microIters, 5);
    auto m_write  = bench_write_nocont<ExecSpace>(std::size_t(a.nParticles), microIters, 5);

    // Convert baseline to per-op (atomicAdd) rate; compare scatter updates vs atomic ops rate.
    // Baseline ops/sec:
    const double atomic_nocont_ops_per_s = m_nocont.ops / m_nocont.seconds;

    const double util = updates_per_s / atomic_nocont_ops_per_s;

    if (ippl::Comm->rank() == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Scatter: dim=" << Dim
                  << " kernel_width=" << kernel.width()
                  << " stencil_points=" << stencil_points
                  << " method=" << a.method
                  << " sort=" << (a.sort ? 1 : 0)
                  << " n=" << a.nParticles
                  << " best_s=" << t_best
                  << " updates/s=" << updates_per_s
                  << "\n";

        std::cout << "Microbench (Kokkos):\n"
                  << "  atomic_nocont: " << (atomic_nocont_ops_per_s/1e9) << " Gop/s\n"
                  << "  atomic_cont  : " << ((m_cont.ops/m_cont.seconds)/1e9) << " Gop/s\n"
                  << "  write_nocont : " << ((m_write.ops/m_write.seconds)/1e9) << " Gop/s\n"
                  << "  utilization(scatter_updates / atomic_nocont_ops) = " << util << "\n";

        // CSV header + row
        // Note: append mode; remove file if you want a clean sweep.
        std::FILE* f = std::fopen(a.csv.c_str(), "a");
        if (f) {
            // If file empty, write header (best-effort)
            std::fseek(f, 0, SEEK_END);
            long sz = std::ftell(f);
            if (sz == 0) {
                std::fprintf(f,
                    "dim,grid,kernel,width,stencil_points,method,sort,nParticles,"
                    "scatter_seconds,scatter_updates_per_s,"
                    "atomic_nocont_ops_per_s,atomic_cont_ops_per_s,write_nocont_ops_per_s,util\n");
            }
            std::fprintf(f,
                "%u,%d,%s,%d,%.0f,%s,%d,%d,%.9f,%.6e,%.6e,%.6e,%.6e,%.6f\n",
                Dim, a.grid, a.kernel.c_str(), kernel.width(), stencil_points,
                a.method.c_str(), (a.sort ? 1 : 0), a.nParticles,
                t_best, updates_per_s,
                atomic_nocont_ops_per_s,
                (m_cont.ops/m_cont.seconds),
                (m_write.ops/m_write.seconds),
                util);
            std::fclose(f);
        } else {
            std::cerr << "Could not open CSV: " << a.csv << "\n";
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    Args a = parse_args(argc, argv);

    using T = double;
    using ExecSpace = Kokkos::DefaultExecutionSpace;

    int rc = 0;

    auto run_dim = [&](auto dim_tag) {
        constexpr unsigned Dim = decltype(dim_tag)::value;

        if (a.kernel == "NGP")       rc = run_scatter<T, ExecSpace, Dim, ippl::Interpolation::NGPKernel<T>>(a);
        else if (a.kernel == "Linear")    rc = run_scatter<T, ExecSpace, Dim, ippl::Interpolation::LinearKernel<T>>(a);
        else if (a.kernel == "Quadratic") rc = run_scatter<T, ExecSpace, Dim, ippl::Interpolation::QuadraticKernel<T>>(a);
        else if (a.kernel == "Cubic")     rc = run_scatter<T, ExecSpace, Dim, ippl::Interpolation::CubicKernel<T>>(a);
        else {
            if (ippl::Comm->rank() == 0) std::cerr << "Unknown --kernel " << a.kernel << "\n";
            rc = 2;
        }
    };

    if (a.dim == 2) run_dim(std::integral_constant<unsigned, 2>{});
    else if (a.dim == 3) run_dim(std::integral_constant<unsigned, 3>{});
    else {
        if (ippl::Comm->rank() == 0) std::cerr << "Unsupported --dim " << a.dim << " (use 2 or 3)\n";
        rc = 2;
    }

    ippl::finalize();
    return rc;
}
