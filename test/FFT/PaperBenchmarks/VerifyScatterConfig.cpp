/**
 * @file VerifyScatterConfig.cpp
 * @brief Verify that configs in tile_sweep_sa_optimal.csv achieve the claimed throughput.
 *
 * TileSweep benchmarks scatter on the oversampled NUFFT grid (bit_ceil(2*N)^3) and
 * records the achieved throughput in the CSV.  This test loads that CSV, re-creates
 * the same scatter setup (grid size, particle count), runs with the exact stored
 * config, and checks that the achieved Mpts/s is within a tolerance of the claimed
 * value.  It is designed to catch regressions and to confirm that the cache configs
 * are hardware-accurate before a paper benchmark run.
 *
 * Usage: ./VerifyScatterConfig [options]
 *   --csv PATH      Path to tile_sweep_sa_optimal.csv
 *   --grid N        Scatter grid size per dim (default: 256 = bit_ceil(2*128))
 *   --rho R         Particles per scatter-grid point (default: 1.25 = 10/8,
 *                   matching TileSweep --rho 10 with sigma=2)
 *   --width W       Only verify kernel width W  (default: all widths in CSV)
 *   --method M      Only verify method M: Tiled|OutputFocused|Atomic
 *                   (default: all methods present in CSV)
 *   --real          Use real-valued field instead of complex
 *   --warmup N      Warmup iterations  (default: 5)
 *   --runs N        Timed iterations   (default: 20)
 *   --tol T         Pass threshold: achieved >= T * claimed  (default: 0.85)
 *   -v, --verbose   Print per-run timings
 *
 * Exit code: 0 if all checks pass, 1 if any claimed throughput is not reproduced.
 */

#include "Ippl.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "FFT/NUFFT/ESKernel.h"
#include "Interpolation/Scatter/ScatterConfig.h"

// ── tiny helpers ─────────────────────────────────────────────────────────────

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
}

// ── CSV row representation ────────────────────────────────────────────────────

struct CsvRow {
    std::string method;
    bool        is_complex;
    int         kernel_width;
    double      rho;
    std::array<int, 3> tile;
    int    team_size;
    int    oversubscription;
    int    z_batches;
    double claimed_tp;  // Mpts/s
};

static std::vector<std::string> csv_split(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto l = tok.find_first_not_of(" \t\r\n");
        auto r = tok.find_last_not_of(" \t\r\n");
        out.push_back(l == std::string::npos ? "" : tok.substr(l, r - l + 1));
    }
    return out;
}

static std::vector<CsvRow> load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ERROR] Cannot open CSV: " << path << "\n";
        return {};
    }
    std::string hdr;
    std::getline(f, hdr);
    if (hdr.find("best_tile_x") == std::string::npos) {
        std::cerr << "[ERROR] Expected BO/SA CSV with best_tile_x column.\n";
        return {};
    }
    const bool has_rho = (hdr.find(",rho,") != std::string::npos);

    std::vector<CsvRow> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto fs = csv_split(line);
        // cols: method, value_type, kernel_width, [rho,] tx, ty, tz, team, osub, zb, tp, ...
        int tc = has_rho ? 4 : 3;
        if ((int)fs.size() < tc + 7) continue;

        // Only known methods
        const std::string& m = fs[0];
        if (m != "Tiled" && m != "OutputFocused" && m != "Atomic") continue;

        CsvRow r;
        r.method          = m;
        r.is_complex      = (fs[1] == "complex");
        r.kernel_width    = std::stoi(fs[2]);
        r.rho             = has_rho ? std::stod(fs[3]) : 0.0;
        r.tile            = {std::stoi(fs[tc]), std::stoi(fs[tc+1]), std::stoi(fs[tc+2])};
        r.team_size       = std::stoi(fs[tc+3]);
        r.oversubscription = std::stoi(fs[tc+4]);
        r.z_batches       = std::stoi(fs[tc+5]);
        r.claimed_tp      = ((int)fs.size() > tc+6) ? std::stod(fs[tc+6]) : 0.0;
        rows.push_back(r);
    }
    return rows;
}

// ── Domain + particle setup ──────────────────────────────────────────────────

static constexpr unsigned Dim = 3;
using real_type    = double;
using complex_type = Kokkos::complex<real_type>;
using ExecSpace    = Kokkos::DefaultExecutionSpace;
using Mesh_t       = ippl::UniformCartesian<real_type, Dim>;
using Centering_t  = Mesh_t::DefaultCentering;

template <typename ValueT>
struct ScatterSetup {
    using Field_t   = ippl::Field<ValueT, Dim, Mesh_t, Centering_t>;
    using PLayout_t = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t   = ippl::ParticleBase<PLayout_t>;

    std::unique_ptr<ippl::FieldLayout<Dim>> layout;
    std::unique_ptr<Mesh_t>                 mesh;
    std::unique_ptr<Field_t>                field;
    std::unique_ptr<PLayout_t>              playout;
    std::unique_ptr<Bunch_t>                bunch;
    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R;
    ippl::ParticleAttrib<ValueT>                        Q;

    void build(size_t n_grid, size_t n_particles, int nghost) {
        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d) domain[d] = ippl::Index(n_grid);
        std::array<bool, Dim> par; par.fill(true);
        layout = std::make_unique<ippl::FieldLayout<Dim>>(MPI_COMM_WORLD, domain, par, true);

        ippl::Vector<real_type, Dim> hx, orig;
        for (unsigned d = 0; d < Dim; ++d) {
            orig[d] = 0.0;
            hx[d]   = 2.0 * M_PI / static_cast<real_type>(n_grid);
        }
        mesh    = std::make_unique<Mesh_t>(domain, hx, orig);
        field   = std::make_unique<Field_t>(*mesh, *layout, nghost);
        playout = std::make_unique<PLayout_t>(*layout, *mesh);
        bunch   = std::make_unique<Bunch_t>(*playout);
        bunch->addAttribute(R);
        bunch->addAttribute(Q);
        bunch->setParticleBC(ippl::BC::PERIODIC);

        const size_t n_local = n_particles / static_cast<size_t>(ippl::Comm->size());
        bunch->create(n_local);

        auto Rv = R.getView();
        Kokkos::Random_XorShift64_Pool<> pool(42 + ippl::Comm->rank());
        Kokkos::parallel_for("init_pos", n_local, KOKKOS_LAMBDA(size_t i) {
            auto gen = pool.get_state();
            for (unsigned d = 0; d < Dim; ++d) Rv(i)[d] = gen.drand() * 2.0 * M_PI;
            pool.free_state(gen);
        });
        auto Qv = Q.getView();
        Kokkos::parallel_for("init_val", n_local, KOKKOS_LAMBDA(size_t i) {
            Qv(i) = ValueT(1);
        });
        Kokkos::fence();
    }
};

// ── Single verification run ───────────────────────────────────────────────────

template <typename ValueT>
double run_and_measure(size_t n_grid, size_t n_particles, const CsvRow& row,
                       int warmup, int runs, bool verbose) {
    // Construct kernel directly from width (beta_factor = 2.30, same as ESKernel)
    using Kernel_t = ippl::nufft::ESKernel<real_type>;
    const Kernel_t kernel(row.kernel_width,
                          static_cast<real_type>(2.30) * row.kernel_width);
    const int nghost = row.kernel_width / 2 + 1;

    ScatterSetup<ValueT> setup;
    setup.build(n_grid, n_particles, nghost);

    // Build config from CSV row, locking method so TileSizeCache is not consulted
    auto cfg                    = ippl::Interpolation::ScatterConfig<Dim>::get_default<ExecSpace>();
    if      (row.method == "Tiled")         cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
    else if (row.method == "OutputFocused") cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
    else                                    cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    cfg.lock_method        = true;
    cfg.enable_tuning      = false;
    cfg.tile_size[0]       = row.tile[0];
    cfg.tile_size[1]       = row.tile[1];
    cfg.tile_size[2]       = row.tile[2];
    if (row.team_size > 0)       cfg.team_size               = row.team_size;
    if (row.oversubscription > 0) cfg.oversubscription_factor = row.oversubscription;
    if (row.z_batches > 0)       cfg.z_batches               = row.z_batches;

    auto reset = [&]() {
        *setup.field = ValueT(0);
        Kokkos::fence();
    };
    auto scatter_once = [&]() {
        reset();
        setup.Q.scatter_kernel(*setup.field, setup.R, kernel, cfg);
        setup.field->accumulateHalo();
        Kokkos::fence();
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) scatter_once();

    // Timed runs
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        setup.Q.scatter_kernel(*setup.field, setup.R, kernel, cfg);
        setup.field->accumulateHalo();
        Kokkos::fence();
        auto   t1  = std::chrono::high_resolution_clock::now();
        double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
        if (verbose && ippl::Comm->rank() == 0)
            std::cout << "    run " << (i + 1) << ": " << ms << " ms\n";
    }

    const double med_ms = median_of(times);
    return (static_cast<double>(n_particles) / (med_ms * 1e-3)) / 1e6;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Parse args
        std::string csv_path;
        size_t      n_grid        = 256;    // bit_ceil(2*128)
        double      rho           = 1.25;   // 10 / sigma^3 = 10/8
        int         warmup        = 5;
        int         runs          = 20;
        double      pass_tol      = 0.85;
        int         filter_w      = -1;
        std::string filter_method;
        bool        use_real      = false;
        bool        verbose       = false;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--csv"    && i+1 < argc) csv_path      = argv[++i];
            else if (a == "--grid"   && i+1 < argc) n_grid        = std::stoul(argv[++i]);
            else if (a == "--rho"    && i+1 < argc) rho           = std::stod(argv[++i]);
            else if (a == "--width"  && i+1 < argc) filter_w      = std::stoi(argv[++i]);
            else if (a == "--method" && i+1 < argc) filter_method = argv[++i];
            else if (a == "--warmup" && i+1 < argc) warmup        = std::stoi(argv[++i]);
            else if (a == "--runs"   && i+1 < argc) runs          = std::stoi(argv[++i]);
            else if (a == "--tol"    && i+1 < argc) pass_tol      = std::stod(argv[++i]);
            else if (a == "--real")                  use_real      = true;
            else if (a == "-v" || a == "--verbose")  verbose       = true;
        }

        if (csv_path.empty()) csv_path = "tile_sweep_sa_optimal.csv";

        const size_t n_particles =
            static_cast<size_t>(rho * static_cast<double>(n_grid) * n_grid * n_grid);

        if (ippl::Comm->rank() == 0) {
            std::cout << "=== VerifyScatterConfig ===\n"
                      << "  CSV         : " << csv_path << "\n"
                      << "  grid        : " << n_grid << "^3\n"
                      << "  rho         : " << rho
                      << "  (" << n_particles << " particles)\n"
                      << "  warmup/runs : " << warmup << " / " << runs << "\n"
                      << "  pass_tol    : achieved >= " << (pass_tol * 100.0) << "% of claimed\n"
                      << "  value_type  : " << (use_real ? "real" : "complex") << "\n";
            if (filter_w > 0)           std::cout << "  filter width: " << filter_w << "\n";
            if (!filter_method.empty()) std::cout << "  filter method: " << filter_method << "\n";
            std::cout << "\n";
        }

        const std::vector<CsvRow> rows = load_csv(csv_path);
        if (rows.empty()) {
            if (ippl::Comm->rank() == 0)
                std::cerr << "[ERROR] No valid rows loaded from " << csv_path << "\n";
            ippl::finalize();
            return 1;
        }

        int n_checked = 0, n_passed = 0, n_failed = 0;

        for (const auto& row : rows) {
            if (filter_w > 0 && row.kernel_width != filter_w) continue;
            if (!filter_method.empty() && row.method != filter_method) continue;
            // Match value type: row.is_complex==true → complex field, use_real==false → complex
            if (row.is_complex == use_real) continue;

            ++n_checked;

            if (ippl::Comm->rank() == 0) {
                std::cout << "--- " << row.method
                          << "  w=" << row.kernel_width
                          << "  tile=(" << row.tile[0] << "," << row.tile[1]
                          << "," << row.tile[2] << ")"
                          << "  team=" << row.team_size
                          << "  zb=" << row.z_batches
                          << "  claimed=" << std::fixed << std::setprecision(1)
                          << row.claimed_tp << " Mpts/s ---\n";
            }

            double achieved;
            if (use_real)
                achieved = run_and_measure<real_type>   (n_grid, n_particles, row, warmup, runs, verbose);
            else
                achieved = run_and_measure<complex_type>(n_grid, n_particles, row, warmup, runs, verbose);

            if (ippl::Comm->rank() == 0) {
                const double ratio = (row.claimed_tp > 0.0) ? achieved / row.claimed_tp : 1.0;
                const bool   pass  = (row.claimed_tp <= 0.0) || (achieved >= pass_tol * row.claimed_tp);
                std::cout << "  achieved : " << std::fixed << std::setprecision(1) << achieved
                          << " Mpts/s  ("
                          << std::setprecision(0) << (ratio * 100.0) << "% of claimed)  "
                          << (pass ? "[PASS]" : "[FAIL]") << "\n\n";
                if (pass) ++n_passed; else ++n_failed;
            }
        }

        if (ippl::Comm->rank() == 0) {
            std::cout << "=== Summary: " << n_checked << " config(s) checked, "
                      << n_passed << " passed, " << n_failed << " failed ===\n";
        }

        if (n_failed > 0) {
            ippl::finalize();
            return 1;
        }
    }
    ippl::finalize();
    return 0;
}
