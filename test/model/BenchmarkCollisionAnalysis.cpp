/**
 * @file BenchmarkCollisionAnalysis.cpp
 * @brief CORRECTED collision analysis with windowed metrics
 *
 * KEY INSIGHT: phi_global and phi_block do NOT capture temporal contention.
 * Sorting can make atomics SLOWER because it increases m_L (cache-line
 * multiplicity within particles-in-flight), even if global overlap is unchanged.
 *
 * This benchmark computes:
 *   - m_L (windowed): Updates per cache line within P_in_flight window
 *   - phi_global: For reference ONLY (not used in model)
 *   - phi_block: For reference ONLY (misses inter-block contention)
 *
 * Usage: ./BenchmarkCollisionAnalysis [options]
 *   --grid N         Grid size (default: 64)
 *   --rho R          Particles per grid point (default: 10)
 *   --tol T          Kernel tolerance (default: 1e-6)
 *   --window W       Particles in flight (default: auto from SM_count)
 *   --dist D         Distribution: uniform, clustered (default: uniform)
 *   -v, --verbose    Verbose output
 */

#include "Ippl.h"
#include <Kokkos_Random.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <numeric>

using namespace ippl;

// ============================================================================
// Parameters
// ============================================================================

struct Params {
    int n_grid = 64;
    double rho = 10.0;
    double kernel_tol = 1e-6;
    size_t window_size = 0;  // 0 = auto
    std::string distribution = "uniform";
    std::string output = "collision_analysis_v2.csv";
    bool verbose = false;

    size_t n_particles() const {
        return static_cast<size_t>(rho * n_grid * n_grid * n_grid);
    }
};

Params parse_args(int argc, char* argv[]) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--grid" && i + 1 < argc) p.n_grid = std::atoi(argv[++i]);
        else if (arg == "--rho" && i + 1 < argc) p.rho = std::atof(argv[++i]);
        else if (arg == "--tol" && i + 1 < argc) p.kernel_tol = std::atof(argv[++i]);
        else if (arg == "--window" && i + 1 < argc) p.window_size = std::stoull(argv[++i]);
        else if (arg == "--dist" && i + 1 < argc) p.distribution = argv[++i];
        else if (arg == "--output" && i + 1 < argc) p.output = argv[++i];
        else if (arg == "-v" || arg == "--verbose") p.verbose = true;
    }
    return p;
}

// ============================================================================
// Results
// ============================================================================

struct Results {
    std::string ordering;

    // WHAT THE MODEL USES:
    double m_L;              // Updates per cache line within window
    double m_L_stddev;
    double m_L_max;
    size_t avg_lines_per_window;

    // FOR REFERENCE ONLY (NOT FOR MODEL):
    double phi_global;       // = N*S / unique_lines (meaningless for contention)
    double phi_block_mean;   // Within-block overlap (misses inter-block contention)

    size_t total_cache_lines;
    size_t total_updates;
    size_t window_size;
};

// ============================================================================
// Morton sorting
// ============================================================================

template <typename real_type, unsigned Dim>
uint64_t morton_key(const ippl::Vector<real_type, Dim>& pos, double h, int n_grid) {
    int ix[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        ix[d] = static_cast<int>(pos[d] / h);
        ix[d] = std::max(0, std::min(n_grid - 1, ix[d]));
    }

    auto expand = [](uint64_t v) {
        v = (v * 0x0001000100010001ULL) & 0xFFFF00000000FFFFULL;
        v = (v * 0x0000000100000001ULL) & 0x00FF0000FF0000FFULL;
        v = (v * 0x0000000000010001ULL) & 0xF00F00F00F00F00FULL;
        v = (v * 0x0000000000000101ULL) & 0x30C30C30C30C30C3ULL;
        v = (v * 0x0000000000000005ULL) & 0x9249249249249249ULL;
        return v;
    };

    if constexpr (Dim == 3) {
        return expand(ix[0]) | (expand(ix[1]) << 1) | (expand(ix[2]) << 2);
    }
    return 0;
}

// ============================================================================
// Main Analyzer
// ============================================================================

template <typename ExecSpace>
class CollisionAnalyzer {
public:
    static constexpr unsigned Dim = 3;
    using real_type = double;
    using Vector = ippl::Vector<real_type, Dim>;
    using MemSpace = typename ExecSpace::memory_space;

    CollisionAnalyzer(Params& params) : params_(params) {
        if (params_.window_size == 0) {
            int sm_count = 80;
#ifdef KOKKOS_ENABLE_CUDA
            int d; cudaGetDevice(&d);
            cudaDeviceProp prop; cudaGetDeviceProperties(&prop, d);
            sm_count = prop.multiProcessorCount;
#elif defined(KOKKOS_ENABLE_HIP)
            int d; hipGetDevice(&d);
            hipDeviceProp_t prop; hipGetDeviceProperties(&prop, d);
            sm_count = prop.multiProcessorCount;
#endif
            params_.window_size = sm_count * 8 * 4;  // SM * blocks_per_SM * team_size
        }
    }

    void run() {
        print_header();

        ippl::NUFFT::ESKernel<real_type> kernel(params_.kernel_tol);
        int w = kernel.width();

        size_t N = params_.n_particles();
        auto positions = generate_particles(N);

        std::vector<Results> results;

        // Unsorted
        results.push_back(analyze(positions, N, w, "unsorted"));

        // Morton sorted
        auto sorted = sort_by_morton(positions, N);
        results.push_back(analyze(sorted, N, w, "morton_sorted"));

        print_results(results);
        write_csv(results);
    }

    Kokkos::View<Vector*, Kokkos::HostSpace> generate_particles(size_t N) {
        Kokkos::View<Vector*, MemSpace> pos_d("pos", N);
        Kokkos::Random_XorShift64_Pool<> pool(42);

        if (params_.distribution == "uniform") {
            Kokkos::parallel_for("init", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(size_t i) {
                    auto gen = pool.get_state();
                    Vector p;
                    for (unsigned d = 0; d < Dim; ++d) p[d] = gen.drand() * 2.0 * M_PI;
                    pos_d(i) = p;
                    pool.free_state(gen);
                });
        } else {
            Kokkos::parallel_for("init", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(size_t i) {
                    auto gen = pool.get_state();
                    Vector p;
                    for (unsigned d = 0; d < Dim; ++d) {
                        double u1 = gen.drand(), u2 = gen.drand();
                        double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-10)) * Kokkos::cos(2.0 * M_PI * u2);
                        p[d] = M_PI + 0.3 * z;
                        while (p[d] < 0) p[d] += 2.0 * M_PI;
                        while (p[d] >= 2.0 * M_PI) p[d] -= 2.0 * M_PI;
                    }
                    pos_d(i) = p;
                    pool.free_state(gen);
                });
        }
        Kokkos::fence();
        return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pos_d);
    }

    Kokkos::View<Vector*, Kokkos::HostSpace> sort_by_morton(
        const Kokkos::View<Vector*, Kokkos::HostSpace>& pos, size_t N)
    {
        double h = 2.0 * M_PI / params_.n_grid;
        std::vector<std::pair<uint64_t, size_t>> keyed(N);
        for (size_t i = 0; i < N; ++i) {
            keyed[i] = {morton_key<real_type, Dim>(pos(i), h, params_.n_grid), i};
        }
        std::sort(keyed.begin(), keyed.end());

        Kokkos::View<Vector*, Kokkos::HostSpace> sorted("sorted", N);
        for (size_t i = 0; i < N; ++i) {
            sorted(i) = pos(keyed[i].second);
        }
        return sorted;
    }

    Results analyze(const Kokkos::View<Vector*, Kokkos::HostSpace>& pos, size_t N, int w, const std::string& ordering) {
        Results r;
        r.ordering = ordering;
        r.window_size = params_.window_size;

        double h = 2.0 * M_PI / params_.n_grid;
        int S = w * w * w;

        constexpr size_t LINE_SIZE = 128;
        constexpr size_t ELEM_SIZE = 16;
        constexpr size_t ELEMS_PER_LINE = LINE_SIZE / ELEM_SIZE;

        auto idx_to_line = [&](size_t idx) { return idx / ELEMS_PER_LINE; };

        // Global statistics
        std::unordered_set<size_t> global_lines;
        r.total_updates = N * S;

        for (size_t i = 0; i < N; ++i) {
            int base[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                base[d] = static_cast<int>(std::floor(pos(i)[d] / h)) - w / 2;
            }
            for (int iz = 0; iz < w; ++iz) {
                for (int iy = 0; iy < w; ++iy) {
                    for (int ix = 0; ix < w; ++ix) {
                        int gx = (base[0] + ix + params_.n_grid) % params_.n_grid;
                        int gy = (base[1] + iy + params_.n_grid) % params_.n_grid;
                        int gz = (base[2] + iz + params_.n_grid) % params_.n_grid;
                        size_t idx = gx + params_.n_grid * (gy + params_.n_grid * gz);
                        global_lines.insert(idx_to_line(idx));
                    }
                }
            }
        }

        r.total_cache_lines = global_lines.size();
        r.phi_global = static_cast<double>(r.total_updates) / r.total_cache_lines;

        // WINDOWED analysis - THE KEY METRIC
        size_t n_windows = (N + params_.window_size - 1) / params_.window_size;
        std::vector<double> m_L_per_window(n_windows);
        size_t sum_lines = 0;

        for (size_t wnd = 0; wnd < n_windows; ++wnd) {
            size_t w_start = wnd * params_.window_size;
            size_t w_end = std::min(w_start + params_.window_size, N);

            std::unordered_set<size_t> window_lines;
            size_t window_updates = 0;

            for (size_t i = w_start; i < w_end; ++i) {
                int base[Dim];
                for (unsigned d = 0; d < Dim; ++d) {
                    base[d] = static_cast<int>(std::floor(pos(i)[d] / h)) - w / 2;
                }
                for (int iz = 0; iz < w; ++iz) {
                    for (int iy = 0; iy < w; ++iy) {
                        for (int ix = 0; ix < w; ++ix) {
                            int gx = (base[0] + ix + params_.n_grid) % params_.n_grid;
                            int gy = (base[1] + iy + params_.n_grid) % params_.n_grid;
                            int gz = (base[2] + iz + params_.n_grid) % params_.n_grid;
                            size_t idx = gx + params_.n_grid * (gy + params_.n_grid * gz);
                            window_lines.insert(idx_to_line(idx));
                            window_updates++;
                        }
                    }
                }
            }

            sum_lines += window_lines.size();
            m_L_per_window[wnd] = (window_lines.size() > 0)
                ? static_cast<double>(window_updates) / window_lines.size()
                : 1.0;
        }

        double sum = 0, max_val = 0;
        for (double x : m_L_per_window) { sum += x; max_val = std::max(max_val, x); }
        r.m_L = sum / n_windows;
        r.m_L_max = max_val;
        r.avg_lines_per_window = sum_lines / n_windows;

        double sq = 0;
        for (double x : m_L_per_window) sq += (x - r.m_L) * (x - r.m_L);
        r.m_L_stddev = (n_windows > 1) ? std::sqrt(sq / (n_windows - 1)) : 0;

        // Per-block (for reference, misses inter-block contention)
        int team_size = 4;
        size_t n_blocks = (N + team_size - 1) / team_size;
        double sum_phi = 0;
        for (size_t b = 0; b < n_blocks; ++b) {
            std::unordered_set<size_t> block_pts;
            size_t b_start = b * team_size;
            size_t b_end = std::min(b_start + team_size, N);
            for (size_t i = b_start; i < b_end; ++i) {
                int base[Dim];
                for (unsigned d = 0; d < Dim; ++d) {
                    base[d] = static_cast<int>(std::floor(pos(i)[d] / h)) - w / 2;
                }
                for (int iz = 0; iz < w; ++iz) {
                    for (int iy = 0; iy < w; ++iy) {
                        for (int ix = 0; ix < w; ++ix) {
                            int gx = (base[0] + ix + params_.n_grid) % params_.n_grid;
                            int gy = (base[1] + iy + params_.n_grid) % params_.n_grid;
                            int gz = (base[2] + iz + params_.n_grid) % params_.n_grid;
                            block_pts.insert(gx + params_.n_grid * (gy + params_.n_grid * gz));
                        }
                    }
                }
            }
            size_t updates = (b_end - b_start) * S;
            sum_phi += (block_pts.size() > 0) ? static_cast<double>(updates) / block_pts.size() : 1.0;
        }
        r.phi_block_mean = sum_phi / n_blocks;

        return r;
    }

    void print_header() {
        if (ippl::Comm->rank() != 0) return;

        ippl::NUFFT::ESKernel<real_type> kernel(params_.kernel_tol);

        std::cout << "\n================================================================\n"
                  << "     CORRECTED Collision Analysis (Windowed m_L)\n"
                  << "================================================================\n"
                  << "Grid: " << params_.n_grid << "^3\n"
                  << "Particles: " << params_.n_particles() << " (rho=" << params_.rho << ")\n"
                  << "Kernel width: " << kernel.width() << "\n"
                  << "Window size: " << params_.window_size << " (P_in_flight)\n"
                  << "Distribution: " << params_.distribution << "\n"
                  << "================================================================\n\n";
    }

    void print_results(const std::vector<Results>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::cout << "=== WINDOWED m_L (for performance model) ===\n\n"
                  << std::left << std::setw(16) << "Ordering"
                  << std::setw(12) << "m_L"
                  << std::setw(12) << "m_L_max"
                  << std::setw(12) << "m_L_std"
                  << std::setw(16) << "Avg lines/wnd"
                  << "\n" << std::string(68, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(16) << r.ordering
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << r.m_L
                      << std::setw(12) << r.m_L_max
                      << std::setw(12) << r.m_L_stddev
                      << std::setw(16) << r.avg_lines_per_window << "\n";
        }

        std::cout << "\n=== Global metrics (for REFERENCE ONLY, NOT for model) ===\n\n"
                  << std::left << std::setw(16) << "Ordering"
                  << std::setw(14) << "phi_global"
                  << std::setw(14) << "phi_block"
                  << std::setw(16) << "Total lines"
                  << "\n" << std::string(60, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(16) << r.ordering
                      << std::fixed << std::setprecision(2)
                      << std::setw(14) << r.phi_global
                      << std::setw(14) << r.phi_block_mean
                      << std::setw(16) << r.total_cache_lines << "\n";
        }

        // Key insight
        if (results.size() >= 2) {
            double m_L_unsorted = results[0].m_L;
            double m_L_sorted = results[1].m_L;
            double phi_unsorted = results[0].phi_global;
            double phi_sorted = results[1].phi_global;

            std::cout << "\n=== KEY INSIGHT ===\n\n";

            std::cout << "phi_global: " << std::fixed << std::setprecision(1)
                      << phi_unsorted << " (unsorted) vs " << phi_sorted << " (sorted)\n"
                      << "  → " << 100 * std::abs(phi_sorted - phi_unsorted) / phi_unsorted << "% change\n"
                      << "  → This does NOT capture temporal contention!\n\n";

            std::cout << "m_L (windowed): " << std::setprecision(2) << m_L_unsorted
                      << " (unsorted) vs " << m_L_sorted << " (sorted)\n";

            double pct_change = 100 * (m_L_sorted - m_L_unsorted) / m_L_unsorted;
            if (m_L_sorted > m_L_unsorted) {
                std::cout << "  → " << std::setprecision(1) << pct_change << "% HIGHER with sorting!\n"
                          << "  → Sorting INCREASES cache-line contention within each wave!\n"
                          << "  → This explains why sorting can make atomics SLOWER.\n";
            } else {
                std::cout << "  → " << -pct_change << "% lower with sorting\n";
            }

            std::cout << "\nFor performance model:\n"
                      << "  A_eff(m_L) = (1/A_L2_unique + (m_L - 1)/A_L2_hot)^(-1)\n"
                      << "  T_atomic = (N * S * eta) / A_eff   [NO WAVES]\n"
                      << "  Use m_L = " << m_L_unsorted << " (unsorted) or " << m_L_sorted << " (sorted)\n";
        }
    }

    void write_csv(const std::vector<Results>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::ofstream out(params_.output);
        out << "ordering,m_L,m_L_max,m_L_stddev,avg_lines_per_window,"
            << "phi_global,phi_block_mean,total_cache_lines,window_size\n";

        for (const auto& r : results) {
            out << r.ordering << ","
                << std::fixed << std::setprecision(4)
                << r.m_L << "," << r.m_L_max << "," << r.m_L_stddev << ","
                << r.avg_lines_per_window << ","
                << r.phi_global << "," << r.phi_block_mean << ","
                << r.total_cache_lines << "," << r.window_size << "\n";
        }
        out.close();
        std::cout << "\nWrote: " << params_.output << "\n";
    }

private:
    Params params_;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        auto params = parse_args(argc, argv);
        CollisionAnalyzer<Kokkos::DefaultExecutionSpace> analyzer(params);
        analyzer.run();
    }
    ippl::finalize();
    return 0;
}