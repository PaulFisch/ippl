/**
 * @file BenchmarkCollisionAnalysis.cpp
 * @brief Analyze collision factor φ for different particle distributions
 *
 * Computes:
 *   - φ_global: Global collision factor (total updates / unique grid points)
 *   - φ_block: Per-block collision factor (affects atomic serialization)
 *   - φ_L: Cache-line collision factor (affects L2 efficiency)
 *
 * Compares:
 *   - Unsorted vs sorted particle orderings
 *   - Different distributions (uniform, clustered, real data)
 *
 * Usage: ./BenchmarkCollisionAnalysis [options]
 *   --grid N         Grid size per dimension (default: 64)
 *   --rho R          Particles per grid point (default: 10)
 *   --tol T          Kernel tolerance (default: 1e-6)
 *   --team-size T    Team size for block analysis (default: 4)
 *   --dist D         Distribution: uniform, clustered (default: uniform)
 *   --output FILE    Output CSV file (default: collision_analysis.csv)
 *   -v, --verbose    Verbose output
 */

#include "Ippl.h"
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <vector>

using namespace ippl;

// ============================================================================
// Parameters
// ============================================================================

struct AnalysisParams {
    int n_grid = 64;
    double rho = 10.0;
    double kernel_tol = 1e-6;
    int team_size = 4;
    std::string distribution = "uniform";
    std::string output = "collision_analysis.csv";
    bool verbose = false;
    bool sweep_rho = false;
    
    size_t n_particles() const {
        return static_cast<size_t>(rho * n_grid * n_grid * n_grid);
    }
};

AnalysisParams parse_args(int argc, char* argv[]) {
    AnalysisParams p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--grid" && i + 1 < argc) {
            p.n_grid = std::atoi(argv[++i]);
        } else if (arg == "--rho" && i + 1 < argc) {
            p.rho = std::atof(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            p.kernel_tol = std::atof(argv[++i]);
        } else if (arg == "--team-size" && i + 1 < argc) {
            p.team_size = std::atoi(argv[++i]);
        } else if (arg == "--dist" && i + 1 < argc) {
            p.distribution = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            p.output = argv[++i];
        } else if (arg == "--sweep-rho") {
            p.sweep_rho = true;
        } else if (arg == "-v" || arg == "--verbose") {
            p.verbose = true;
        }
    }
    return p;
}

// ============================================================================
// Collision Factor Results
// ============================================================================

struct CollisionResults {
    std::string distribution;
    std::string ordering;       // "unsorted" or "sorted"
    int n_grid;
    int kernel_width;
    size_t n_particles;
    double rho;
    int team_size;
    
    // Global statistics
    double phi_global;          // Total updates / unique grid points
    size_t unique_grid_points;
    size_t total_updates;
    
    // Per-block statistics
    double phi_block_mean;
    double phi_block_stddev;
    double phi_block_min;
    double phi_block_max;
    double phi_block_median;
    
    // Cache-line statistics (for memory model)
    double phi_L;               // Grid points / cache lines
    size_t unique_cache_lines;
    
    // Grid point reuse (for understanding collision structure)
    double avg_updates_per_point;
    size_t max_updates_per_point;
};

// ============================================================================
// Morton/Z-order sorting for spatial locality
// ============================================================================

template <typename real_type, unsigned Dim>
struct MortonKey {
    using Vector = ippl::Vector<real_type, Dim>;
    
    static uint64_t compute(const Vector& pos, real_type h, int n_grid) {
        // Convert to integer grid coordinates
        int ix[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            ix[d] = static_cast<int>(pos[d] / h);
            ix[d] = std::max(0, std::min(n_grid - 1, ix[d]));
        }
        
        // Interleave bits for Morton code
        if constexpr (Dim == 3) {
            return interleave3(ix[0], ix[1], ix[2]);
        } else if constexpr (Dim == 2) {
            return interleave2(ix[0], ix[1]);
        } else {
            return static_cast<uint64_t>(ix[0]);
        }
    }
    
private:
    static uint64_t expandBits(uint64_t v) {
        // Spread bits for 3D Morton code (21 bits each)
        v = (v * 0x0001000100010001ULL) & 0xFFFF00000000FFFFULL;
        v = (v * 0x0000000100000001ULL) & 0x00FF0000FF0000FFULL;
        v = (v * 0x0000000000010001ULL) & 0xF00F00F00F00F00FULL;
        v = (v * 0x0000000000000101ULL) & 0x30C30C30C30C30C3ULL;
        v = (v * 0x0000000000000005ULL) & 0x9249249249249249ULL;
        return v;
    }
    
    static uint64_t interleave3(int x, int y, int z) {
        return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
    }
    
    static uint64_t interleave2(int x, int y) {
        uint64_t xi = x, yi = y;
        xi = (xi | (xi << 16)) & 0x0000FFFF0000FFFFULL;
        xi = (xi | (xi << 8)) & 0x00FF00FF00FF00FFULL;
        xi = (xi | (xi << 4)) & 0x0F0F0F0F0F0F0F0FULL;
        xi = (xi | (xi << 2)) & 0x3333333333333333ULL;
        xi = (xi | (xi << 1)) & 0x5555555555555555ULL;
        
        yi = (yi | (yi << 16)) & 0x0000FFFF0000FFFFULL;
        yi = (yi | (yi << 8)) & 0x00FF00FF00FF00FFULL;
        yi = (yi | (yi << 4)) & 0x0F0F0F0F0F0F0F0FULL;
        yi = (yi | (yi << 2)) & 0x3333333333333333ULL;
        yi = (yi | (yi << 1)) & 0x5555555555555555ULL;
        
        return xi | (yi << 1);
    }
};

// ============================================================================
// Collision Analyzer
// ============================================================================

template <typename ExecSpace>
class CollisionAnalyzer {
public:
    static constexpr unsigned Dim = 3;
    using real_type = double;
    using MemSpace = typename ExecSpace::memory_space;
    using Vector = ippl::Vector<real_type, Dim>;
    
    CollisionAnalyzer(const AnalysisParams& params) : params_(params) {}
    
    void run() {
        print_header();
        
        ippl::NUFFT::ESKernel<real_type> kernel(params_.kernel_tol);
        int w = kernel.width();
        
        // Generate particles
        size_t N = params_.n_particles();
        auto positions = generate_particles(N);
        
        std::vector<CollisionResults> results;
        
        // Analyze unsorted
        results.push_back(analyze(positions, N, w, "unsorted"));
        
        // Sort by Morton code
        auto sorted_positions = sort_particles(positions, N);
        results.push_back(analyze(sorted_positions, N, w, "sorted"));
        
        // Output
        print_results(results);
        write_csv(results);
        
        // Sweep rho if requested
        if (params_.sweep_rho) {
            sweep_particle_density(w);
        }
    }
    
    Kokkos::View<Vector*, Kokkos::HostSpace> generate_particles(size_t N) {
        Kokkos::View<Vector*, MemSpace> positions_d("positions", N);
        Kokkos::Random_XorShift64_Pool<> rand_pool(42);
        
        if (params_.distribution == "uniform") {
            Kokkos::parallel_for("init_uniform", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    Vector pos;
                    for (unsigned d = 0; d < Dim; ++d) {
                        pos[d] = gen.drand() * 2.0 * M_PI;
                    }
                    positions_d(i) = pos;
                    rand_pool.free_state(gen);
                });
        } else if (params_.distribution == "clustered") {
            Kokkos::parallel_for("init_clustered", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    Vector pos;
                    for (unsigned d = 0; d < Dim; ++d) {
                        double u1 = gen.drand();
                        double u2 = gen.drand();
                        double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-10))
                                   * Kokkos::cos(2.0 * M_PI * u2);
                        pos[d] = M_PI + 0.3 * z;
                        while (pos[d] < 0) pos[d] += 2.0 * M_PI;
                        while (pos[d] >= 2.0 * M_PI) pos[d] -= 2.0 * M_PI;
                    }
                    positions_d(i) = pos;
                    rand_pool.free_state(gen);
                });
        }
        
        Kokkos::fence();
        
        // Copy to host for analysis
        auto positions = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), positions_d);
        return positions;
    }
    
    Kokkos::View<Vector*, Kokkos::HostSpace> sort_particles(
        const Kokkos::View<Vector*, Kokkos::HostSpace>& positions, size_t N) 
    {
        // Compute Morton keys
        double h = 2.0 * M_PI / params_.n_grid;
        std::vector<std::pair<uint64_t, size_t>> keyed(N);
        
        for (size_t i = 0; i < N; ++i) {
            keyed[i] = {MortonKey<real_type, Dim>::compute(positions(i), h, params_.n_grid), i};
        }
        
        // Sort by key
        std::sort(keyed.begin(), keyed.end());
        
        // Reorder positions
        Kokkos::View<Vector*, Kokkos::HostSpace> sorted("sorted", N);
        for (size_t i = 0; i < N; ++i) {
            sorted(i) = positions(keyed[i].second);
        }
        
        return sorted;
    }
    
    CollisionResults analyze(const Kokkos::View<Vector*, Kokkos::HostSpace>& positions,
                             size_t N, int w, const std::string& ordering) 
    {
        CollisionResults r;
        r.distribution = params_.distribution;
        r.ordering = ordering;
        r.n_grid = params_.n_grid;
        r.kernel_width = w;
        r.n_particles = N;
        r.rho = params_.rho;
        r.team_size = params_.team_size;
        
        double h = 2.0 * M_PI / params_.n_grid;
        int S = w * w * w;  // Updates per particle (3D)
        
        // Global collision analysis
        std::unordered_map<size_t, size_t> grid_point_counts;
        std::unordered_set<size_t> cache_lines;
        
        constexpr size_t CACHE_LINE = 128;
        constexpr size_t ELEM_SIZE = 16;  // sizeof(complex<double>)
        constexpr size_t ELEMS_PER_LINE = CACHE_LINE / ELEM_SIZE;
        
        r.total_updates = 0;
        
        for (size_t i = 0; i < N; ++i) {
            Vector pos = positions(i);
            int base[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                base[d] = static_cast<int>(std::floor(pos[d] / h)) - w / 2;
            }
            
            // Enumerate stencil points
            for (int iz = 0; iz < w; ++iz) {
                for (int iy = 0; iy < w; ++iy) {
                    for (int ix = 0; ix < w; ++ix) {
                        int gx = (base[0] + ix + params_.n_grid) % params_.n_grid;
                        int gy = (base[1] + iy + params_.n_grid) % params_.n_grid;
                        int gz = (base[2] + iz + params_.n_grid) % params_.n_grid;
                        
                        size_t idx = gx + params_.n_grid * (gy + params_.n_grid * gz);
                        grid_point_counts[idx]++;
                        cache_lines.insert(idx / ELEMS_PER_LINE);
                        r.total_updates++;
                    }
                }
            }
        }
        
        r.unique_grid_points = grid_point_counts.size();
        r.unique_cache_lines = cache_lines.size();
        r.phi_global = static_cast<double>(r.total_updates) / r.unique_grid_points;
        r.phi_L = static_cast<double>(r.unique_grid_points) / r.unique_cache_lines;
        
        // Grid point reuse statistics
        size_t max_count = 0;
        double sum_count = 0;
        for (const auto& [idx, count] : grid_point_counts) {
            max_count = std::max(max_count, count);
            sum_count += count;
        }
        r.avg_updates_per_point = sum_count / grid_point_counts.size();
        r.max_updates_per_point = max_count;
        
        // Per-block collision analysis
        size_t n_blocks = (N + params_.team_size - 1) / params_.team_size;
        std::vector<double> phi_blocks(n_blocks);
        
        for (size_t block = 0; block < n_blocks; ++block) {
            std::unordered_set<size_t> block_points;
            size_t block_start = block * params_.team_size;
            size_t block_end = std::min(block_start + params_.team_size, N);
            
            for (size_t i = block_start; i < block_end; ++i) {
                Vector pos = positions(i);
                int base[Dim];
                for (unsigned d = 0; d < Dim; ++d) {
                    base[d] = static_cast<int>(std::floor(pos[d] / h)) - w / 2;
                }
                
                for (int iz = 0; iz < w; ++iz) {
                    for (int iy = 0; iy < w; ++iy) {
                        for (int ix = 0; ix < w; ++ix) {
                            int gx = (base[0] + ix + params_.n_grid) % params_.n_grid;
                            int gy = (base[1] + iy + params_.n_grid) % params_.n_grid;
                            int gz = (base[2] + iz + params_.n_grid) % params_.n_grid;
                            size_t idx = gx + params_.n_grid * (gy + params_.n_grid * gz);
                            block_points.insert(idx);
                        }
                    }
                }
            }
            
            size_t block_updates = (block_end - block_start) * S;
            phi_blocks[block] = static_cast<double>(block_updates) / block_points.size();
        }
        
        // Per-block statistics
        double sum = 0, sq_sum = 0;
        double min_phi = phi_blocks[0], max_phi = phi_blocks[0];
        for (double phi : phi_blocks) {
            sum += phi;
            sq_sum += phi * phi;
            min_phi = std::min(min_phi, phi);
            max_phi = std::max(max_phi, phi);
        }
        
        r.phi_block_mean = sum / n_blocks;
        r.phi_block_stddev = std::sqrt((sq_sum - sum * sum / n_blocks) / (n_blocks - 1));
        r.phi_block_min = min_phi;
        r.phi_block_max = max_phi;
        
        std::sort(phi_blocks.begin(), phi_blocks.end());
        r.phi_block_median = (n_blocks % 2 == 0)
            ? (phi_blocks[n_blocks/2 - 1] + phi_blocks[n_blocks/2]) / 2
            : phi_blocks[n_blocks/2];
        
        return r;
    }
    
    void sweep_particle_density(int w) {
        if (ippl::Comm->rank() != 0) return;
        
        std::cout << "\n=== Particle Density Sweep ===\n\n";
        std::cout << std::left 
                  << std::setw(8) << "rho"
                  << std::setw(12) << "N"
                  << std::setw(10) << "φ_unsort"
                  << std::setw(10) << "φ_sorted"
                  << std::setw(10) << "φ_L_un"
                  << std::setw(10) << "φ_L_sort"
                  << "\n";
        std::cout << std::string(60, '-') << "\n";
        
        for (double rho : {1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0}) {
            AnalysisParams p = params_;
            p.rho = rho;
            size_t N = p.n_particles();
            
            auto positions = generate_particles(N);
            auto unsorted = analyze(positions, N, w, "unsorted");
            
            auto sorted_pos = sort_particles(positions, N);
            auto sorted = analyze(sorted_pos, N, w, "sorted");
            
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(8) << rho
                      << std::setw(12) << N
                      << std::setw(10) << std::setprecision(2) << unsorted.phi_global
                      << std::setw(10) << sorted.phi_global
                      << std::setw(10) << unsorted.phi_L
                      << std::setw(10) << sorted.phi_L
                      << "\n";
        }
    }
    
    void print_header() {
        if (ippl::Comm->rank() != 0) return;
        
        ippl::NUFFT::ESKernel<real_type> kernel(params_.kernel_tol);
        
        std::cout << "\n================================================================\n";
        std::cout << "     Collision Factor Analysis for AtomicScatter\n";
        std::cout << "================================================================\n";
        std::cout << "Grid: " << params_.n_grid << "^3\n";
        std::cout << "Particles: " << params_.n_particles() << " (ρ = " << params_.rho << ")\n";
        std::cout << "Kernel width: " << kernel.width() << "\n";
        std::cout << "Team size: " << params_.team_size << "\n";
        std::cout << "Distribution: " << params_.distribution << "\n";
        std::cout << "================================================================\n\n";
    }
    
    void print_results(const std::vector<CollisionResults>& results) {
        if (ippl::Comm->rank() != 0) return;
        
        std::cout << "=== Global Collision Factors ===\n\n";
        std::cout << std::left << std::setw(12) << "Ordering"
                  << std::setw(10) << "φ_global"
                  << std::setw(10) << "φ_L"
                  << std::setw(14) << "Unique pts"
                  << std::setw(14) << "Cache lines"
                  << "\n";
        std::cout << std::string(60, '-') << "\n";
        
        for (const auto& r : results) {
            std::cout << std::left << std::setw(12) << r.ordering
                      << std::fixed << std::setprecision(3)
                      << std::setw(10) << r.phi_global
                      << std::setw(10) << r.phi_L
                      << std::setw(14) << r.unique_grid_points
                      << std::setw(14) << r.unique_cache_lines
                      << "\n";
        }
        
        std::cout << "\n=== Per-Block Collision Factors (team_size=" << params_.team_size << ") ===\n\n";
        std::cout << std::left << std::setw(12) << "Ordering"
                  << std::setw(10) << "Mean"
                  << std::setw(10) << "Stddev"
                  << std::setw(10) << "Min"
                  << std::setw(10) << "Max"
                  << std::setw(10) << "Median"
                  << "\n";
        std::cout << std::string(62, '-') << "\n";
        
        for (const auto& r : results) {
            std::cout << std::left << std::setw(12) << r.ordering
                      << std::fixed << std::setprecision(3)
                      << std::setw(10) << r.phi_block_mean
                      << std::setw(10) << r.phi_block_stddev
                      << std::setw(10) << r.phi_block_min
                      << std::setw(10) << r.phi_block_max
                      << std::setw(10) << r.phi_block_median
                      << "\n";
        }
        
        std::cout << "\n=== Grid Point Reuse ===\n\n";
        for (const auto& r : results) {
            std::cout << r.ordering << ":\n";
            std::cout << "  Avg updates/point: " << std::fixed << std::setprecision(2) 
                      << r.avg_updates_per_point << "\n";
            std::cout << "  Max updates/point: " << r.max_updates_per_point << "\n";
        }
        
        // Interpretation
        std::cout << "\n=== Interpretation ===\n";
        if (results.size() >= 2) {
            double phi_unsorted = results[0].phi_global;
            double phi_sorted = results[1].phi_global;
            double phi_L_unsorted = results[0].phi_L;
            double phi_L_sorted = results[1].phi_L;
            
            std::cout << "\nSorting effect:\n";
            if (phi_sorted > phi_unsorted) {
                std::cout << "  φ increases by " << std::fixed << std::setprecision(1) 
                          << 100 * (phi_sorted - phi_unsorted) / phi_unsorted << "%"
                          << " → MORE atomic contention\n";
            } else {
                std::cout << "  φ decreases by " << std::fixed << std::setprecision(1) 
                          << 100 * (phi_unsorted - phi_sorted) / phi_unsorted << "%"
                          << " → LESS atomic contention\n";
            }
            
            if (phi_L_sorted < phi_L_unsorted) {
                std::cout << "  φ_L decreases by " << std::fixed << std::setprecision(1)
                          << 100 * (phi_L_unsorted - phi_L_sorted) / phi_L_unsorted << "%"
                          << " → BETTER cache locality\n";
            }
            
            std::cout << "\nFor performance model:\n";
            std::cout << "  Use φ_block_mean = " << results[0].phi_block_mean 
                      << " (unsorted) or " << results[1].phi_block_mean << " (sorted)\n";
            std::cout << "  This is the φ that affects A_eff(φ) = (1/A_1 + (φ-1)/A_∞)^(-1)\n";
        }
    }
    
    void write_csv(const std::vector<CollisionResults>& results) {
        if (ippl::Comm->rank() != 0) return;
        
        std::ofstream out(params_.output);
        out << "distribution,ordering,n_grid,kernel_width,n_particles,rho,team_size,"
            << "phi_global,unique_grid_points,total_updates,"
            << "phi_block_mean,phi_block_stddev,phi_block_min,phi_block_max,phi_block_median,"
            << "phi_L,unique_cache_lines,"
            << "avg_updates_per_point,max_updates_per_point\n";
        
        for (const auto& r : results) {
            out << r.distribution << ","
                << r.ordering << ","
                << r.n_grid << ","
                << r.kernel_width << ","
                << r.n_particles << ","
                << std::fixed << std::setprecision(1) << r.rho << ","
                << r.team_size << ","
                << std::setprecision(4) << r.phi_global << ","
                << r.unique_grid_points << ","
                << r.total_updates << ","
                << r.phi_block_mean << ","
                << r.phi_block_stddev << ","
                << r.phi_block_min << ","
                << r.phi_block_max << ","
                << r.phi_block_median << ","
                << r.phi_L << ","
                << r.unique_cache_lines << ","
                << r.avg_updates_per_point << ","
                << r.max_updates_per_point << "\n";
        }
        
        out.close();
        std::cout << "\nWrote results to: " << params_.output << "\n";
    }
    
private:
    AnalysisParams params_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    
    {
        auto params = parse_args(argc, argv);
        CollisionAnalyzer<Kokkos::DefaultExecutionSpace> analyzer(params);
        analyzer.run();
    }
    
    ippl::finalize();
    return EXIT_SUCCESS;
}
