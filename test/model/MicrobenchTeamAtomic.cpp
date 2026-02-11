/**
 * @file MicrobenchTeamAtomic.cpp
 * @brief Team-based atomic scatter microbenchmark mimicking IPPL AtomicScatter
 *
 * This benchmark replicates the structure of the actual AtomicScatter kernel:
 *   - Team policy with configurable team size
 *   - Shared memory scratch for kernel weights
 *   - ThreadVectorRange for stencil iteration
 *   - Complex atomic adds to grid
 *
 * Measures:
 *   - Phase A: Weight computation (ESKernel-like math)
 *   - Phase B: Atomic scatter to grid
 *   - Barrier overhead
 *   - Full kernel timing
 *
 * NO IPPL dependency - pure Kokkos benchmark.
 *
 * Usage: ./MicrobenchTeamAtomic [options]
 *   --N <size>       Number of particles (default: 1M)
 *   --grid <n>       Grid size per dimension (default: 64)
 *   --W <width>      Kernel width (default: 4)
 *   --team <size>    Team size (default: 4)
 *   --vector <size>  Vector length (default: auto)
 *   --runs <n>       Benchmark iterations (default: 20)
 *   --phase-a-only   Measure only Phase A (no atomics)
 *   --phase-b-only   Measure only Phase B (atomics only)
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>

// ============================================================================
// Timer
// ============================================================================

class GPUTimer {
public:
    void start() {
        Kokkos::fence();
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        Kokkos::fence();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Statistics
// ============================================================================

struct Stats {
    double mean, stddev, min, max, median;
};

Stats compute_stats(const std::vector<double>& v) {
    Stats s{};
    size_t n = v.size();
    if (n == 0) return s;
    
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    
    double sq_sum = 0;
    for (double x : v) sq_sum += (x - s.mean) * (x - s.mean);
    s.stddev = (n > 1) ? std::sqrt(sq_sum / (n - 1)) : 0;
    
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    
    std::vector<double> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    s.median = (n % 2 == 0) ? (sorted[n/2 - 1] + sorted[n/2]) / 2 : sorted[n/2];
    
    return s;
}

// ============================================================================
// Parameters
// ============================================================================

struct Params {
    size_t N = 1000000;
    int n_grid = 64;
    int W = 4;           // Kernel width
    int team_size = 4;
    int vector_length = 0;  // 0 = auto
    int runs = 20;
    int warmup = 5;
    bool phase_a_only = false;
    bool phase_b_only = false;
    std::string output = "team_atomic.csv";
    bool verbose = false;
};

Params parse_args(int argc, char* argv[]) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) {
            p.N = std::stoull(argv[++i]);
        } else if (arg == "--grid" && i + 1 < argc) {
            p.n_grid = std::atoi(argv[++i]);
        } else if (arg == "--W" && i + 1 < argc) {
            p.W = std::atoi(argv[++i]);
        } else if (arg == "--team" && i + 1 < argc) {
            p.team_size = std::atoi(argv[++i]);
        } else if (arg == "--vector" && i + 1 < argc) {
            p.vector_length = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            p.runs = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            p.output = argv[++i];
        } else if (arg == "--phase-a-only") {
            p.phase_a_only = true;
        } else if (arg == "--phase-b-only") {
            p.phase_b_only = true;
        } else if (arg == "-v" || arg == "--verbose") {
            p.verbose = true;
        }
    }
    return p;
}

// ============================================================================
// ESKernel-like weight computation
// ============================================================================

KOKKOS_INLINE_FUNCTION
double es_kernel_1d(double x, double beta, double c) {
    // Simplified ES kernel: exp(beta * (sqrt(1 - x^2) - 1))
    double t = 1.0 - x * x;
    if (t <= 0) return 0.0;
    return Kokkos::exp(beta * (Kokkos::sqrt(t) - 1.0)) * c;
}

// ============================================================================
// Team-based Scatter Kernel
// ============================================================================

template <typename ExecSpace>
class TeamAtomicBenchmark {
public:
    using MemSpace = typename ExecSpace::memory_space;
    using complex_type = Kokkos::complex<double>;
    using ScratchView = Kokkos::View<double*, typename ExecSpace::scratch_memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    
    TeamAtomicBenchmark(const Params& params) : params_(params) {
        // Calculate vector length if not specified
        if (params_.vector_length == 0) {
            params_.vector_length = params_.W * params_.W * params_.W;
        }
        
        // ESKernel parameters (simplified)
        beta_ = 2.3 * params_.W;
        c_ = 1.0 / (2.0 * M_PI);
        h_ = 2.0 * M_PI / params_.n_grid;
    }
    
    void run() {
        print_header();
        
        // Allocate data
        size_t grid_size = params_.n_grid * params_.n_grid * params_.n_grid;
        positions_ = Kokkos::View<double*[3], MemSpace>("positions", params_.N);
        values_ = Kokkos::View<complex_type*, MemSpace>("values", params_.N);
        grid_ = Kokkos::View<complex_type*, MemSpace>("grid", grid_size);
        
        // Initialize
        initialize();
        
        // Run benchmarks
        Results results;
        
        if (!params_.phase_b_only) {
            results.phase_a_ms = benchmark_phase_a();
        }
        
        if (!params_.phase_a_only) {
            results.phase_b_ms = benchmark_phase_b();
        }
        
        results.full_kernel_ms = benchmark_full_kernel();
        results.with_barrier_ms = benchmark_with_barrier();
        
        // Print results
        print_results(results);
        write_csv(results);
    }
    
    void initialize() {
        auto pos = positions_;
        auto vals = values_;
        int n_grid = params_.n_grid;
        
        Kokkos::Random_XorShift64_Pool<> pool(42);
        
        Kokkos::parallel_for("init", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
            KOKKOS_LAMBDA(const size_t i) {
                auto gen = pool.get_state();
                for (int d = 0; d < 3; ++d) {
                    pos(i, d) = gen.drand() * 2.0 * M_PI;
                }
                vals(i) = complex_type(1.0, 0.0);
                pool.free_state(gen);
            });
        
        Kokkos::deep_copy(grid_, complex_type(0.0, 0.0));
        Kokkos::fence();
    }
    
    double benchmark_phase_a() {
        // Phase A only: compute weights, no atomics
        
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
        using Member = typename TeamPolicy::member_type;
        
        int W = params_.W;
        int S = W * W * W;
        size_t scratch_size = ScratchView::shmem_size(3 * W);  // 1D weights for each dim
        
        auto pos = positions_;
        double beta = beta_;
        double c = c_;
        double h = h_;
        int n_grid = params_.n_grid;
        
        auto policy = TeamPolicy(params_.N, params_.team_size, params_.vector_length)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_size));
        
        GPUTimer timer;
        std::vector<double> times;
        
        // Dummy output for Phase A
        Kokkos::View<double*, MemSpace> dummy("dummy", params_.N);
        
        for (int i = 0; i < params_.warmup; ++i) {
            Kokkos::parallel_for("warmup_A", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    size_t p = team.league_rank();
                    
                    // Shared memory for 1D weights
                    ScratchView weights_x(team.team_scratch(0), W);
                    ScratchView weights_y(team.team_scratch(0), W);
                    ScratchView weights_z(team.team_scratch(0), W);
                    
                    // Phase A: compute kernel weights
                    double x = pos(p, 0);
                    double y = pos(p, 1);
                    double z = pos(p, 2);
                    
                    // Grid indices
                    int ix0 = static_cast<int>(Kokkos::floor(x / h)) - W / 2;
                    int iy0 = static_cast<int>(Kokkos::floor(y / h)) - W / 2;
                    int iz0 = static_cast<int>(Kokkos::floor(z / h)) - W / 2;
                    
                    // Compute 1D weights (team cooperates)
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W), [&](int j) {
                        double xj = ((ix0 + j + 0.5) * h - x) / h;
                        double yj = ((iy0 + j + 0.5) * h - y) / h;
                        double zj = ((iz0 + j + 0.5) * h - z) / h;
                        weights_x(j) = es_kernel_1d(xj * 2.0 / W, beta, c);
                        weights_y(j) = es_kernel_1d(yj * 2.0 / W, beta, c);
                        weights_z(j) = es_kernel_1d(zj * 2.0 / W, beta, c);
                    });
                    
                    team.team_barrier();
                    
                    // Just sum weights to prevent optimization
                    double sum = 0;
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, S), [&](int s, double& local) {
                        int ix = s % W;
                        int iy = (s / W) % W;
                        int iz = s / (W * W);
                        local += weights_x(ix) * weights_y(iy) * weights_z(iz);
                    }, sum);
                    
                    if (team.team_rank() == 0) {
                        dummy(p) = sum;
                    }
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench_A", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    size_t p = team.league_rank();
                    
                    ScratchView weights_x(team.team_scratch(0), W);
                    ScratchView weights_y(team.team_scratch(0), W);
                    ScratchView weights_z(team.team_scratch(0), W);
                    
                    double x = pos(p, 0);
                    double y = pos(p, 1);
                    double z = pos(p, 2);
                    
                    int ix0 = static_cast<int>(Kokkos::floor(x / h)) - W / 2;
                    int iy0 = static_cast<int>(Kokkos::floor(y / h)) - W / 2;
                    int iz0 = static_cast<int>(Kokkos::floor(z / h)) - W / 2;
                    
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W), [&](int j) {
                        double xj = ((ix0 + j + 0.5) * h - x) / h;
                        double yj = ((iy0 + j + 0.5) * h - y) / h;
                        double zj = ((iz0 + j + 0.5) * h - z) / h;
                        weights_x(j) = es_kernel_1d(xj * 2.0 / W, beta, c);
                        weights_y(j) = es_kernel_1d(yj * 2.0 / W, beta, c);
                        weights_z(j) = es_kernel_1d(zj * 2.0 / W, beta, c);
                    });
                    
                    team.team_barrier();
                    
                    double sum = 0;
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, S), [&](int s, double& local) {
                        int ix = s % W;
                        int iy = (s / W) % W;
                        int iz = s / (W * W);
                        local += weights_x(ix) * weights_y(iy) * weights_z(iz);
                    }, sum);
                    
                    if (team.team_rank() == 0) {
                        dummy(p) = sum;
                    }
                });
            times.push_back(timer.stop());
        }
        
        return compute_stats(times).median * 1000.0;
    }
    
    double benchmark_phase_b() {
        // Phase B only: atomics with precomputed weights
        
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
        using Member = typename TeamPolicy::member_type;
        
        int W = params_.W;
        int S = W * W * W;
        
        auto grid = grid_;
        auto pos = positions_;
        auto vals = values_;
        double h = h_;
        int n_grid = params_.n_grid;
        
        auto policy = TeamPolicy(params_.N, params_.team_size, params_.vector_length);
        
        GPUTimer timer;
        std::vector<double> times;
        
        for (int i = 0; i < params_.warmup; ++i) {
            Kokkos::deep_copy(grid_, complex_type(0.0, 0.0));
            Kokkos::parallel_for("warmup_B", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    size_t p = team.league_rank();
                    complex_type val = vals(p);
                    
                    double x = pos(p, 0);
                    double y = pos(p, 1);
                    double z = pos(p, 2);
                    
                    int ix0 = static_cast<int>(Kokkos::floor(x / h)) - W / 2;
                    int iy0 = static_cast<int>(Kokkos::floor(y / h)) - W / 2;
                    int iz0 = static_cast<int>(Kokkos::floor(z / h)) - W / 2;
                    
                    // All threads scatter with weight = 1 (simplified)
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, S), [&](int s) {
                        int ix = ((ix0 + s % W) % n_grid + n_grid) % n_grid;
                        int iy = ((iy0 + (s / W) % W) % n_grid + n_grid) % n_grid;
                        int iz = ((iz0 + s / (W * W)) % n_grid + n_grid) % n_grid;
                        size_t idx = ix + n_grid * (iy + n_grid * iz);
                        
                        Kokkos::atomic_add(&grid(idx).real(), val.real());
                        Kokkos::atomic_add(&grid(idx).imag(), val.imag());
                    });
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < params_.runs; ++run) {
            Kokkos::deep_copy(grid_, complex_type(0.0, 0.0));
            Kokkos::fence();
            
            timer.start();
            Kokkos::parallel_for("bench_B", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    size_t p = team.league_rank();
                    complex_type val = vals(p);
                    
                    double x = pos(p, 0);
                    double y = pos(p, 1);
                    double z = pos(p, 2);
                    
                    int ix0 = static_cast<int>(Kokkos::floor(x / h)) - W / 2;
                    int iy0 = static_cast<int>(Kokkos::floor(y / h)) - W / 2;
                    int iz0 = static_cast<int>(Kokkos::floor(z / h)) - W / 2;
                    
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, S), [&](int s) {
                        int ix = ((ix0 + s % W) % n_grid + n_grid) % n_grid;
                        int iy = ((iy0 + (s / W) % W) % n_grid + n_grid) % n_grid;
                        int iz = ((iz0 + s / (W * W)) % n_grid + n_grid) % n_grid;
                        size_t idx = ix + n_grid * (iy + n_grid * iz);
                        
                        Kokkos::atomic_add(&grid(idx).real(), val.real());
                        Kokkos::atomic_add(&grid(idx).imag(), val.imag());
                    });
                });
            times.push_back(timer.stop());
        }
        
        return compute_stats(times).median * 1000.0;
    }
    
    double benchmark_full_kernel() {
        // Full kernel: Phase A + Phase B
        
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
        using Member = typename TeamPolicy::member_type;
        
        int W = params_.W;
        int S = W * W * W;
        size_t scratch_size = ScratchView::shmem_size(3 * W);
        
        auto grid = grid_;
        auto pos = positions_;
        auto vals = values_;
        double beta = beta_;
        double c = c_;
        double h = h_;
        int n_grid = params_.n_grid;
        
        auto policy = TeamPolicy(params_.N, params_.team_size, params_.vector_length)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_size));
        
        GPUTimer timer;
        std::vector<double> times;
        
        for (int i = 0; i < params_.warmup; ++i) {
            Kokkos::deep_copy(grid_, complex_type(0.0, 0.0));
            Kokkos::parallel_for("warmup_full", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    size_t p = team.league_rank();
                    complex_type val = vals(p);
                    
                    ScratchView weights_x(team.team_scratch(0), W);
                    ScratchView weights_y(team.team_scratch(0), W);
                    ScratchView weights_z(team.team_scratch(0), W);
                    
                    double x = pos(p, 0);
                    double y = pos(p, 1);
                    double z = pos(p, 2);
                    
                    int ix0 = static_cast<int>(Kokkos::floor(x / h)) - W / 2;
                    int iy0 = static_cast<int>(Kokkos::floor(y / h)) - W / 2;
                    int iz0 = static_cast<int>(Kokkos::floor(z / h)) - W / 2;
                    
                    // Phase A
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W), [&](int j) {
                        double xj = ((ix0 + j + 0.5) * h - x) / h;
                        double yj = ((iy0 + j + 0.5) * h - y) / h;
                        double zj = ((iz0 + j + 0.5) * h - z) / h;
                        weights_x(j) = es_kernel_1d(xj * 2.0 / W, beta, c);
                        weights_y(j) = es_kernel_1d(yj * 2.0 / W, beta, c);
                        weights_z(j) = es_kernel_1d(zj * 2.0 / W, beta, c);
                    });
                    
                    team.team_barrier();
                    
                    // Phase B
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, S), [&](int s) {
                        int lx = s % W;
                        int ly = (s / W) % W;
                        int lz = s / (W * W);
                        
                        double w = weights_x(lx) * weights_y(ly) * weights_z(lz);
                        
                        int ix = ((ix0 + lx) % n_grid + n_grid) % n_grid;
                        int iy = ((iy0 + ly) % n_grid + n_grid) % n_grid;
                        int iz = ((iz0 + lz) % n_grid + n_grid) % n_grid;
                        size_t idx = ix + n_grid * (iy + n_grid * iz);
                        
                        Kokkos::atomic_add(&grid(idx).real(), w * val.real());
                        Kokkos::atomic_add(&grid(idx).imag(), w * val.imag());
                    });
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < params_.runs; ++run) {
            Kokkos::deep_copy(grid_, complex_type(0.0, 0.0));
            Kokkos::fence();
            
            timer.start();
            Kokkos::parallel_for("bench_full", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    size_t p = team.league_rank();
                    complex_type val = vals(p);
                    
                    ScratchView weights_x(team.team_scratch(0), W);
                    ScratchView weights_y(team.team_scratch(0), W);
                    ScratchView weights_z(team.team_scratch(0), W);
                    
                    double x = pos(p, 0);
                    double y = pos(p, 1);
                    double z = pos(p, 2);
                    
                    int ix0 = static_cast<int>(Kokkos::floor(x / h)) - W / 2;
                    int iy0 = static_cast<int>(Kokkos::floor(y / h)) - W / 2;
                    int iz0 = static_cast<int>(Kokkos::floor(z / h)) - W / 2;
                    
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W), [&](int j) {
                        double xj = ((ix0 + j + 0.5) * h - x) / h;
                        double yj = ((iy0 + j + 0.5) * h - y) / h;
                        double zj = ((iz0 + j + 0.5) * h - z) / h;
                        weights_x(j) = es_kernel_1d(xj * 2.0 / W, beta, c);
                        weights_y(j) = es_kernel_1d(yj * 2.0 / W, beta, c);
                        weights_z(j) = es_kernel_1d(zj * 2.0 / W, beta, c);
                    });
                    
                    team.team_barrier();
                    
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, S), [&](int s) {
                        int lx = s % W;
                        int ly = (s / W) % W;
                        int lz = s / (W * W);
                        
                        double w = weights_x(lx) * weights_y(ly) * weights_z(lz);
                        
                        int ix = ((ix0 + lx) % n_grid + n_grid) % n_grid;
                        int iy = ((iy0 + ly) % n_grid + n_grid) % n_grid;
                        int iz = ((iz0 + lz) % n_grid + n_grid) % n_grid;
                        size_t idx = ix + n_grid * (iy + n_grid * iz);
                        
                        Kokkos::atomic_add(&grid(idx).real(), w * val.real());
                        Kokkos::atomic_add(&grid(idx).imag(), w * val.imag());
                    });
                });
            times.push_back(timer.stop());
        }
        
        return compute_stats(times).median * 1000.0;
    }
    
    double benchmark_with_barrier() {
        // Same as full kernel but with extra barriers (to measure barrier cost)
        
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
        using Member = typename TeamPolicy::member_type;
        
        int W = params_.W;
        int S = W * W * W;
        size_t scratch_size = ScratchView::shmem_size(3 * W);
        
        auto grid = grid_;
        auto pos = positions_;
        auto vals = values_;
        double beta = beta_;
        double c = c_;
        double h = h_;
        int n_grid = params_.n_grid;
        
        auto policy = TeamPolicy(params_.N, params_.team_size, params_.vector_length)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_size));
        
        GPUTimer timer;
        std::vector<double> times;
        
        for (int run = 0; run < params_.runs; ++run) {
            Kokkos::deep_copy(grid_, complex_type(0.0, 0.0));
            Kokkos::fence();
            
            timer.start();
            Kokkos::parallel_for("bench_barrier", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    size_t p = team.league_rank();
                    complex_type val = vals(p);
                    
                    ScratchView weights_x(team.team_scratch(0), W);
                    ScratchView weights_y(team.team_scratch(0), W);
                    ScratchView weights_z(team.team_scratch(0), W);
                    
                    double x = pos(p, 0);
                    double y = pos(p, 1);
                    double z = pos(p, 2);
                    
                    int ix0 = static_cast<int>(Kokkos::floor(x / h)) - W / 2;
                    int iy0 = static_cast<int>(Kokkos::floor(y / h)) - W / 2;
                    int iz0 = static_cast<int>(Kokkos::floor(z / h)) - W / 2;
                    
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W), [&](int j) {
                        double xj = ((ix0 + j + 0.5) * h - x) / h;
                        double yj = ((iy0 + j + 0.5) * h - y) / h;
                        double zj = ((iz0 + j + 0.5) * h - z) / h;
                        weights_x(j) = es_kernel_1d(xj * 2.0 / W, beta, c);
                        weights_y(j) = es_kernel_1d(yj * 2.0 / W, beta, c);
                        weights_z(j) = es_kernel_1d(zj * 2.0 / W, beta, c);
                    });
                    
                    team.team_barrier();
                    team.team_barrier();  // Extra barrier 1
                    team.team_barrier();  // Extra barrier 2
                    
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, S), [&](int s) {
                        int lx = s % W;
                        int ly = (s / W) % W;
                        int lz = s / (W * W);
                        
                        double w = weights_x(lx) * weights_y(ly) * weights_z(lz);
                        
                        int ix = ((ix0 + lx) % n_grid + n_grid) % n_grid;
                        int iy = ((iy0 + ly) % n_grid + n_grid) % n_grid;
                        int iz = ((iz0 + lz) % n_grid + n_grid) % n_grid;
                        size_t idx = ix + n_grid * (iy + n_grid * iz);
                        
                        Kokkos::atomic_add(&grid(idx).real(), w * val.real());
                        Kokkos::atomic_add(&grid(idx).imag(), w * val.imag());
                    });
                });
            times.push_back(timer.stop());
        }
        
        return compute_stats(times).median * 1000.0;
    }
    
    struct Results {
        double phase_a_ms = 0;
        double phase_b_ms = 0;
        double full_kernel_ms = 0;
        double with_barrier_ms = 0;
    };
    
    void print_header() {
        std::cout << "\n================================================================\n";
        std::cout << "     Team-Based Atomic Scatter Microbenchmark\n";
        std::cout << "================================================================\n";
        std::cout << "N:           " << params_.N << " particles\n";
        std::cout << "Grid:        " << params_.n_grid << "^3\n";
        std::cout << "W:           " << params_.W << " (S = " << params_.W * params_.W * params_.W << ")\n";
        std::cout << "Team size:   " << params_.team_size << "\n";
        std::cout << "Vector len:  " << params_.vector_length << "\n";
        std::cout << "Runs:        " << params_.runs << "\n";
        std::cout << "================================================================\n\n";
    }
    
    void print_results(const Results& r) {
        int S = params_.W * params_.W * params_.W;
        double total_atomics = 2.0 * params_.N * S;  // 2 per complex
        
        std::cout << "=== Results ===\n\n";
        
        if (r.phase_a_ms > 0) {
            double ns_per_particle_a = r.phase_a_ms * 1e6 / params_.N;
            std::cout << "Phase A (weights):     " << std::fixed << std::setprecision(3) 
                      << r.phase_a_ms << " ms (" << ns_per_particle_a << " ns/particle)\n";
        }
        
        if (r.phase_b_ms > 0) {
            double ns_per_particle_b = r.phase_b_ms * 1e6 / params_.N;
            double atomics_per_sec = total_atomics / (r.phase_b_ms * 1e-3);
            std::cout << "Phase B (atomics):     " << std::fixed << std::setprecision(3)
                      << r.phase_b_ms << " ms (" << ns_per_particle_b << " ns/particle)\n";
            std::cout << "  Atomic rate:         " << std::scientific << atomics_per_sec << " /sec\n";
        }
        
        double ns_per_particle_full = r.full_kernel_ms * 1e6 / params_.N;
        double mpts = params_.N / (r.full_kernel_ms * 1e-3) / 1e6;
        std::cout << "Full kernel:           " << std::fixed << std::setprecision(3)
                  << r.full_kernel_ms << " ms (" << ns_per_particle_full << " ns/particle)\n";
        std::cout << "  Throughput:          " << std::setprecision(1) << mpts << " Mpts/s\n";
        
        double barrier_overhead = r.with_barrier_ms - r.full_kernel_ms;
        double barrier_per_particle = barrier_overhead * 1e6 / params_.N / 2;  // 2 extra barriers
        std::cout << "Barrier overhead:      " << std::fixed << std::setprecision(3)
                  << barrier_overhead << " ms (2 extra barriers)\n";
        std::cout << "  Per barrier:         " << std::setprecision(2) << barrier_per_particle << " ns/particle\n";
        
        if (r.phase_a_ms > 0 && r.phase_b_ms > 0) {
            std::cout << "\nPhase breakdown:\n";
            std::cout << "  Phase A / Total:     " << std::fixed << std::setprecision(1)
                      << 100 * r.phase_a_ms / r.full_kernel_ms << "%\n";
            std::cout << "  Phase B / Total:     " 
                      << 100 * r.phase_b_ms / r.full_kernel_ms << "%\n";
            std::cout << "  Sum / Total:         "
                      << 100 * (r.phase_a_ms + r.phase_b_ms) / r.full_kernel_ms << "%\n";
        }
    }
    
    void write_csv(const Results& r) {
        std::ofstream out(params_.output);
        out << "N,n_grid,W,S,team_size,vector_length,";
        out << "phase_a_ms,phase_b_ms,full_kernel_ms,with_barrier_ms,";
        out << "ns_per_particle,mpts_per_sec,barrier_ns_per_particle\n";
        
        int S = params_.W * params_.W * params_.W;
        double ns_per = r.full_kernel_ms * 1e6 / params_.N;
        double mpts = params_.N / (r.full_kernel_ms * 1e-3) / 1e6;
        double barrier_ns = (r.with_barrier_ms - r.full_kernel_ms) * 1e6 / params_.N / 2;
        
        out << params_.N << "," << params_.n_grid << "," << params_.W << "," << S << ",";
        out << params_.team_size << "," << params_.vector_length << ",";
        out << std::fixed << std::setprecision(4);
        out << r.phase_a_ms << "," << r.phase_b_ms << "," << r.full_kernel_ms << "," << r.with_barrier_ms << ",";
        out << ns_per << "," << mpts << "," << barrier_ns << "\n";
        
        out.close();
        std::cout << "\nWrote results to: " << params_.output << "\n";
    }
    
private:
    Params params_;
    double beta_, c_, h_;
    
    Kokkos::View<double*[3], MemSpace> positions_;
    Kokkos::View<complex_type*, MemSpace> values_;
    Kokkos::View<complex_type*, MemSpace> grid_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    
    {
        Params params = parse_args(argc, argv);
        TeamAtomicBenchmark<Kokkos::DefaultExecutionSpace> benchmark(params);
        benchmark.run();
    }
    
    Kokkos::finalize();
    return 0;
}
