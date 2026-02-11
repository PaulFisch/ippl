/**
 * @file MicrobenchAtomicVariable.cpp
 * @brief Standalone microbenchmark for atomic throughput vs collision factor
 *
 * This benchmark measures A_eff(φ) - the effective atomic throughput as a
 * function of collision factor φ. It validates the interpolation formula:
 *
 *   A_eff(φ) = (1/A_1 + (φ-1)/A_∞)^(-1)
 *
 * where A_1 is the rate with unique addresses and A_∞ is the rate at full contention.
 *
 * NO IPPL dependency - pure Kokkos benchmark.
 *
 * Usage: ./MicrobenchAtomicVariable [options]
 *   --N <size>       Number of atomic operations (default: 10M)
 *   --runs <n>       Benchmark iterations (default: 20)
 *   --phi <list>     Collision factors (default: 1,2,4,8,16,32,64)
 *   --pattern <p>    Collision pattern: strided, random, blocked (default: strided)
 *   --double         Use double precision (default)
 *   --float          Use single precision
 *   --complex        Use complex<double>
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
#include <sstream>

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
// Benchmark Parameters
// ============================================================================

struct Params {
    size_t N = 10000000;
    int runs = 20;
    int warmup = 5;
    std::vector<int> phi_values = {1, 2, 4, 8, 16, 32, 64};
    std::string pattern = "strided";  // strided, random, blocked
    std::string precision = "double"; // double, float, complex
    std::string output = "atomic_variable.csv";
    bool verbose = false;
};

Params parse_args(int argc, char* argv[]) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) {
            p.N = std::stoull(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            p.runs = std::atoi(argv[++i]);
        } else if (arg == "--phi" && i + 1 < argc) {
            p.phi_values.clear();
            std::stringstream ss(argv[++i]);
            std::string val;
            while (std::getline(ss, val, ',')) {
                p.phi_values.push_back(std::stoi(val));
            }
        } else if (arg == "--pattern" && i + 1 < argc) {
            p.pattern = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            p.output = argv[++i];
        } else if (arg == "--double") {
            p.precision = "double";
        } else if (arg == "--float") {
            p.precision = "float";
        } else if (arg == "--complex") {
            p.precision = "complex";
        } else if (arg == "-v" || arg == "--verbose") {
            p.verbose = true;
        }
    }
    return p;
}

// ============================================================================
// Benchmark Kernels
// ============================================================================

template <typename T, typename ExecSpace>
struct AtomicBenchmark {
    using MemSpace = typename ExecSpace::memory_space;
    
    // Measure A_1: unique addresses
    static double measure_A1(size_t N, int warmup, int runs) {
        Kokkos::View<T*, MemSpace> data("data", N);
        Kokkos::deep_copy(data, T(0));
        
        GPUTimer timer;
        std::vector<double> rates;
        
        for (int i = 0; i < warmup; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    Kokkos::atomic_add(&data(i), T(1));
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < runs; ++run) {
            timer.start();
            Kokkos::parallel_for("A1", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    Kokkos::atomic_add(&data(i), T(1));
                });
            double t = timer.stop();
            rates.push_back(N / t);
        }
        
        return compute_stats(rates).median;
    }
    
    // Measure A_∞: full contention
    static double measure_Ainf(size_t N, int warmup, int runs) {
        Kokkos::View<T*, MemSpace> data("data", 1);
        Kokkos::deep_copy(data, T(0));
        
        GPUTimer timer;
        std::vector<double> rates;
        
        for (int i = 0; i < warmup; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t) {
                    Kokkos::atomic_add(&data(0), T(1));
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < runs; ++run) {
            timer.start();
            Kokkos::parallel_for("Ainf", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t) {
                    Kokkos::atomic_add(&data(0), T(1));
                });
            double t = timer.stop();
            rates.push_back(N / t);
        }
        
        return compute_stats(rates).median;
    }
    
    // Measure A_eff with strided pattern (thread i writes to i/phi)
    static double measure_Aeff_strided(size_t N, int phi, int warmup, int runs) {
        size_t n_unique = N / phi;
        if (n_unique == 0) n_unique = 1;
        
        Kokkos::View<T*, MemSpace> data("data", n_unique);
        Kokkos::deep_copy(data, T(0));
        
        GPUTimer timer;
        std::vector<double> rates;
        
        for (int i = 0; i < warmup; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    size_t idx = (i / phi) % n_unique;
                    Kokkos::atomic_add(&data(idx), T(1));
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < runs; ++run) {
            timer.start();
            Kokkos::parallel_for("Aeff", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    size_t idx = (i / phi) % n_unique;
                    Kokkos::atomic_add(&data(idx), T(1));
                });
            double t = timer.stop();
            rates.push_back(N / t);
        }
        
        return compute_stats(rates).median;
    }
    
    // Measure A_eff with random pattern
    static double measure_Aeff_random(size_t N, int phi, int warmup, int runs) {
        size_t n_unique = N / phi;
        if (n_unique == 0) n_unique = 1;
        
        Kokkos::View<T*, MemSpace> data("data", n_unique);
        Kokkos::View<size_t*, MemSpace> indices("indices", N);
        Kokkos::deep_copy(data, T(0));
        
        // Generate random indices
        Kokkos::Random_XorShift64_Pool<> pool(42);
        Kokkos::parallel_for("init_indices", Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(const size_t i) {
                auto gen = pool.get_state();
                indices(i) = gen.urand64() % n_unique;
                pool.free_state(gen);
            });
        Kokkos::fence();
        
        GPUTimer timer;
        std::vector<double> rates;
        
        for (int i = 0; i < warmup; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    Kokkos::atomic_add(&data(indices(i)), T(1));
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < runs; ++run) {
            timer.start();
            Kokkos::parallel_for("Aeff", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    Kokkos::atomic_add(&data(indices(i)), T(1));
                });
            double t = timer.stop();
            rates.push_back(N / t);
        }
        
        return compute_stats(rates).median;
    }
    
    // Measure A_eff with blocked pattern (threads in same warp hit same addresses)
    static double measure_Aeff_blocked(size_t N, int phi, int warmup, int runs) {
        size_t n_unique = N / phi;
        if (n_unique == 0) n_unique = 1;
        
        Kokkos::View<T*, MemSpace> data("data", n_unique);
        Kokkos::deep_copy(data, T(0));
        
        // Block size matches warp size for maximum contention
        int block_size = 32;
        
        GPUTimer timer;
        std::vector<double> rates;
        
        for (int i = 0; i < warmup; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    // Threads in same block hit same address
                    size_t block = i / block_size;
                    size_t idx = block % n_unique;
                    Kokkos::atomic_add(&data(idx), T(1));
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < runs; ++run) {
            timer.start();
            Kokkos::parallel_for("Aeff", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    size_t block = i / block_size;
                    size_t idx = block % n_unique;
                    Kokkos::atomic_add(&data(idx), T(1));
                });
            double t = timer.stop();
            rates.push_back(N / t);
        }
        
        return compute_stats(rates).median;
    }
};

// ============================================================================
// Complex benchmark specialization
// ============================================================================

template <typename ExecSpace>
struct AtomicBenchmark<Kokkos::complex<double>, ExecSpace> {
    using T = Kokkos::complex<double>;
    using MemSpace = typename ExecSpace::memory_space;
    
    static double measure_A1(size_t N, int warmup, int runs) {
        Kokkos::View<T*, MemSpace> data("data", N);
        
        GPUTimer timer;
        std::vector<double> rates;
        
        for (int i = 0; i < warmup; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    Kokkos::atomic_add(&data(i).real(), 1.0);
                    Kokkos::atomic_add(&data(i).imag(), 1.0);
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < runs; ++run) {
            timer.start();
            Kokkos::parallel_for("A1", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    Kokkos::atomic_add(&data(i).real(), 1.0);
                    Kokkos::atomic_add(&data(i).imag(), 1.0);
                });
            double t = timer.stop();
            rates.push_back(2.0 * N / t);  // 2 atomics per element
        }
        
        return compute_stats(rates).median;
    }
    
    static double measure_Ainf(size_t N, int warmup, int runs) {
        Kokkos::View<T*, MemSpace> data("data", 1);
        
        GPUTimer timer;
        std::vector<double> rates;
        
        for (int i = 0; i < warmup; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t) {
                    Kokkos::atomic_add(&data(0).real(), 1.0);
                    Kokkos::atomic_add(&data(0).imag(), 1.0);
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < runs; ++run) {
            timer.start();
            Kokkos::parallel_for("Ainf", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t) {
                    Kokkos::atomic_add(&data(0).real(), 1.0);
                    Kokkos::atomic_add(&data(0).imag(), 1.0);
                });
            double t = timer.stop();
            rates.push_back(2.0 * N / t);
        }
        
        return compute_stats(rates).median;
    }
    
    static double measure_Aeff_strided(size_t N, int phi, int warmup, int runs) {
        size_t n_unique = N / phi;
        if (n_unique == 0) n_unique = 1;
        
        Kokkos::View<T*, MemSpace> data("data", n_unique);
        
        GPUTimer timer;
        std::vector<double> rates;
        
        for (int i = 0; i < warmup; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    size_t idx = (i / phi) % n_unique;
                    Kokkos::atomic_add(&data(idx).real(), 1.0);
                    Kokkos::atomic_add(&data(idx).imag(), 1.0);
                });
            Kokkos::fence();
        }
        
        for (int run = 0; run < runs; ++run) {
            timer.start();
            Kokkos::parallel_for("Aeff", Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(const size_t i) {
                    size_t idx = (i / phi) % n_unique;
                    Kokkos::atomic_add(&data(idx).real(), 1.0);
                    Kokkos::atomic_add(&data(idx).imag(), 1.0);
                });
            double t = timer.stop();
            rates.push_back(2.0 * N / t);
        }
        
        return compute_stats(rates).median;
    }
    
    // Simplified implementations for other patterns
    static double measure_Aeff_random(size_t N, int phi, int warmup, int runs) {
        return measure_Aeff_strided(N, phi, warmup, runs);  // Simplified
    }
    
    static double measure_Aeff_blocked(size_t N, int phi, int warmup, int runs) {
        return measure_Aeff_strided(N, phi, warmup, runs);  // Simplified
    }
};

// ============================================================================
// Main
// ============================================================================

template <typename T>
void run_benchmark(const Params& params) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using Bench = AtomicBenchmark<T, ExecSpace>;
    
    std::cout << "\n================================================================\n";
    std::cout << "     Atomic Throughput vs Collision Factor\n";
    std::cout << "================================================================\n";
    std::cout << "Precision: " << params.precision << "\n";
    std::cout << "Pattern:   " << params.pattern << "\n";
    std::cout << "N:         " << params.N << "\n";
    std::cout << "Runs:      " << params.runs << "\n";
    std::cout << "================================================================\n\n";
    
    // Measure A_1 and A_∞
    double A_1 = Bench::measure_A1(params.N, params.warmup, params.runs);
    double A_inf = Bench::measure_Ainf(params.N, params.warmup, params.runs);
    
    std::cout << "Baseline rates:\n";
    std::cout << "  A_1 (unique):   " << std::scientific << A_1 << " atomics/sec\n";
    std::cout << "  A_∞ (hotspot):  " << A_inf << " atomics/sec\n";
    std::cout << "  Ratio A_1/A_∞:  " << std::fixed << std::setprecision(1) 
              << A_1 / A_inf << "x\n\n";
    
    // Measure A_eff for each phi
    std::vector<double> A_eff_values;
    std::vector<double> A_eff_predicted;
    
    std::cout << "A_eff vs φ:\n";
    std::cout << std::left << std::setw(8) << "φ"
              << std::setw(16) << "A_eff"
              << std::setw(16) << "Predicted"
              << std::setw(10) << "Error"
              << std::setw(12) << "A_eff/A_1"
              << "\n";
    std::cout << std::string(62, '-') << "\n";
    
    for (int phi : params.phi_values) {
        double A_eff;
        if (params.pattern == "strided") {
            A_eff = Bench::measure_Aeff_strided(params.N, phi, params.warmup, params.runs);
        } else if (params.pattern == "random") {
            A_eff = Bench::measure_Aeff_random(params.N, phi, params.warmup, params.runs);
        } else {
            A_eff = Bench::measure_Aeff_blocked(params.N, phi, params.warmup, params.runs);
        }
        
        // Model prediction: A_eff(φ) = (1/A_1 + (φ-1)/A_∞)^(-1)
        double predicted = 1.0 / (1.0 / A_1 + (phi - 1.0) / A_inf);
        double error_pct = 100.0 * std::abs(A_eff - predicted) / A_eff;
        
        A_eff_values.push_back(A_eff);
        A_eff_predicted.push_back(predicted);
        
        std::cout << std::left << std::setw(8) << phi
                  << std::scientific << std::setw(16) << A_eff
                  << std::setw(16) << predicted
                  << std::fixed << std::setprecision(1) << std::setw(10) << error_pct << "%"
                  << std::setw(12) << std::setprecision(3) << A_eff / A_1
                  << "\n";
    }
    
    // Compute fit quality
    double sse = 0, ss_tot = 0;
    double mean_A = std::accumulate(A_eff_values.begin(), A_eff_values.end(), 0.0) / A_eff_values.size();
    for (size_t i = 0; i < A_eff_values.size(); ++i) {
        sse += (A_eff_values[i] - A_eff_predicted[i]) * (A_eff_values[i] - A_eff_predicted[i]);
        ss_tot += (A_eff_values[i] - mean_A) * (A_eff_values[i] - mean_A);
    }
    double r_squared = 1.0 - sse / ss_tot;
    double rmse = std::sqrt(sse / A_eff_values.size());
    
    std::cout << "\nModel fit quality:\n";
    std::cout << "  R² = " << std::fixed << std::setprecision(4) << r_squared << "\n";
    std::cout << "  RMSE = " << std::scientific << rmse << " atomics/sec\n";
    
    // Write CSV
    std::ofstream out(params.output);
    out << "phi,A_eff,predicted,error_pct,A_eff_over_A1\n";
    for (size_t i = 0; i < params.phi_values.size(); ++i) {
        double error_pct = 100.0 * std::abs(A_eff_values[i] - A_eff_predicted[i]) / A_eff_values[i];
        out << params.phi_values[i] << ","
            << std::scientific << A_eff_values[i] << ","
            << A_eff_predicted[i] << ","
            << std::fixed << error_pct << ","
            << A_eff_values[i] / A_1 << "\n";
    }
    out.close();
    
    std::cout << "\nWrote results to: " << params.output << "\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    
    {
        Params params = parse_args(argc, argv);
        
        if (params.precision == "float") {
            run_benchmark<float>(params);
        } else if (params.precision == "complex") {
            run_benchmark<Kokkos::complex<double>>(params);
        } else {
            run_benchmark<double>(params);
        }
    }
    
    Kokkos::finalize();
    return 0;
}
