/**
 * @file BenchmarkCalibration.cpp
 * @brief Calibrate device constants for AtomicScatter performance model
 *
 * Measures:
 *   - A_1: Atomic throughput with unique addresses (no collisions)
 *   - A_∞: Atomic throughput at full contention (hotspot)
 *   - A_eff(φ): Atomic throughput at various collision factors
 *   - BW_HBM: Effective HBM bandwidth for scatter patterns
 *   - Occupancy limits
 *
 * Output: JSON file with calibrated constants for use in performance model
 *
 * Usage: ./BenchmarkCalibration [options]
 *   --N <size>       Number of elements for benchmarks (default: 10M)
 *   --runs <n>       Benchmark iterations (default: 20)
 *   --output <file>  Output JSON file (default: device_constants.json)
 *   --phi <list>     Collision factors to test (default: 1,2,4,8,16,32)
 *   -v, --verbose    Verbose output
 */

#include "Ippl.h"
#include <Kokkos_Random.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <vector>
#include <sstream>

using namespace ippl;

// ============================================================================
// Manual Timer Class (matches BenchmarkRoofline style)
// ============================================================================

class ManualTimer {
public:
    using clock_type = std::chrono::high_resolution_clock;
    using time_point = clock_type::time_point;

    void start() {
        Kokkos::fence();  // Ensure all previous GPU work is complete
        start_time_ = clock_type::now();
    }

    double stop() {
        Kokkos::fence();  // Ensure all GPU work from this section is complete
        auto end_time = clock_type::now();
        auto duration = std::chrono::duration<double>(end_time - start_time_);
        return duration.count();  // Returns seconds
    }

private:
    time_point start_time_;
};

// ============================================================================
// Parameters
// ============================================================================

struct CalibParams {
    size_t N = 10000000;
    int runs = 20;
    int warmup = 5;
    std::string output = "device_constants.json";
    std::vector<int> phi_values = {1, 2, 4, 8, 16, 32};
    bool verbose = false;
};

CalibParams parse_args(int argc, char* argv[]) {
    CalibParams p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) {
            p.N = std::stoull(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            p.runs = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            p.output = argv[++i];
        } else if (arg == "--phi" && i + 1 < argc) {
            p.phi_values.clear();
            std::stringstream ss(argv[++i]);
            std::string val;
            while (std::getline(ss, val, ',')) {
                p.phi_values.push_back(std::stoi(val));
            }
        } else if (arg == "-v" || arg == "--verbose") {
            p.verbose = true;
        }
    }
    return p;
}

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
// Calibration Results
// ============================================================================

struct CalibrationResults {
    // Device info
    std::string device_name;
    int sm_count;
    size_t shared_mem_per_sm;
    size_t max_threads_per_sm;
    
    // Atomic rates
    double A_1;           // Unique addresses (atomics/sec)
    double A_1_stddev;
    double A_inf;         // Full contention (atomics/sec)
    double A_inf_stddev;
    
    // A_eff vs phi curve
    std::vector<int> phi_values;
    std::vector<double> A_eff_values;
    std::vector<double> A_eff_stddev;
    
    // Fitted model parameters
    double A_1_fit;       // Fitted A_1 from curve
    double A_inf_fit;     // Fitted A_∞ from curve
    double fit_rmse;      // Fit quality
    
    // Bandwidth
    double BW_stream;     // Streaming bandwidth (bytes/sec)
    double BW_scatter;    // Scatter pattern bandwidth (bytes/sec)
    double BW_random;     // Random access bandwidth (bytes/sec)
    
    // Complex atomics
    double A_1_complex;   // With complex<double>
    double A_inf_complex;
};

// ============================================================================
// Benchmark Kernels
// ============================================================================

template <typename ExecSpace>
class DeviceCalibrator {
public:
    using MemSpace = typename ExecSpace::memory_space;
    
    DeviceCalibrator(const CalibParams& params) : params_(params) {}
    
    CalibrationResults run() {
        CalibrationResults results;
        
        get_device_info(results);
        
        if (ippl::Comm->rank() == 0) {
            std::cout << "\n================================================================\n";
            std::cout << "     Device Calibration for Performance Model\n";
            std::cout << "================================================================\n";
            std::cout << "Device: " << results.device_name << "\n";
            std::cout << "SMs: " << results.sm_count << "\n";
            std::cout << "N = " << params_.N << ", runs = " << params_.runs << "\n";
            std::cout << "================================================================\n\n";
        }
        
        // Run calibrations
        calibrate_A1(results);
        calibrate_Ainf(results);
        calibrate_Aeff_curve(results);
        fit_atomic_model(results);
        calibrate_bandwidth(results);
        calibrate_complex_atomics(results);
        
        // Output results
        print_results(results);
        write_json(results);
        
        return results;
    }
    
    void get_device_info(CalibrationResults& r) {
        r.device_name = "Unknown";
        
#ifdef KOKKOS_ENABLE_CUDA
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        r.device_name = prop.name;
        r.sm_count = prop.multiProcessorCount;
        r.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
        r.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
#elif defined(KOKKOS_ENABLE_HIP)
        int device;
        hipGetDevice(&device);
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, device);
        r.device_name = prop.name;
        r.sm_count = prop.multiProcessorCount;
        r.shared_mem_per_sm = prop.maxSharedMemoryPerMultiProcessor;
        r.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
#else
        r.sm_count = Kokkos::DefaultExecutionSpace().concurrency();
        r.shared_mem_per_sm = 48 * 1024;  // Typical
        r.max_threads_per_sm = 2048;
#endif
    }
    
    void calibrate_A1(CalibrationResults& r) {
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "Calibrating A_1 (unique addresses)...\n";
        }
        
        Kokkos::View<double*, MemSpace> data("data", params_.N);
        Kokkos::deep_copy(data, 0.0);
        
        ManualTimer timer;
        std::vector<double> rates;
        
        // Warmup
        for (int i = 0; i < params_.warmup; ++i) {
            Kokkos::parallel_for("warmup_A1", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(const size_t i) {
                    Kokkos::atomic_add(&data(i), 1.0);
                });
            Kokkos::fence();
        }
        
        // Benchmark
        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench_A1", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(const size_t i) {
                    Kokkos::atomic_add(&data(i), 1.0);
                });
            double t = timer.stop();
            rates.push_back(params_.N / t);
        }
        
        Stats s = compute_stats(rates);
        r.A_1 = s.median;
        r.A_1_stddev = s.stddev;
        
        if (ippl::Comm->rank() == 0) {
            std::cout << "  A_1 = " << std::scientific << r.A_1 
                      << " ± " << r.A_1_stddev << " atomics/sec\n";
        }
    }
    
    void calibrate_Ainf(CalibrationResults& r) {
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "Calibrating A_∞ (hotspot)...\n";
        }
        
        Kokkos::View<double*, MemSpace> data("data", 1);
        Kokkos::deep_copy(data, 0.0);
        
        ManualTimer timer;
        std::vector<double> rates;
        
        // Warmup
        for (int i = 0; i < params_.warmup; ++i) {
            Kokkos::parallel_for("warmup_Ainf", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(const size_t) {
                    Kokkos::atomic_add(&data(0), 1.0);
                });
            Kokkos::fence();
        }
        
        // Benchmark
        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench_Ainf", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(const size_t) {
                    Kokkos::atomic_add(&data(0), 1.0);
                });
            double t = timer.stop();
            rates.push_back(params_.N / t);
        }
        
        Stats s = compute_stats(rates);
        r.A_inf = s.median;
        r.A_inf_stddev = s.stddev;
        
        if (ippl::Comm->rank() == 0) {
            std::cout << "  A_∞ = " << std::scientific << r.A_inf 
                      << " ± " << r.A_inf_stddev << " atomics/sec\n";
            std::cout << "  A_1/A_∞ ratio = " << std::fixed << r.A_1 / r.A_inf << "x\n";
        }
    }
    
    void calibrate_Aeff_curve(CalibrationResults& r) {
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "Calibrating A_eff vs φ curve...\n";
        }
        
        r.phi_values = params_.phi_values;
        r.A_eff_values.resize(params_.phi_values.size());
        r.A_eff_stddev.resize(params_.phi_values.size());
        
        for (size_t p = 0; p < params_.phi_values.size(); ++p) {
            int phi = params_.phi_values[p];
            
            // Create array where phi threads collide on each address
            size_t n_unique = params_.N / phi;
            Kokkos::View<double*, MemSpace> data("data", n_unique);
            Kokkos::deep_copy(data, 0.0);
            
            ManualTimer timer;
            std::vector<double> rates;
            
            // Each thread i writes to data[i / phi]
            auto N = params_.N;
            
            // Warmup
            for (int i = 0; i < params_.warmup; ++i) {
                Kokkos::parallel_for("warmup_phi", Kokkos::RangePolicy<ExecSpace>(0, N),
                    KOKKOS_LAMBDA(const size_t i) {
                        size_t idx = i / phi;
                        if (idx < n_unique) {
                            Kokkos::atomic_add(&data(idx), 1.0);
                        }
                    });
                Kokkos::fence();
            }
            
            // Benchmark
            for (int run = 0; run < params_.runs; ++run) {
                timer.start();
                Kokkos::parallel_for("bench_phi", Kokkos::RangePolicy<ExecSpace>(0, N),
                    KOKKOS_LAMBDA(const size_t i) {
                        size_t idx = i / phi;
                        if (idx < n_unique) {
                            Kokkos::atomic_add(&data(idx), 1.0);
                        }
                    });
                double t = timer.stop();
                rates.push_back(N / t);
            }
            
            Stats s = compute_stats(rates);
            r.A_eff_values[p] = s.median;
            r.A_eff_stddev[p] = s.stddev;
            
            if (ippl::Comm->rank() == 0) {
                std::cout << "  A_eff(φ=" << phi << ") = " << std::scientific 
                          << r.A_eff_values[p] << " atomics/sec\n";
            }
        }
    }
    
    void fit_atomic_model(CalibrationResults& r) {
        // Fit the model: A_eff(φ) = (1/A_1 + (φ-1)/A_∞)^(-1)
        // Using least squares on 1/A_eff = 1/A_1 + (φ-1)/A_∞
        
        // Linear regression: y = a + b*x where
        // y = 1/A_eff, x = (φ-1), a = 1/A_1, b = 1/A_∞
        
        size_t n = r.phi_values.size();
        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        
        for (size_t i = 0; i < n; ++i) {
            double x = r.phi_values[i] - 1.0;
            double y = 1.0 / r.A_eff_values[i];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        
        double det = n * sum_xx - sum_x * sum_x;
        double a = (sum_xx * sum_y - sum_x * sum_xy) / det;
        double b = (n * sum_xy - sum_x * sum_y) / det;
        
        r.A_1_fit = 1.0 / a;
        r.A_inf_fit = 1.0 / b;
        
        // Compute RMSE
        double sse = 0;
        for (size_t i = 0; i < n; ++i) {
            double phi = r.phi_values[i];
            double predicted = 1.0 / (1.0 / r.A_1_fit + (phi - 1.0) / r.A_inf_fit);
            double residual = r.A_eff_values[i] - predicted;
            sse += residual * residual;
        }
        r.fit_rmse = std::sqrt(sse / n);
        
        if (ippl::Comm->rank() == 0) {
            std::cout << "\nFitted Model:\n";
            std::cout << "  A_1_fit  = " << std::scientific << r.A_1_fit << "\n";
            std::cout << "  A_∞_fit  = " << r.A_inf_fit << "\n";
            std::cout << "  RMSE     = " << r.fit_rmse << "\n";
        }
    }
    
    void calibrate_bandwidth(CalibrationResults& r) {
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "\nCalibrating bandwidth...\n";
        }
        
        Kokkos::View<double*, MemSpace> src("src", params_.N);
        Kokkos::View<double*, MemSpace> dst("dst", params_.N);
        Kokkos::View<size_t*, MemSpace> indices("indices", params_.N);
        
        Kokkos::deep_copy(src, 1.0);
        
        ManualTimer timer;
        
        // Streaming bandwidth (coalesced)
        {
            std::vector<double> bw;
            for (int i = 0; i < params_.warmup; ++i) {
                Kokkos::parallel_for("warmup_stream", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t i) { dst(i) = src(i); });
                Kokkos::fence();
            }
            for (int run = 0; run < params_.runs; ++run) {
                timer.start();
                Kokkos::parallel_for("bench_stream", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t i) { dst(i) = src(i); });
                double t = timer.stop();
                bw.push_back(2.0 * params_.N * sizeof(double) / t);
            }
            r.BW_stream = compute_stats(bw).median;
        }
        
        // Random access bandwidth
        {
            // Initialize random indices
            Kokkos::Random_XorShift64_Pool<> pool(42);
            auto N = params_.N;
            Kokkos::parallel_for("init_indices", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = pool.get_state();
                    indices(i) = gen.urand64() % N;
                    pool.free_state(gen);
                });
            Kokkos::fence();
            
            std::vector<double> bw;
            for (int i = 0; i < params_.warmup; ++i) {
                Kokkos::parallel_for("warmup_random", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t i) { dst(i) = src(indices(i)); });
                Kokkos::fence();
            }
            for (int run = 0; run < params_.runs; ++run) {
                timer.start();
                Kokkos::parallel_for("bench_random", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t i) { dst(i) = src(indices(i)); });
                double t = timer.stop();
                bw.push_back(2.0 * params_.N * sizeof(double) / t);
            }
            r.BW_random = compute_stats(bw).median;
        }
        
        // Scatter pattern (write-focused with some locality)
        {
            // Stride access pattern
            int stride = 17;  // Prime to avoid cache line alignment
            Kokkos::parallel_for("init_scatter_indices", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(const size_t i) {
                    indices(i) = (i * stride) % N;
                });
            Kokkos::fence();
            
            std::vector<double> bw;
            for (int i = 0; i < params_.warmup; ++i) {
                Kokkos::parallel_for("warmup_scatter", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t i) { 
                        Kokkos::atomic_add(&dst(indices(i)), src(i)); 
                    });
                Kokkos::fence();
            }
            for (int run = 0; run < params_.runs; ++run) {
                timer.start();
                Kokkos::parallel_for("bench_scatter", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t i) { 
                        Kokkos::atomic_add(&dst(indices(i)), src(i)); 
                    });
                double t = timer.stop();
                bw.push_back(3.0 * params_.N * sizeof(double) / t);  // Read src, read-modify-write dst
            }
            r.BW_scatter = compute_stats(bw).median;
        }
        
        if (ippl::Comm->rank() == 0) {
            std::cout << "  BW_stream  = " << r.BW_stream / 1e9 << " GB/s\n";
            std::cout << "  BW_random  = " << r.BW_random / 1e9 << " GB/s\n";
            std::cout << "  BW_scatter = " << r.BW_scatter / 1e9 << " GB/s\n";
        }
    }
    
    void calibrate_complex_atomics(CalibrationResults& r) {
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "\nCalibrating complex atomics...\n";
        }
        
        using complex_type = Kokkos::complex<double>;
        
        Kokkos::View<complex_type*, MemSpace> data_unique("data_unique", params_.N);
        Kokkos::View<complex_type*, MemSpace> data_hotspot("data_hotspot", 1);
        
        ManualTimer timer;
        
        // A_1 for complex
        {
            std::vector<double> rates;
            for (int i = 0; i < params_.warmup; ++i) {
                Kokkos::parallel_for("warmup_A1_complex", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t i) {
                        auto val = data_unique(i);
                        Kokkos::atomic_add(&data_unique(i).real(), 1.0);
                        Kokkos::atomic_add(&data_unique(i).imag(), 1.0);
                    });
                Kokkos::fence();
            }
            for (int run = 0; run < params_.runs; ++run) {
                timer.start();
                Kokkos::parallel_for("bench_A1_complex", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t i) {
                        Kokkos::atomic_add(&data_unique(i).real(), 1.0);
                        Kokkos::atomic_add(&data_unique(i).imag(), 1.0);
                    });
                double t = timer.stop();
                rates.push_back(2 * params_.N / t);  // 2 atomics per particle
            }
            r.A_1_complex = compute_stats(rates).median;
        }
        
        // A_∞ for complex
        {
            std::vector<double> rates;
            for (int i = 0; i < params_.warmup; ++i) {
                Kokkos::parallel_for("warmup_Ainf_complex", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t) {
                        Kokkos::atomic_add(&data_hotspot(0).real(), 1.0);
                        Kokkos::atomic_add(&data_hotspot(0).imag(), 1.0);
                    });
                Kokkos::fence();
            }
            for (int run = 0; run < params_.runs; ++run) {
                timer.start();
                Kokkos::parallel_for("bench_Ainf_complex", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                    KOKKOS_LAMBDA(const size_t) {
                        Kokkos::atomic_add(&data_hotspot(0).real(), 1.0);
                        Kokkos::atomic_add(&data_hotspot(0).imag(), 1.0);
                    });
                double t = timer.stop();
                rates.push_back(2 * params_.N / t);
            }
            r.A_inf_complex = compute_stats(rates).median;
        }
        
        if (ippl::Comm->rank() == 0) {
            std::cout << "  A_1_complex  = " << std::scientific << r.A_1_complex << " atomics/sec\n";
            std::cout << "  A_∞_complex  = " << r.A_inf_complex << " atomics/sec\n";
            std::cout << "  Complex/Real ratio: " << std::fixed 
                      << (r.A_1_complex / r.A_1) << "x (unique), "
                      << (r.A_inf_complex / r.A_inf) << "x (hotspot)\n";
        }
    }
    
    void print_results(const CalibrationResults& r) {
        if (ippl::Comm->rank() != 0) return;
        
        std::cout << "\n================================================================\n";
        std::cout << "                    Calibration Summary\n";
        std::cout << "================================================================\n";
        std::cout << "Device: " << r.device_name << "\n";
        std::cout << "SMs: " << r.sm_count << "\n\n";
        
        std::cout << "Atomic Rates:\n";
        std::cout << "  A_1 (direct):  " << std::scientific << r.A_1 << " atomics/sec\n";
        std::cout << "  A_∞ (direct):  " << r.A_inf << " atomics/sec\n";
        std::cout << "  A_1 (fitted):  " << r.A_1_fit << " atomics/sec\n";
        std::cout << "  A_∞ (fitted):  " << r.A_inf_fit << " atomics/sec\n";
        std::cout << "  Ratio A_1/A_∞: " << std::fixed << r.A_1 / r.A_inf << "x\n\n";
        
        std::cout << "Bandwidth:\n";
        std::cout << "  Stream:  " << r.BW_stream / 1e9 << " GB/s\n";
        std::cout << "  Random:  " << r.BW_random / 1e9 << " GB/s\n";
        std::cout << "  Scatter: " << r.BW_scatter / 1e9 << " GB/s\n\n";
        
        std::cout << "Complex Atomics (2× per particle):\n";
        std::cout << "  A_1:  " << std::scientific << r.A_1_complex << " atomics/sec\n";
        std::cout << "  A_∞:  " << r.A_inf_complex << " atomics/sec\n";
        std::cout << "================================================================\n";
    }
    
    void write_json(const CalibrationResults& r) {
        if (ippl::Comm->rank() != 0) return;
        
        std::ofstream out(params_.output);
        out << "{\n";
        out << "  \"device_name\": \"" << r.device_name << "\",\n";
        out << "  \"sm_count\": " << r.sm_count << ",\n";
        out << "  \"shared_mem_per_sm\": " << r.shared_mem_per_sm << ",\n";
        out << "  \"max_threads_per_sm\": " << r.max_threads_per_sm << ",\n\n";
        
        out << "  \"atomic_rates\": {\n";
        out << "    \"A_1\": " << std::scientific << r.A_1 << ",\n";
        out << "    \"A_1_stddev\": " << r.A_1_stddev << ",\n";
        out << "    \"A_inf\": " << r.A_inf << ",\n";
        out << "    \"A_inf_stddev\": " << r.A_inf_stddev << ",\n";
        out << "    \"A_1_fit\": " << r.A_1_fit << ",\n";
        out << "    \"A_inf_fit\": " << r.A_inf_fit << ",\n";
        out << "    \"fit_rmse\": " << r.fit_rmse << "\n";
        out << "  },\n\n";
        
        out << "  \"A_eff_curve\": [\n";
        for (size_t i = 0; i < r.phi_values.size(); ++i) {
            out << "    {\"phi\": " << r.phi_values[i] 
                << ", \"A_eff\": " << r.A_eff_values[i]
                << ", \"stddev\": " << r.A_eff_stddev[i] << "}";
            if (i < r.phi_values.size() - 1) out << ",";
            out << "\n";
        }
        out << "  ],\n\n";
        
        out << "  \"bandwidth\": {\n";
        out << "    \"stream\": " << r.BW_stream << ",\n";
        out << "    \"random\": " << r.BW_random << ",\n";
        out << "    \"scatter\": " << r.BW_scatter << "\n";
        out << "  },\n\n";
        
        out << "  \"complex_atomics\": {\n";
        out << "    \"A_1_complex\": " << r.A_1_complex << ",\n";
        out << "    \"A_inf_complex\": " << r.A_inf_complex << "\n";
        out << "  }\n";
        
        out << "}\n";
        out.close();
        
        std::cout << "\nWrote calibration to: " << params_.output << "\n";
    }
    
private:
    CalibParams params_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    
    {
        auto params = parse_args(argc, argv);
        DeviceCalibrator<Kokkos::DefaultExecutionSpace> calibrator(params);
        calibrator.run();
    }
    
    ippl::finalize();
    return EXIT_SUCCESS;
}
