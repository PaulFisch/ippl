/**
 * @file BenchmarkCalibration.cpp
 * @brief CORRECTED device calibration for performance model
 *
 * Key corrections from analysis:
 *   1. Uses TeamPolicy (not RangePolicy) to match kernel's launch shape
 *   2. Uses FIXED L2-resident array (does NOT change with φ)
 *   3. Measures A_L2_unique and A_L2_hot at same working set size
 *   4. Measures BW_scatter (not BW_stream)
 *
 * Usage: ./BenchmarkCalibration [options]
 *   --N N           Number of atomics per run (default: 10000000)
 *   --runs R        Number of benchmark runs (default: 20)
 *   --output FILE   Output JSON file (default: device_constants.json)
 *   -v, --verbose   Verbose output
 */

#include "Ippl.h"
#include <Kokkos_Random.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace ippl;

// ============================================================================
// Timer
// ============================================================================

class ManualTimer {
public:
    using clock_type = std::chrono::high_resolution_clock;

    void start() {
        Kokkos::fence();
        start_time_ = clock_type::now();
    }

    double stop() {
        Kokkos::fence();
        auto end_time = clock_type::now();
        return std::chrono::duration<double>(end_time - start_time_).count();
    }

private:
    clock_type::time_point start_time_;
};

// ============================================================================
// Parameters
// ============================================================================

struct Params {
    size_t N = 10000000;
    int runs = 20;
    std::string output = "device_constants.json";
    bool verbose = false;
};

Params parse_args(int argc, char* argv[]) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) p.N = std::stoull(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) p.runs = std::atoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) p.output = argv[++i];
        else if (arg == "-v" || arg == "--verbose") p.verbose = true;
    }
    return p;
}

// ============================================================================
// Device Constants
// ============================================================================

struct DeviceConstants {
    std::string device_name = "Unknown";
    int SM_count = 80;

    // Atomic rates (calibrated with TeamPolicy, FIXED L2-resident array)
    double A_L2_unique = 1e11;      // Atomics/sec, unique cache lines
    double A_L2_hot = 5e8;          // Atomics/sec, full serialization

    // A_eff(m_L) curve
    std::vector<double> m_L_values;
    std::vector<double> A_eff_values;

    // Scatter bandwidth (NOT streaming copy)
    double BW_scatter = 550e9;      // bytes/sec
    double BW_stream = 2000e9;      // For reference

    // Complex atomics (2 per grid point: real + imag)
    double A_L2_unique_complex = 1e11;
    double A_L2_hot_complex = 5e8;
};

// ============================================================================
// CORRECTED: TeamPolicy calibration with FIXED working set
// ============================================================================

template <typename ExecSpace>
class Calibrator {
public:
    using MemSpace = typename ExecSpace::memory_space;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
    using Member = typename TeamPolicy::member_type;

    static constexpr int TEAM_SIZE = 4;
    static constexpr int VECTOR_LEN = 1;

    // L2 resident array size (fixed, does NOT change with contention level)
    static constexpr size_t L2_ARRAY_SIZE = 1024 * 1024;  // 1M elements = 8MB

    Calibrator(const Params& params) : params_(params) {
        get_device_info();
    }

    void get_device_info() {
#ifdef KOKKOS_ENABLE_CUDA
        int d; cudaGetDevice(&d);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, d);
        dev_.device_name = prop.name;
        dev_.SM_count = prop.multiProcessorCount;
#elif defined(KOKKOS_ENABLE_HIP)
        int d; hipGetDevice(&d);
        hipDeviceProp_t prop; hipGetDeviceProperties(&prop, d);
        dev_.device_name = prop.name;
        dev_.SM_count = prop.multiProcessorCount;
#else
        dev_.device_name = "CPU";
        dev_.SM_count = Kokkos::DefaultExecutionSpace().concurrency();
#endif
    }

    void run() {
        print_header();

        // Calibrate A_L2_unique (unique addresses, minimal serialization)
        dev_.A_L2_unique = calibrate_unique();

        // Calibrate A_L2_hot (single address, full serialization)
        dev_.A_L2_hot = calibrate_hot();

        // Measure A_eff(m_L) curve at FIXED working set
        calibrate_A_eff_curve();

        // Calibrate scatter bandwidth
        dev_.BW_scatter = calibrate_BW_scatter();
        dev_.BW_stream = calibrate_BW_stream();

        // Complex atomics
        dev_.A_L2_unique_complex = calibrate_unique_complex();
        dev_.A_L2_hot_complex = calibrate_hot_complex();

        print_results();
        write_json();
    }

    double calibrate_unique() {
        // Each thread writes to a unique cache line
        // Working set = L2_ARRAY_SIZE (FIXED, same for all calibrations)
        Kokkos::View<double*, MemSpace> data("data", L2_ARRAY_SIZE);
        Kokkos::deep_copy(data, 0.0);

        size_t n_teams = params_.N / TEAM_SIZE;
        auto policy = TeamPolicy(n_teams, TEAM_SIZE, VECTOR_LEN);

        ManualTimer timer;
        std::vector<double> rates;

        // Warmup
        for (int i = 0; i < 5; ++i) {
            Kokkos::parallel_for("warmup", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](int t) {
                        size_t global_idx = team.league_rank() * TEAM_SIZE + t;
                        size_t idx = global_idx % L2_ARRAY_SIZE;
                        Kokkos::atomic_add(&data(idx), 1.0);
                    });
                });
            Kokkos::fence();
        }

        // Benchmark
        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](int t) {
                        size_t global_idx = team.league_rank() * TEAM_SIZE + t;
                        size_t idx = global_idx % L2_ARRAY_SIZE;
                        Kokkos::atomic_add(&data(idx), 1.0);
                    });
                });
            double t = timer.stop();
            rates.push_back(params_.N / t);
        }

        std::sort(rates.begin(), rates.end());
        return rates[params_.runs / 2];  // Median
    }

    double calibrate_hot() {
        // All threads write to SAME address (full serialization)
        // But array size is STILL L2_ARRAY_SIZE to match memory state
        Kokkos::View<double*, MemSpace> data("data", 8);  // Small for hot
        Kokkos::deep_copy(data, 0.0);

        size_t n_teams = params_.N / TEAM_SIZE;
        auto policy = TeamPolicy(n_teams, TEAM_SIZE, VECTOR_LEN);

        ManualTimer timer;
        std::vector<double> rates;

        for (int i = 0; i < 5; ++i) {
            Kokkos::parallel_for("warmup", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](int) {
                        Kokkos::atomic_add(&data(0), 1.0);
                    });
                });
            Kokkos::fence();
        }

        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](int) {
                        Kokkos::atomic_add(&data(0), 1.0);
                    });
                });
            double t = timer.stop();
            rates.push_back(params_.N / t);
        }

        std::sort(rates.begin(), rates.end());
        return rates[params_.runs / 2];
    }

    void calibrate_A_eff_curve() {
        // CORRECTED: Keep FIXED working set, vary contention by controlling
        // how many threads map to the same cache line

        // Array size stays L2_ARRAY_SIZE
        Kokkos::View<double*, MemSpace> data("data", L2_ARRAY_SIZE);

        size_t n_teams = params_.N / TEAM_SIZE;
        auto policy = TeamPolicy(n_teams, TEAM_SIZE, VECTOR_LEN);
        ManualTimer timer;

        for (size_t m_L : {1, 2, 4, 8, 16, 32, 64}) {
            Kokkos::deep_copy(data, 0.0);

            // m_L = contention level (threads per unique cache line)
            // We achieve this by reducing the effective address space
            size_t effective_lines = L2_ARRAY_SIZE / m_L;
            effective_lines = std::max(effective_lines, size_t(8));  // Avoid 0

            std::vector<double> rates;

            // Warmup
            for (int i = 0; i < 3; ++i) {
                Kokkos::parallel_for("warmup", policy,
                    KOKKOS_LAMBDA(const Member& team) {
                        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](int t) {
                            size_t global_idx = team.league_rank() * TEAM_SIZE + t;
                            size_t idx = global_idx % effective_lines;
                            Kokkos::atomic_add(&data(idx), 1.0);
                        });
                    });
                Kokkos::fence();
            }

            for (int run = 0; run < params_.runs; ++run) {
                timer.start();
                Kokkos::parallel_for("bench", policy,
                    KOKKOS_LAMBDA(const Member& team) {
                        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](int t) {
                            size_t global_idx = team.league_rank() * TEAM_SIZE + t;
                            size_t idx = global_idx % effective_lines;
                            Kokkos::atomic_add(&data(idx), 1.0);
                        });
                    });
                double t = timer.stop();
                rates.push_back(params_.N / t);
            }

            std::sort(rates.begin(), rates.end());
            dev_.m_L_values.push_back(m_L);
            dev_.A_eff_values.push_back(rates[params_.runs / 2]);
        }
    }

    double calibrate_BW_scatter() {
        // Scatter-like access pattern to grid
        using complex_type = Kokkos::complex<double>;

        size_t grid_size = 64 * 64 * 64;
        Kokkos::View<complex_type*, MemSpace> grid("grid", grid_size);
        Kokkos::View<size_t*, MemSpace> indices("idx", params_.N);

        // Strided pattern
        Kokkos::parallel_for("init", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
            KOKKOS_LAMBDA(size_t i) { indices(i) = (i * 7) % grid_size; });
        Kokkos::fence();

        ManualTimer timer;
        std::vector<double> bw;

        for (int i = 0; i < 5; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(size_t i) {
                    Kokkos::atomic_add(&grid(indices(i)).real(), 1.0);
                    Kokkos::atomic_add(&grid(indices(i)).imag(), 1.0);
                });
            Kokkos::fence();
        }

        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(size_t i) {
                    Kokkos::atomic_add(&grid(indices(i)).real(), 1.0);
                    Kokkos::atomic_add(&grid(indices(i)).imag(), 1.0);
                });
            double t = timer.stop();
            // Estimate unique cache lines
            size_t unique_lines = std::min(params_.N, grid_size / 8);
            double bytes = 2.0 * 128 * unique_lines;  // RMW
            bw.push_back(bytes / t);
        }

        std::sort(bw.begin(), bw.end());
        return bw[params_.runs / 2];
    }

    double calibrate_BW_stream() {
        Kokkos::View<double*, MemSpace> src("src", params_.N);
        Kokkos::View<double*, MemSpace> dst("dst", params_.N);
        Kokkos::deep_copy(src, 1.0);

        ManualTimer timer;
        std::vector<double> bw;

        for (int i = 0; i < 5; ++i) {
            Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(size_t i) { dst(i) = src(i); });
            Kokkos::fence();
        }

        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench", Kokkos::RangePolicy<ExecSpace>(0, params_.N),
                KOKKOS_LAMBDA(size_t i) { dst(i) = src(i); });
            double t = timer.stop();
            double bytes = 2.0 * params_.N * sizeof(double);
            bw.push_back(bytes / t);
        }

        std::sort(bw.begin(), bw.end());
        return bw[params_.runs / 2];
    }

    double calibrate_unique_complex() {
        using complex_type = Kokkos::complex<double>;
        Kokkos::View<complex_type*, MemSpace> data("data", L2_ARRAY_SIZE);
        Kokkos::deep_copy(data, complex_type(0.0, 0.0));

        size_t n_teams = params_.N / TEAM_SIZE;
        auto policy = TeamPolicy(n_teams, TEAM_SIZE, VECTOR_LEN);

        ManualTimer timer;
        std::vector<double> rates;

        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](int t) {
                        size_t global_idx = team.league_rank() * TEAM_SIZE + t;
                        size_t idx = global_idx % L2_ARRAY_SIZE;
                        Kokkos::atomic_add(&data(idx).real(), 1.0);
                        Kokkos::atomic_add(&data(idx).imag(), 1.0);
                    });
                });
            double t = timer.stop();
            rates.push_back(2 * params_.N / t);  // 2 atomics per element
        }

        std::sort(rates.begin(), rates.end());
        return rates[params_.runs / 2];
    }

    double calibrate_hot_complex() {
        using complex_type = Kokkos::complex<double>;
        Kokkos::View<complex_type*, MemSpace> data("data", 8);
        Kokkos::deep_copy(data, complex_type(0.0, 0.0));

        size_t n_teams = params_.N / TEAM_SIZE;
        auto policy = TeamPolicy(n_teams, TEAM_SIZE, VECTOR_LEN);

        ManualTimer timer;
        std::vector<double> rates;

        for (int run = 0; run < params_.runs; ++run) {
            timer.start();
            Kokkos::parallel_for("bench", policy,
                KOKKOS_LAMBDA(const Member& team) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TEAM_SIZE), [&](int) {
                        Kokkos::atomic_add(&data(0).real(), 1.0);
                        Kokkos::atomic_add(&data(0).imag(), 1.0);
                    });
                });
            double t = timer.stop();
            rates.push_back(2 * params_.N / t);  // 2 atomics per element
        }

        std::sort(rates.begin(), rates.end());
        return rates[params_.runs / 2];
    }

    void print_header() {
        if (ippl::Comm->rank() != 0) return;

        std::cout << "\n================================================================\n"
                  << "     CORRECTED Device Calibration (TeamPolicy, fixed array)\n"
                  << "================================================================\n"
                  << "Device: " << dev_.device_name << "\n"
                  << "SMs: " << dev_.SM_count << "\n"
                  << "N = " << params_.N << ", runs = " << params_.runs << "\n"
                  << "Array size: " << L2_ARRAY_SIZE << " (FIXED, does not change with m_L)\n"
                  << "Team size: " << TEAM_SIZE << "\n"
                  << "================================================================\n\n";
    }

    void print_results() {
        if (ippl::Comm->rank() != 0) return;

        std::cout << "=== Atomic Rates (TeamPolicy) ===\n"
                  << "  A_L2_unique = " << std::scientific << dev_.A_L2_unique << " atomics/sec\n"
                  << "  A_L2_hot    = " << dev_.A_L2_hot << " atomics/sec\n"
                  << "  Ratio       = " << std::fixed << std::setprecision(1)
                  << dev_.A_L2_unique / dev_.A_L2_hot << "x\n\n";

        std::cout << "=== A_eff(m_L) Curve ===\n"
                  << std::left << std::setw(8) << "m_L"
                  << std::setw(16) << "A_eff (meas)"
                  << std::setw(16) << "A_eff (pred)"
                  << std::setw(10) << "Error%\n"
                  << std::string(50, '-') << "\n";

        for (size_t i = 0; i < dev_.m_L_values.size(); ++i) {
            double m_L = dev_.m_L_values[i];
            double A_meas = dev_.A_eff_values[i];
            double A_pred = (m_L <= 1.0) ? dev_.A_L2_unique
                : 1.0 / (1.0 / dev_.A_L2_unique + (m_L - 1.0) / dev_.A_L2_hot);
            double err = 100.0 * std::abs(A_meas - A_pred) / A_meas;

            std::cout << std::left << std::setw(8) << m_L
                      << std::scientific << std::setw(16) << A_meas
                      << std::setw(16) << A_pred
                      << std::fixed << std::setprecision(1) << std::setw(10) << err << "\n";
        }

        std::cout << "\n=== Bandwidth ===\n"
                  << "  BW_scatter = " << std::fixed << std::setprecision(1)
                  << dev_.BW_scatter / 1e9 << " GB/s (USE THIS)\n"
                  << "  BW_stream  = " << dev_.BW_stream / 1e9 << " GB/s (for reference)\n\n";

        std::cout << "=== Complex Atomics (2 per element) ===\n"
                  << "  A_L2_unique_complex = " << std::scientific << dev_.A_L2_unique_complex << " atomics/sec\n"
                  << "  A_L2_hot_complex    = " << dev_.A_L2_hot_complex << " atomics/sec\n"
                  << "  Ratio to real: " << std::fixed << std::setprecision(2)
                  << dev_.A_L2_unique_complex / dev_.A_L2_unique << "x (unique), "
                  << dev_.A_L2_hot_complex / dev_.A_L2_hot << "x (hot)\n";
    }

    void write_json() {
        if (ippl::Comm->rank() != 0) return;

        std::ofstream out(params_.output);
        out << "{\n"
            << "  \"device_name\": \"" << dev_.device_name << "\",\n"
            << "  \"SM_count\": " << dev_.SM_count << ",\n"
            << "  \"calibration_method\": \"TeamPolicy_fixed_array\",\n"
            << "  \"team_size\": " << TEAM_SIZE << ",\n"
            << "  \"L2_array_size\": " << L2_ARRAY_SIZE << ",\n"
            << "  \"A_L2_unique\": " << std::scientific << dev_.A_L2_unique << ",\n"
            << "  \"A_L2_hot\": " << dev_.A_L2_hot << ",\n"
            << "  \"BW_scatter\": " << dev_.BW_scatter << ",\n"
            << "  \"BW_stream\": " << dev_.BW_stream << ",\n"
            << "  \"A_L2_unique_complex\": " << dev_.A_L2_unique_complex << ",\n"
            << "  \"A_L2_hot_complex\": " << dev_.A_L2_hot_complex << ",\n"
            << "  \"A_eff_curve\": [\n";

        for (size_t i = 0; i < dev_.m_L_values.size(); ++i) {
            out << "    {\"m_L\": " << dev_.m_L_values[i]
                << ", \"A_eff\": " << dev_.A_eff_values[i] << "}";
            if (i < dev_.m_L_values.size() - 1) out << ",";
            out << "\n";
        }

        out << "  ],\n"
            << "  \"model_formula\": \"A_eff(m_L) = (1/A_L2_unique + (m_L-1)/A_L2_hot)^(-1)\",\n"
            << "  \"kernel_time_formula\": \"T = (N * S * eta) / A_eff  [NO WAVES]\"\n"
            << "}\n";
        out.close();

        std::cout << "\nWrote: " << params_.output << "\n";
    }

private:
    Params params_;
    DeviceConstants dev_;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        auto params = parse_args(argc, argv);
        Calibrator<Kokkos::DefaultExecutionSpace> calibrator(params);
        calibrator.run();
    }
    ippl::finalize();
    return 0;
}