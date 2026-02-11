# -----------------------------------------------------------------------------
# CMakeLists.txt for AtomicScatter Performance Model Benchmarks
#
# This integrates with IPPL's build system. Add as a subdirectory:
#   add_subdirectory(performance_model/integrated)
#
# Uses IPPL's AddIpplIntegrationTest.cmake helper for test registration.
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# Include the IPPL test helper (should already be available from parent)
if(NOT COMMAND add_ippl_integration_test)
    message(FATAL_ERROR "add_ippl_integration_test not defined. "
                        "Include AddIpplIntegrationTest.cmake from IPPL first.")
endif()

# -----------------------------------------------------------------------------
# Main Benchmarks (use actual IPPL ESKernel and scatter APIs)
# -----------------------------------------------------------------------------

# AtomicScatter model validation benchmark
add_ippl_integration_test(BenchmarkAtomicScatter
    SOURCES BenchmarkAtomicScatter.cpp
    ARGS --grid 64 --rho 10 --runs 10
    NUM_PROCS 1
    TIMEOUT 600
    LABELS benchmark performance_model
)

# Calibration benchmark (measures device constants A_1, A_inf, BW)
add_ippl_integration_test(BenchmarkCalibration
    SOURCES BenchmarkCalibration.cpp
    ARGS --N 10000000 --runs 20
    NUM_PROCS 1
    TIMEOUT 300
    LABELS benchmark calibration
)

# Collision factor analyzer
add_ippl_integration_test(BenchmarkCollisionAnalysis
    SOURCES BenchmarkCollisionAnalysis.cpp
    ARGS --grid 64 --rho 10
    NUM_PROCS 1
    TIMEOUT 300
    LABELS benchmark analysis
)

# Kernel width sweep (for accuracy vs throughput plot)
add_ippl_integration_test(BenchmarkWidthSweep
    SOURCES BenchmarkAtomicScatter.cpp
    ARGS --grid 64 --rho 10 --sweep-w --runs 10
    NUM_PROCS 1
    TIMEOUT 1200
    LABELS benchmark sweep
)

# -----------------------------------------------------------------------------
# Standalone microbenchmarks (pure Kokkos, no IPPL - for understanding GPU behavior)
# These bypass IPPL to isolate atomic/memory behavior
# -----------------------------------------------------------------------------

# Only build these if not requiring full IPPL integration test infrastructure
if(TARGET Kokkos::kokkos)
    # Atomic throughput vs collision factor
    add_executable(MicrobenchAtomicVariable MicrobenchAtomicVariable.cpp)
    target_link_libraries(MicrobenchAtomicVariable PRIVATE Kokkos::kokkos)

    # Team-based atomic scatter (mimics IPPL kernel structure)
    add_executable(MicrobenchTeamAtomic MicrobenchTeamAtomic.cpp)
    target_link_libraries(MicrobenchTeamAtomic PRIVATE Kokkos::kokkos)
endif()
