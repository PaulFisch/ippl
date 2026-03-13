# -----------------------------------------------------------------------------
# test/FFT/PaperBenchmarks/CMakeLists.txt
#
# Benchmark executables used to generate data for the PASC '26 paper
# "A Performance-Portable, Massively Parallel Distributed Nonuniform FFT".
#
# Each target maps to one or more paper figures:
#   BenchKernels           – Fig 1, 2  (kernel throughput vs tolerance)
#   TileSweep              – Fig 3     (tile-size / kernel-width heatmap)
#   BreakdownNUFFT         – Fig 4     (NUFFT component time breakdown)
#   BenchDensities         – Fig 5     (comparison with cuFINUFFT)
#   BenchmarkNUFFT         – Fig 6, 7  (distributed strong-scaling)
#   TestPrunedFFTPerformance – Fig 8   (pruned vs full FFT)
#
# Additional scatter / gather micro-benchmarks:
#   BenchmarkScatter
#   BenchmarkGather
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding paper benchmarks from: ${_relPath}")

# All targets in this folder need the local directory on the include path so
# that BenchmarkUtils.h and finufft_wrapper.h are found by their neighbours.
set(PAPER_BENCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Helper macro: like add_ippl_integration_test but also adds the local dir.
macro(add_paper_benchmark TARGET)
    add_ippl_integration_test(${TARGET} ${ARGN})
    target_include_directories(${TARGET} PRIVATE "${PAPER_BENCH_DIR}")
endmacro()

# ── Kernel benchmark (Figs 1, 2) ─────────────────────────────────────────────
add_paper_benchmark(BenchKernels   LABELS paper benchmark fft)
target_compile_options(BenchKernels PRIVATE "-lineinfo")

# ── Tile-size sweep / BO optimiser (Fig 3) ────────────────────────────────────
add_paper_benchmark(TileSweep      LABELS paper benchmark fft)

# ── NUFFT time breakdown (Fig 4) ──────────────────────────────────────────────
add_paper_benchmark(BreakdownNUFFT LABELS paper benchmark fft)
if(TARGET finufft OR TARGET cufinufft)
    target_link_libraries(BreakdownNUFFT PRIVATE finufft cufinufft)
endif()

# ── cuFINUFFT comparison (Fig 5) ──────────────────────────────────────────────
add_paper_benchmark(BenchDensities LABELS paper benchmark fft)
if(TARGET finufft OR TARGET cufinufft)
    target_link_libraries(BenchDensities PRIVATE finufft cufinufft)
endif()

# ── Distributed strong-scaling (Figs 6, 7) ────────────────────────────────────
add_paper_benchmark(BenchmarkNUFFT LABELS paper benchmark fft)
if(TARGET finufft OR TARGET cufinufft)
    target_link_libraries(BenchmarkNUFFT PRIVATE finufft cufinufft)
endif()

# ── Pruned-FFT performance (Fig 8) ────────────────────────────────────────────
add_paper_benchmark(TestPrunedFFTPerformance LABELS paper benchmark fft)

# ── Scatter / gather micro-benchmarks ─────────────────────────────────────────
add_paper_benchmark(BenchmarkScatter LABELS paper benchmark fft)
add_paper_benchmark(BenchmarkGather  LABELS paper benchmark fft)
add_paper_benchmark(SingleNode  LABELS paper benchmark fft)
if(TARGET finufft OR TARGET cufinufft)
    target_link_libraries(SingleNode PRIVATE finufft cufinufft)
endif()
