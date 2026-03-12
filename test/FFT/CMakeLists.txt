# -----------------------------------------------------------------------------
# test/FFT/CMakeLists.txt
#
# Integration tests for FFT functionality. These build example executables that run and validate FFT
# behavior across supported configurations.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding FFT integration tests from: ${_relPath}")

add_ippl_integration_test(TestFFTCC LABELS integration fft)
add_ippl_integration_test(TestFFTRC LABELS integration fft)
add_ippl_integration_test(TestPrunedFFT LABELS integration fft)
add_ippl_integration_test(TestSine LABELS integration fft)
add_ippl_integration_test(TestCos LABELS integration fft)
add_ippl_integration_test(TestCos1 LABELS integration fft)
add_ippl_integration_test(TestNUFFT1 LABELS integration fft)
add_ippl_integration_test(TestNUFFT1_compare LABELS integration fft)
add_ippl_integration_test(TestNUFFT1_new LABELS integration fft)
add_ippl_integration_test(TestNUFFT2 LABELS integration fft)
add_ippl_integration_test(TestNUFFT2_new LABELS integration fft)
add_ippl_integration_test(TestNUFFT2_old LABELS integration fft)
add_ippl_integration_test(TestNUFFTPerformance LABELS integration fft benchmark)
add_ippl_integration_test(TestNUFFTParamSweep LABELS integration fft benchmark)
add_ippl_integration_test(TestScatterGatherKernel LABELS integration fft)
add_ippl_integration_test(ScatterDebug LABELS integration fft)
add_ippl_test(DebugFFT)

if(TARGET finufft OR TARGET cufinufft)
    target_link_libraries(TestNUFFTPerformance PRIVATE finufft cufinufft)
    target_link_libraries(TestNUFFT2_new PRIVATE finufft cufinufft)
    target_link_libraries(TestNUFFT2_old PRIVATE finufft cufinufft)
    target_link_libraries(TestNUFFT1_new PRIVATE finufft cufinufft)
    target_link_libraries(TestNUFFT1_compare PRIVATE finufft cufinufft)
    target_link_libraries(TestNUFFT2 PRIVATE finufft cufinufft)
    target_link_libraries(TestNUFFTParamSweep PRIVATE finufft cufinufft)
    target_link_libraries(TestNUFFT1 PRIVATE finufft cufinufft)
endif()


# Paper benchmarks live in their own subdirectory
add_subdirectory(PaperBenchmarks)