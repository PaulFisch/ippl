# -----------------------------------------------------------------------------
# test/Interpolation/CMakeLists.txt
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding Scatter microbench tests from: ${_relPath}")

add_ippl_integration_test(BenchScatterAtomic LABELS integration scatter benchmark)
