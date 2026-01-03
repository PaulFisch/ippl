file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

add_ippl_test(FFT)
add_ippl_test(NUFFT LINK_FINUFFT)
add_ippl_test(NUFFTAccuracy LINK_FINUFFT)
