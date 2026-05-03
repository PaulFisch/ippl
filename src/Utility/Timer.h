//
// Class Timer
//   This class is used in IpplTimings.
//
//
//
#ifndef IPPL_TIMER_H
#define IPPL_TIMER_H

#ifndef IPPL_ENABLE_TIMER_FENCES
// Soften from #warning to #pragma message — Timer.h is transitively included
// from many TUs and the warning was shouting on every translation unit.
#pragma message("IPPL_ENABLE_TIMER_FENCES not set by CMake; defaulting to false.")
#define IPPL_ENABLE_TIMER_FENCES false
#endif

#include <cstdint>

class Timer {
public:
    static bool enableFences;

    Timer();

    void clear();  // Set all accumulated times to 0
    void start();  // Start timer
    void stop();   // Stop timer

    double elapsed();  // Report clock time accumulated in seconds

private:
    double elapsed_m;
    // high_resolution_clock tick counts since epoch — kept as int64 in the
    // header so consumers don't pay for <chrono>'s pull-in of <format>.
    std::int64_t start_m = 0, stop_m = 0;
};

#endif
