#ifndef IPPL_INTERPOLATION_AUTO_TUNE_H
#define IPPL_INTERPOLATION_AUTO_TUNE_H

// ============================================================================
// Width-2 (PIC / CIC) scatter auto-tuning pre-pass.
//
// On first run with no `tile_sweep_sa_optimal.csv` (and no IPPL_TILE_CSV
// env override), runOnFirstUse() benchmarks a small candidate set of
// (tile, team, osub, z_batches) configurations for the Atomic / Tiled /
// OutputFocused scatter implementations at the kernel width PIC actually
// uses (CIC = 2), picks the best per method, and writes a CSV in the
// layout TileSizeCache parses.
//
// Cost is bounded by design — a fixed 32^3 grid with O(1e5) particles, a
// handful of tile sizes per tunable method. On a modest GPU this is well
// under a second; on Serial we skip the benchmark entirely (there is
// nothing to tune — Atomic with team_size 1 is the only valid config) and
// just write a single Atomic row so subsequent loads are consistent.
//
// IPPL_AUTO_TUNE=0 in the environment opts out.
// ============================================================================

#include <string>

namespace ippl::Interpolation::AutoTune {

    /// Run the width-2 scatter auto-tune pre-pass, write the resulting CSV
    /// next to the running executable, and return true if a CSV exists at
    /// the requested path on exit. Rank 0 owns the writing; all ranks
    /// barrier before returning so subsequent TileSizeCache loads are
    /// consistent across ranks. Idempotent: skips the benchmark if the
    /// file is already present.
    bool runOnFirstUse(const std::string& output_path = "tile_sweep_sa_optimal.csv");

}  // namespace ippl::Interpolation::AutoTune

#endif  // IPPL_INTERPOLATION_AUTO_TUNE_H
