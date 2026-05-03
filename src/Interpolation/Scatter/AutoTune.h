#ifndef IPPL_INTERPOLATION_AUTO_TUNE_H
#define IPPL_INTERPOLATION_AUTO_TUNE_H

// ============================================================================
// Width-2 (PIC / CIC) scatter / gather auto-tuning pre-pass.
//
// Default behaviour: NO sweep is run. Sensible per-exec-space defaults are
// seeded into TileSizeCache and GatherCache by Ippl::initialize, so the
// first scatter/gather call uses a known-good configuration without paying
// any benchmarking cost or touching the filesystem.
//
// Opt in to the sweep with `IPPL_AUTO_TUNE=1` in the environment. When
// enabled, runOnFirstUse() benchmarks a small candidate set of
// (tile, team, osub, z_batches) configurations for the Atomic / Tiled /
// OutputFocused scatter implementations at the kernel width PIC actually
// uses (CIC = 2), picks the best per method, and writes a CSV in the
// layout TileSizeCache parses. Progress is reported through ippl::Info at
// info level >= 1 (`--info 1`).
//
// Cost is bounded by design — a fixed 32^3 grid with O(1e5) particles, a
// handful of tile sizes per tunable method. On a modest GPU this is well
// under a second; on Serial we skip the benchmark entirely (there is
// nothing to tune — Atomic with team_size 1 is the only valid config) and
// just write a single Atomic row so subsequent loads are consistent.
// ============================================================================

#include <string>

namespace ippl::Interpolation::AutoTune {

    /// Populate TileSizeCache and GatherCache with built-in per-exec-space
    /// defaults. Called from ippl::initialize before any scatter/gather is
    /// issued, so the first call has a usable configuration without going
    /// through the auto-tune sweep or reading a CSV from cwd.
    /// No-op if a CSV at one of the standard search paths is already loaded.
    void seedBuiltinDefaults();

    /// Run the width-2 scatter/gather auto-tune pre-pass when explicitly
    /// requested via `IPPL_AUTO_TUNE=1`, write the resulting CSVs next to
    /// the running executable, and return true if a CSV exists at the
    /// requested path on exit. When the env var is not set (default),
    /// returns false without doing any work — callers should fall back to
    /// the seeded defaults in TileSizeCache / GatherCache.
    /// Rank 0 owns the writing; all ranks barrier before returning so any
    /// subsequent cache reload is consistent across ranks.
    bool runOnFirstUse(const std::string& output_path = "tile_sweep_sa_optimal.csv");

}  // namespace ippl::Interpolation::AutoTune

#endif  // IPPL_INTERPOLATION_AUTO_TUNE_H
