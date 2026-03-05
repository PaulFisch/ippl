#ifndef IPPL_TILE_SIZE_CACHE_H
#define IPPL_TILE_SIZE_CACHE_H

// ============================================================================
// TileSizeCache
// ============================================================================
//
// Singleton that loads the optimal tile sizes produced by BenchmarkTileSweep
// (the *_optimal.csv file) and exposes them to Scatter::dispatch at runtime.
//
// CSV format (written by write_optimal_csv):
//   method,kernel_width,optimal_tile_size,throughput_Mpts_s,time_ms
//   Tiled,4,3,1234.5,6.789
//   OutputFocused,4,2,1100.0,7.5
//   ...
//
// For the rectangular-tile extension the same file is augmented by the BO
// optimiser (write_sa_csv), whose format is:
//   method,value_type,kernel_width,best_tile_x,best_tile_y,best_tile_z,
//   best_team_size,best_oversubscription_factor,throughput_Mpts_s,time_ms,evaluations
//
// TileSizeCache tries the rectangular file first, falls back to the uniform
// file, and finally falls back to the default tile size baked into ScatterConfig.
//
// Environment / file discovery (checked in order):
//   1. IPPL_TILE_CSV=<path>   — explicit override
//   2. ./tile_sweep_sa_optimal.csv   (rectangular BO results)
//   3. ./tile_sweep_optimal.csv      (uniform sweep results)
//
// Thread safety: the singleton is initialised once (via std::call_once) and
// then read-only.  Safe for use from multiple OpenMP threads / MPI ranks.
// ============================================================================

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Interpolation/Scatter/ScatterConfig.h"

namespace ippl {
namespace Interpolation {

// Key for a cache lookup: (method, kernel_width, is_complex)
struct TileCacheKey {
    ScatterMethod method;
    int kernel_width;
    bool is_complex;

    bool operator==(const TileCacheKey& o) const noexcept {
        return method == o.method && kernel_width == o.kernel_width
               && is_complex == o.is_complex;
    }
};

struct TileCacheKeyHash {
    std::size_t operator()(const TileCacheKey& k) const noexcept {
        // Simple polynomial hash
        std::size_t h = static_cast<std::size_t>(k.method);
        h             = h * 31 + static_cast<std::size_t>(k.kernel_width);
        h             = h * 31 + static_cast<std::size_t>(k.is_complex);
        return h;
    }
};

// Value: per-dimension tile sizes plus team/oversubscription parameters.
// Lower dimensions use the first N entries of `tile`.
struct TileCacheEntry {
    std::array<int, 3> tile    = {1, 1, 1};
    int team_size              = -1;   // -1 → not present in CSV, keep ScatterConfig default
    int oversubscription_factor = -1;  // -1 → not present in CSV, keep ScatterConfig default
    bool is_rectangular        = false;  // true if tile_x != tile_y or tile_y != tile_z
};

class TileSizeCache {
public:
    // ------------------------------------------------------------------
    // Singleton access
    // ------------------------------------------------------------------
    static TileSizeCache& instance() {
        static TileSizeCache inst;
        std::call_once(inst.init_flag_, [&]() { inst.load(); });
        return inst;
    }

    // ------------------------------------------------------------------
    // Lookup: returns the full cache entry, or std::nullopt if not found.
    // Callers should fall back to ScatterConfig defaults on nullopt.
    // ------------------------------------------------------------------
    std::optional<TileCacheEntry> get(ScatterMethod method, int kernel_width,
                                      bool is_complex) const {
        TileCacheKey key{method, kernel_width, is_complex};
        auto it = entries_.find(key);
        if (it == entries_.end())
            return std::nullopt;
        return it->second;
    }

    // Convenience overload: returns only the tile sizes as a Vector<int, Dim>.
    // Kept for backward compatibility with any existing callers.
    template <unsigned Dim>
    std::optional<Vector<int, Dim>> get_tile(ScatterMethod method, int kernel_width,
                                             bool is_complex) const {
        auto entry = get(method, kernel_width, is_complex);
        if (!entry.has_value())
            return std::nullopt;

        const auto& e = entry.value();
        Vector<int, Dim> tile;
        for (unsigned d = 0; d < Dim; ++d)
            tile[d] = e.tile[d < 3 ? d : 2];  // clamp to available dimensions
        return tile;
    }

    bool loaded() const noexcept { return loaded_; }
    const std::string& source() const noexcept { return source_path_; }

    // ------------------------------------------------------------------
    // Allow explicit reload (useful for testing)
    // ------------------------------------------------------------------
    void reload(const std::string& path = "") {
        entries_.clear();
        loaded_      = false;
        source_path_ = "";
        if (!path.empty())
            load_file(path);
        else
            load();
    }

private:
    TileSizeCache() = default;

    std::once_flag init_flag_;
    std::unordered_map<TileCacheKey, TileCacheEntry, TileCacheKeyHash> entries_;
    bool loaded_             = false;
    std::string source_path_ = "";

    // ------------------------------------------------------------------
    // Discover and load the CSV
    // ------------------------------------------------------------------
    void load() {
        // 1. Explicit environment variable override
        if (const char* env = std::getenv("IPPL_TILE_CSV")) {
            if (load_file(std::string(env)))
                return;
            std::cerr << "[TileSizeCache] Warning: IPPL_TILE_CSV=" << env
                      << " could not be read, falling back to defaults.\n";
        }

        // 2. Rectangular BO results (preferred — more specific)
        if (load_file("tile_sweep_sa_optimal.csv"))
            return;

        // 3. Uniform sweep results
        if (load_file("tile_sweep_optimal.csv"))
            return;

        // 4. Nothing found — silent; Scatter will use ScatterConfig defaults.
    }

    // ------------------------------------------------------------------
    // Parse a single CSV file.
    // Supports both formats:
    //
    //   Uniform format (write_optimal_csv):
    //     method,kernel_width,optimal_tile_size,throughput_Mpts_s,time_ms
    //
    //   Rectangular format (write_sa_csv):
    //     method,value_type,kernel_width,best_tile_x,best_tile_y,best_tile_z,
    //     best_team_size,best_oversubscription_factor,throughput_Mpts_s,time_ms,evaluations
    //
    // Returns true if the file was found and at least one row was parsed.
    // ------------------------------------------------------------------
    bool load_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            return false;

        std::string line;
        // Read header and detect format
        if (!std::getline(f, line))
            return false;

        // Detect by presence of "best_tile_x" (rectangular) or "optimal_tile_size" (uniform)
        const bool is_rect    = (line.find("best_tile_x") != std::string::npos);
        const bool is_uniform = (line.find("optimal_tile_size") != std::string::npos);

        if (!is_rect && !is_uniform) {
            // Unknown format — silently skip rather than crash.
            return false;
        }

        int rows_parsed = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            if (is_rect)
                rows_parsed += parse_rect_row(line) ? 1 : 0;
            else
                rows_parsed += parse_uniform_row(line) ? 1 : 0;
        }

        if (rows_parsed == 0)
            return false;

        loaded_      = true;
        source_path_ = path;
        return true;
    }

    // ------------------------------------------------------------------
    // Parse one row of the uniform format:
    //   method,kernel_width,optimal_tile_size,throughput_Mpts_s,time_ms
    // ------------------------------------------------------------------
    bool parse_uniform_row(const std::string& line) {
        std::vector<std::string> fields = split_csv(line);
        // Expect at least 3 usable columns (method, width, tile)
        if (fields.size() < 3)
            return false;

        const std::string& method_str = fields[0];
        int width                     = parse_int(fields[1]);
        int tile                      = parse_int(fields[2]);

        if (width <= 0 || tile <= 0)
            return false;

        ScatterMethod method;
        if (!parse_method(method_str, method))
            return false;

        // Uniform CSV has no value_type column — insert entries for both
        for (bool is_complex : {false, true}) {
            TileCacheKey key{method, width, is_complex};
            TileCacheEntry entry;
            entry.tile.fill(tile);
            entry.is_rectangular        = false;
            entry.team_size             = -1;  // not available in uniform format
            entry.oversubscription_factor = -1;
            entries_[key]               = entry;
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Parse one row of the rectangular BO format:
    //   method,value_type,kernel_width,
    //   best_tile_x,best_tile_y,best_tile_z,
    //   best_team_size,best_oversubscription_factor,
    //   throughput_Mpts_s,time_ms,evaluations
    //
    // Indices:  0        1           2
    //           3        4           5
    //           6        7
    //           8        9           10
    // ------------------------------------------------------------------
    bool parse_rect_row(const std::string& line) {
        std::vector<std::string> fields = split_csv(line);
        // Need at least columns 0-7 (8 fields); columns 8-10 are optional
        if (fields.size() < 8)
            return false;

        const std::string& method_str     = fields[0];
        const std::string& value_type_str = fields[1];
        int width                         = parse_int(fields[2]);
        int tx                            = parse_int(fields[3]);
        int ty                            = parse_int(fields[4]);
        int tz                            = parse_int(fields[5]);
        int team_size                     = parse_int(fields[6]);
        int oversubscription              = parse_int(fields[7]);

        if (width <= 0 || tx <= 0 || ty <= 0 || tz <= 0)
            return false;
        // team_size and oversubscription are allowed to be -1 (invalid → keep default)

        ScatterMethod method;
        if (!parse_method(method_str, method))
            return false;

        // value_type: "complex" → is_complex=true, "real" → false, "" → both
        bool is_complex = (value_type_str == "complex");

        auto insert_entry = [&](bool complex) {
            TileCacheKey key{method, width, complex};
            TileCacheEntry entry;
            entry.tile                  = {tx, ty, tz};
            entry.is_rectangular        = (tx != ty || ty != tz);
            entry.team_size             = (team_size > 0) ? team_size : -1;
            entry.oversubscription_factor =
                (oversubscription > 0) ? oversubscription : -1;

            // Prefer rectangular over uniform: only overwrite if new entry is
            // more specific (rectangular beats uniform, same-type beats wildcard)
            auto it = entries_.find(key);
            if (it == entries_.end() || !it->second.is_rectangular || entry.is_rectangular) {
                entries_[key] = entry;
            }
        };

        if (value_type_str.empty()) {
            insert_entry(false);
            insert_entry(true);
        } else {
            insert_entry(is_complex);
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    static bool parse_method(const std::string& s, ScatterMethod& out) {
        if (s == "Tiled") {
            out = ScatterMethod::Tiled;
            return true;
        }
        if (s == "OutputFocused") {
            out = ScatterMethod::OutputFocused;
            return true;
        }
        return false;
    }

    static int parse_int(const std::string& s) {
        try {
            return std::stoi(s);
        } catch (...) {
            return -1;
        }
    }

    static std::vector<std::string> split_csv(const std::string& line) {
        std::vector<std::string> out;
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            auto l = tok.find_first_not_of(" \t\r\n");
            auto r = tok.find_last_not_of(" \t\r\n");
            out.push_back((l == std::string::npos) ? "" : tok.substr(l, r - l + 1));
        }
        return out;
    }
};

}  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_TILE_SIZE_CACHE_H