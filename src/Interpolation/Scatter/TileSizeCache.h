#ifndef IPPL_TILE_SIZE_CACHE_H
#define IPPL_TILE_SIZE_CACHE_H

// ============================================================================
// TileSizeCache
// ============================================================================
//
// Singleton that loads optimal configurations produced by BenchmarkTileSweep
// and exposes them to Scatter::dispatch at runtime.
//
// Supported CSV files (checked in order):
//   1. $IPPL_TILE_CSV              — explicit path override
//   2. tile_sweep_sa_optimal.csv   — BO-optimised + Atomic results (preferred)
//   3. tile_sweep_optimal.csv      — uniform sweep results (fallback)
//
// ── Formats ──────────────────────────────────────────────────────────────────
//
// Uniform sweep (write_optimal_csv), with rho column:
//   method,kernel_width,rho,optimal_tile_size,throughput_Mpts_s,time_ms
//
// BO/Atomic (write_bo_csv), with rho column:
//   method,value_type,kernel_width,rho,
//   best_tile_x,best_tile_y,best_tile_z,
//   best_team_size,best_oversubscription_factor,best_z_batches,
//   throughput_Mpts_s,time_ms,kernel_evaluations,preflight_rejections
//
// Both formats are also parseable WITHOUT the rho column (old files).
// In that case rho defaults to 0.0 ("unspecified").
//
// ── Density-aware lookup ─────────────────────────────────────────────────────
//
// Multiple rows with the same (method, width, is_complex) but different rho
// values coexist.  get() / get_best() accept an optional rho parameter:
//   rho <= 0  → return the entry with the highest throughput (ignore density)
//   rho >  0  → return the entry whose recorded rho is closest to the query
//               (entries with rho=0 are a low-priority fallback)
//
// ── Method selection ─────────────────────────────────────────────────────────
//
// get_best(width, is_complex, rho) returns the method+config with the highest
// throughput for the query density across Atomic, Tiled, and OutputFocused.
//
// Thread safety: initialised once via std::call_once, then read-only.
// ============================================================================

#include <algorithm>
#include <array>
#include <cmath>
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

// ── Cache key ────────────────────────────────────────────────────────────────
struct TileCacheKey {
    ScatterMethod method;
    int           kernel_width;
    bool          is_complex;

    bool operator==(const TileCacheKey& o) const noexcept {
        return method == o.method && kernel_width == o.kernel_width
               && is_complex == o.is_complex;
    }
};

struct TileCacheKeyHash {
    std::size_t operator()(const TileCacheKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.method);
        h             = h * 31 + static_cast<std::size_t>(k.kernel_width);
        h             = h * 31 + static_cast<std::size_t>(k.is_complex);
        return h;
    }
};

// ── Cache entry ───────────────────────────────────────────────────────────────
struct TileCacheEntry {
    std::array<int, 3> tile     = {1, 1, 1};
    int  team_size              = -1;   // -1 → keep ScatterConfig default
    int  oversubscription_factor = -1;  // -1 → keep ScatterConfig default
    int  z_batches              = -1;   // -1 → keep ScatterConfig default
    bool is_rectangular         = false;
    double throughput_Mpts_s    = 0.0;  // used for best-method selection
    double rho                  = 0.0;  // 0 = unspecified (old CSV, no rho col)
};

// ── Result of get_best() ─────────────────────────────────────────────────────
struct BestCacheEntry {
    ScatterMethod  method;
    TileCacheEntry entry;
};

// ─────────────────────────────────────────────────────────────────────────────
class TileSizeCache {
public:
    static TileSizeCache& instance() {
        static TileSizeCache inst;
        std::call_once(inst.init_flag_, [&]() { inst.load(); });
        return inst;
    }

    // ------------------------------------------------------------------
    // Lookup by explicit method + optional density hint.
    // ------------------------------------------------------------------
    std::optional<TileCacheEntry> get(ScatterMethod method, int kernel_width,
                                      bool is_complex, double rho = 0.0) const {
        TileCacheKey key{method, kernel_width, is_complex};
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.empty())
            return std::nullopt;
        const TileCacheEntry* e = closest_entry(it->second, rho);
        return e ? std::optional<TileCacheEntry>(*e) : std::nullopt;
    }

    // ------------------------------------------------------------------
    // Auto method selection: best throughput across ALL methods at the
    // closest recorded density.
    // ------------------------------------------------------------------
    std::optional<BestCacheEntry> get_best(int kernel_width, bool is_complex,
                                           double rho = 0.0) const {
        std::optional<BestCacheEntry> best;
        double best_tp = -1.0;

        for (const auto& [key, vec] : entries_) {
            if (key.kernel_width != kernel_width || key.is_complex != is_complex)
                continue;
            if (vec.empty())
                continue;
            const TileCacheEntry* e = closest_entry(vec, rho);
            if (e && e->throughput_Mpts_s > best_tp) {
                best_tp = e->throughput_Mpts_s;
                best    = BestCacheEntry{key.method, *e};
            }
        }
        return best;
    }

    // Convenience: tile-only lookup (backward compat).
    template <unsigned Dim>
    std::optional<Vector<int, Dim>> get_tile(ScatterMethod method, int kernel_width,
                                             bool is_complex, double rho = 0.0) const {
        auto entry = get(method, kernel_width, is_complex, rho);
        if (!entry.has_value())
            return std::nullopt;
        const auto& e = entry.value();
        Vector<int, Dim> tile;
        for (unsigned d = 0; d < Dim; ++d)
            tile[d] = e.tile[d < 3 ? d : 2];
        return tile;
    }

    bool loaded() const noexcept { return loaded_; }
    const std::string& source() const noexcept { return source_path_; }

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
    // Key = (method, width, is_complex); value = all density variants.
    std::unordered_map<TileCacheKey, std::vector<TileCacheEntry>, TileCacheKeyHash> entries_;
    bool        loaded_      = false;
    std::string source_path_ = "";

    // ------------------------------------------------------------------
    // Density-aware entry selection.
    //   rho <= 0 → highest throughput among all entries
    //   rho >  0 → entry with closest positive rho; rho=0 entries are fallback
    // ------------------------------------------------------------------
    static const TileCacheEntry* closest_entry(const std::vector<TileCacheEntry>& vec,
                                                double rho) {
        if (vec.empty())
            return nullptr;

        if (rho <= 0.0) {
            return &*std::max_element(vec.begin(), vec.end(),
                [](const TileCacheEntry& a, const TileCacheEntry& b) {
                    return a.throughput_Mpts_s < b.throughput_Mpts_s;
                });
        }

        const TileCacheEntry* best      = nullptr;
        double                best_dist = std::numeric_limits<double>::max();

        for (const auto& e : vec) {
            // rho=0 entries used only when no positive-rho entry exists
            double dist = (e.rho > 0.0) ? std::abs(e.rho - rho) : 1e30;
            if (dist < best_dist) {
                best_dist = dist;
                best      = &e;
            }
        }
        // If only rho=0 entries exist, fall back to highest-throughput among them
        if (best_dist >= 1e29) {
            return &*std::max_element(vec.begin(), vec.end(),
                [](const TileCacheEntry& a, const TileCacheEntry& b) {
                    return a.throughput_Mpts_s < b.throughput_Mpts_s;
                });
        }
        return best;
    }

    // ------------------------------------------------------------------
    // Insert with conflict resolution.
    // Rows whose rho values are within 0.1% are treated as the same density
    // bucket and the higher-throughput entry wins.
    // ------------------------------------------------------------------
    void insert_entry(ScatterMethod method, int width, bool is_complex,
                      TileCacheEntry entry) {
        TileCacheKey key{method, width, is_complex};
        auto&        vec = entries_[key];

        constexpr double rho_tol_frac = 0.001;
        for (auto& existing : vec) {
            double tol = rho_tol_frac * std::max({existing.rho, entry.rho, 1.0});
            if (std::abs(existing.rho - entry.rho) <= tol) {
                // Same density bucket: rectangular beats uniform; higher tp wins
                if (!existing.is_rectangular && entry.is_rectangular)
                    existing = entry;
                else if (existing.is_rectangular == entry.is_rectangular
                         && entry.throughput_Mpts_s > existing.throughput_Mpts_s)
                    existing = entry;
                return;
            }
        }
        vec.push_back(std::move(entry));
    }

    // ------------------------------------------------------------------
    // File discovery
    // ------------------------------------------------------------------
    void load() {
        if (const char* env = std::getenv("IPPL_TILE_CSV")) {
            if (load_file(std::string(env)))
                return;
            std::cerr << "[TileSizeCache] Warning: IPPL_TILE_CSV=" << env
                      << " could not be read, falling back.\n";
        }
        if (load_file("tile_sweep_sa_optimal.csv"))
            return;
        load_file("tile_sweep_optimal.csv");
    }

    // ------------------------------------------------------------------
    // CSV parser: supports both with-rho and without-rho layouts.
    // ------------------------------------------------------------------
    bool load_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            return false;

        std::string line;
        if (!std::getline(f, line))
            return false;

        const bool is_rect    = (line.find("best_tile_x")     != std::string::npos);
        const bool is_uniform = (line.find("optimal_tile_size") != std::string::npos);
        if (!is_rect && !is_uniform)
            return false;

        const bool has_rho = (line.find(",rho,") != std::string::npos);

        int rows = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            rows += is_rect ? (parse_rect_row(line, has_rho)    ? 1 : 0)
                            : (parse_uniform_row(line, has_rho)  ? 1 : 0);
        }
        if (rows == 0)
            return false;

        loaded_      = true;
        source_path_ = path;
        return true;
    }

    // ------------------------------------------------------------------
    // Uniform-format row:
    //   with rho:    method,kernel_width,rho,optimal_tile,throughput,time
    //   without rho: method,kernel_width,optimal_tile,throughput,time
    // ------------------------------------------------------------------
    bool parse_uniform_row(const std::string& line, bool has_rho) {
        auto fields = split_csv(line);
        // Need at least: method, width, [rho,] tile
        const int min_f = has_rho ? 4 : 3;
        if ((int)fields.size() < min_f)
            return false;

        ScatterMethod method;
        if (!parse_method(fields[0], method))
            return false;

        int    width    = parse_int(fields[1]);
        double rho      = has_rho ? parse_double(fields[2]) : 0.0;
        int    tile_col = has_rho ? 3 : 2;
        int    tp_col   = tile_col + 1;

        int    tile       = parse_int(fields[tile_col]);
        double throughput = ((int)fields.size() > tp_col) ? parse_double(fields[tp_col]) : 0.0;

        if (width <= 0 || tile <= 0)
            return false;

        // Uniform has no value_type column → insert for both
        for (bool cx : {false, true}) {
            TileCacheEntry e;
            e.tile.fill(tile);
            e.is_rectangular      = false;
            e.team_size           = -1;
            e.oversubscription_factor = -1;
            e.z_batches           = -1;
            e.throughput_Mpts_s   = throughput;
            e.rho                 = rho;
            insert_entry(method, width, cx, e);
        }
        return true;
    }

    // ------------------------------------------------------------------
    // BO/Atomic-format row:
    //   with rho:
    //     method,value_type,kernel_width,rho,
    //     tx,ty,tz, team_size,osub,z_batches,
    //     throughput,time,kernel_evals,preflight
    //     col: 0  1          2            3
    //          4  5  6       7     8    9
    //          10  11  12   13
    //   without rho: same with cols from 3 shifted left by 1
    // ------------------------------------------------------------------
    bool parse_rect_row(const std::string& line, bool has_rho) {
        auto fields = split_csv(line);
        // Minimum: through z_batches (col 9 with rho, col 8 without)
        const int min_f = has_rho ? 10 : 9;
        if ((int)fields.size() < min_f)
            return false;

        ScatterMethod method;
        if (!parse_method(fields[0], method))
            return false;

        const std::string& vtype = fields[1];
        int    width    = parse_int(fields[2]);
        double rho      = has_rho ? parse_double(fields[3]) : 0.0;
        int    tc       = has_rho ? 4 : 3;  // tile column start

        if ((int)fields.size() < tc + 6)
            return false;

        int tx     = parse_int(fields[tc]);
        int ty     = parse_int(fields[tc + 1]);
        int tz     = parse_int(fields[tc + 2]);
        int team   = parse_int(fields[tc + 3]);
        int osub   = parse_int(fields[tc + 4]);
        int zb     = parse_int(fields[tc + 5]);
        int tp_col = tc + 6;

        double throughput = ((int)fields.size() > tp_col) ? parse_double(fields[tp_col]) : 0.0;

        if (width <= 0 || tx <= 0 || ty <= 0 || tz <= 0)
            return false;

        bool is_cx = (vtype == "complex");

        auto do_insert = [&](bool cx) {
            TileCacheEntry e;
            e.tile                    = {tx, ty, tz};
            e.is_rectangular          = (tx != ty || ty != tz);
            e.team_size               = (team > 0) ? team : -1;
            e.oversubscription_factor = (osub > 0)  ? osub : -1;
            e.z_batches               = (zb   > 0)  ? zb   : -1;
            e.throughput_Mpts_s       = throughput;
            e.rho                     = rho;
            insert_entry(method, width, cx, e);
        };

        if (vtype.empty()) {
            do_insert(false);
            do_insert(true);
        } else {
            do_insert(is_cx);
        }
        return true;
    }

    // ------------------------------------------------------------------
    static bool parse_method(const std::string& s, ScatterMethod& out) {
        if (s == "Tiled")         { out = ScatterMethod::Tiled;         return true; }
        if (s == "OutputFocused") { out = ScatterMethod::OutputFocused; return true; }
        if (s == "Atomic")        { out = ScatterMethod::Atomic;        return true; }
        return false;
    }
    static int    parse_int(const std::string& s) { try { return std::stoi(s); } catch(...) { return -1;  } }
    static double parse_double(const std::string& s) { try { return std::stod(s); } catch(...) { return 0.0; } }

    static std::vector<std::string> split_csv(const std::string& line) {
        std::vector<std::string> out;
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            auto l = tok.find_first_not_of(" \t\r\n");
            auto r = tok.find_last_not_of(" \t\r\n");
            out.push_back(l == std::string::npos ? "" : tok.substr(l, r - l + 1));
        }
        return out;
    }
};

}  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_TILE_SIZE_CACHE_H