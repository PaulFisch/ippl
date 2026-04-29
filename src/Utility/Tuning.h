#ifndef IPPL_TUNING_H
#define IPPL_TUNING_H

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

#include <impl/Kokkos_Profiling_Interface.hpp>
#define IPPL_TUNING_ENABLED 1

namespace ippl {

template <unsigned Dim, typename TileType>
class TileSizeTuner {
public:
    using TileConfig = TileType;

private:
    // Tuning parameters (following Kokkos convention)
    static constexpr double tuning_min  = 0.0;
    static constexpr double tuning_max  = 1.0;
    static constexpr double tuning_step = 0.05;  // 20 steps per dimension

    std::array<size_t, Dim> variable_ids_{};
    size_t context_id_   = 0;
    bool initialized_    = false;
    bool context_active_ = false;

    std::vector<int> candidates_;
    TileConfig default_tile_;
    size_t max_scratch_ = 0;
    std::function<size_t(const TileConfig&)> scratch_calc_;

public:
    TileSizeTuner() = default;

    template <typename ScratchCalculator>
    void initialize(const std::string& kernel_name,
                    const std::vector<int>& candidates,
                    size_t max_scratch,
                    ScratchCalculator&& calc,
                    const TileConfig& default_tile) {
        if (initialized_) return;

        candidates_    = candidates;
        max_scratch_   = max_scratch;
        scratch_calc_  = std::forward<ScratchCalculator>(calc);
        default_tile_  = default_tile;

        // Sort candidates for consistent mapping
        std::sort(candidates_.begin(), candidates_.end());

#if IPPL_TUNING_ENABLED
        if (Kokkos::Tools::Experimental::have_tuning_tool()) {
            using namespace Kokkos::Tools::Experimental;

            for (unsigned d = 0; d < Dim; ++d) {
                VariableInfo info;
                info.type          = ValueType::kokkos_value_double;
                info.category      = StatisticalCategory::kokkos_value_interval;
                info.valueQuantity = CandidateValueType::kokkos_value_range;
                info.candidates    = make_candidate_range(
                    tuning_min, tuning_max, tuning_step, false, false);

                variable_ids_[d] = declare_output_type(
                    kernel_name + "_tile_size_" + std::to_string(d), info);
            }
        }
#endif

        initialized_ = true;
    }

    bool is_initialized() const { return initialized_; }

    // Map normalized value [0,1] to candidate index
    int map_to_candidate(double normalized) const {
        if (candidates_.empty()) return 1;

        // Clamp to [0, 1]
        normalized = std::max(0.0, std::min(1.0, normalized));

        // Map to index
        size_t idx = static_cast<size_t>(normalized * (candidates_.size() - 1) + 0.5);
        idx = std::min(idx, candidates_.size() - 1);

        return candidates_[idx];
    }

    // Map candidate value back to normalized [0,1]
    double map_to_normalized(int candidate) const {
        if (candidates_.size() <= 1) return 0.5;

        auto it = std::lower_bound(candidates_.begin(), candidates_.end(), candidate);
        size_t idx = (it != candidates_.end())
                     ? std::distance(candidates_.begin(), it)
                     : candidates_.size() - 1;

        return static_cast<double>(idx) / (candidates_.size() - 1);
    }

    TileConfig begin() {
        if (!initialized_) {
            return default_tile_;
        }

        TileConfig result = default_tile_;

#if IPPL_TUNING_ENABLED
        if (Kokkos::Tools::Experimental::have_tuning_tool()) {
            using namespace Kokkos::Tools::Experimental;

            context_id_ = get_new_context_id();
            begin_context(context_id_);
            context_active_ = true;

            // Request tuned values for each dimension
            std::array<VariableValue, Dim> values;
            for (unsigned d = 0; d < Dim; ++d) {
                double default_norm = map_to_normalized(default_tile_[d]);
                values[d] = make_variable_value(variable_ids_[d], default_norm);
            }

            request_output_values(context_id_, Dim, values.data());

            // Map normalized values back to tile sizes
            for (unsigned d = 0; d < Dim; ++d) {
                result[d] = map_to_candidate(values[d].value.double_value);
            }

            // Validate scratch constraint - scale down if needed
            result = fit_to_scratch(result);
        }
#endif

        return result;
    }

    void end() {
#if IPPL_TUNING_ENABLED
        if (context_active_ && Kokkos::Tools::Experimental::have_tuning_tool()) {
            Kokkos::Tools::Experimental::end_context(context_id_);
            context_active_ = false;
        }
#endif
    }

private:
    // Scale down proportionally if tile doesn't fit in scratch
    TileConfig fit_to_scratch(const TileConfig& tile) const {
        if (scratch_calc_(tile) <= max_scratch_) {
            return tile;
        }

        // Binary search for largest scale factor that fits
        TileConfig result = tile;
        double lo = 0.0, hi = 1.0;

        for (int iter = 0; iter < 20; ++iter) {
            double mid = (lo + hi) / 2.0;

            TileConfig test;
            for (unsigned d = 0; d < Dim; ++d) {
                int scaled = static_cast<int>(tile[d] * mid);
                test[d] = snap_to_candidate(std::max(1, scaled));
            }

            if (scratch_calc_(test) <= max_scratch_) {
                lo = mid;
                result = test;
            } else {
                hi = mid;
            }
        }

        // Final fallback - minimum tile
        if (scratch_calc_(result) > max_scratch_) {
            for (unsigned d = 0; d < Dim; ++d) {
                result[d] = candidates_.front();
            }
        }

        return result;
    }

    int snap_to_candidate(int value) const {
        if (candidates_.empty()) return value;

        // Find largest candidate <= value
        auto it = std::upper_bound(candidates_.begin(), candidates_.end(), value);
        if (it == candidates_.begin()) {
            return candidates_.front();
        }
        return *std::prev(it);
    }
};

}  // namespace ippl

#endif  // IPPL_TUNING_H