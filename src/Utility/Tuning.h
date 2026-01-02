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
    size_t variable_id_  = 0;
    size_t context_id_   = 0;
    size_t default_index_ = 0;
    bool initialized_    = false;
    bool context_active_ = false;

    std::vector<TileConfig> valid_configs_;
    std::vector<int64_t> config_indices_;

public:
    TileSizeTuner() = default;

    template <typename ScratchCalculator>
    void initialize(const std::string& kernel_name,
                    const std::vector<int>& candidates,
                    size_t max_scratch,
                    ScratchCalculator&& calc,
                    const TileConfig& default_tile) {
        if (initialized_) return;

        generate_valid_configs(candidates, max_scratch, std::forward<ScratchCalculator>(calc));

        if (valid_configs_.empty()) {
            // Try smallest possible configuration
            TileConfig min_tile;
            for (unsigned d = 0; d < Dim; ++d) {
                min_tile[d] = candidates.front();
            }
            if (calc(min_tile) <= max_scratch) {
                valid_configs_.push_back(min_tile);
            }
        }

        if (valid_configs_.empty()) {
            Kokkos::abort("TileSizeTuner: No valid tile configuration fits in scratch!");
        }

        default_index_ = find_closest_config(default_tile);

        config_indices_.resize(valid_configs_.size());
        for (size_t i = 0; i < valid_configs_.size(); ++i) {
            config_indices_[i] = static_cast<int64_t>(i);
        }

#if IPPL_TUNING_ENABLED
        if (Kokkos::Tools::Experimental::have_tuning_tool()) {
            using namespace Kokkos::Tools::Experimental;

            VariableInfo info;
            info.type          = ValueType::kokkos_value_int64;
            info.category      = StatisticalCategory::kokkos_value_categorical;
            info.valueQuantity = CandidateValueType::kokkos_value_set;

            SetOrRange cands;
            cands.set       = ValueSet{config_indices_.size(), config_indices_.data()};
            info.candidates = cands;

            variable_id_ = declare_output_type(kernel_name + "_tile_config", info);
        }
#endif

        initialized_ = true;
    }

    bool is_initialized() const { return initialized_; }
    size_t num_configurations() const { return valid_configs_.size(); }

    // Call before binning/kernel - returns tile configuration to use
    TileConfig begin() {
        if (!initialized_ || valid_configs_.empty()) {
            TileConfig fallback;
            for (unsigned d = 0; d < Dim; ++d) fallback[d] = 1;
            return fallback;
        }

        size_t config_index = default_index_;

#if IPPL_TUNING_ENABLED
        if (Kokkos::Tools::Experimental::have_tuning_tool()) {
            using namespace Kokkos::Tools::Experimental;

            context_id_ = get_new_context_id();
            begin_context(context_id_);
            context_active_ = true;

            VariableValue value = make_variable_value(variable_id_,
                                                       static_cast<int64_t>(default_index_));
            request_output_values(context_id_, 1, &value);

            config_index = static_cast<size_t>(value.value.int_value);
            if (config_index >= valid_configs_.size()) {
                config_index = default_index_;
            }
        }
#endif

        return valid_configs_[config_index];
    }

    // Call after kernel + fence
    void end() {
#if IPPL_TUNING_ENABLED
        if (context_active_ && Kokkos::Tools::Experimental::have_tuning_tool()) {
            Kokkos::Tools::Experimental::end_context(context_id_);
            context_active_ = false;
        }
#endif
    }

private:
    template <typename ScratchCalculator>
    void generate_valid_configs(const std::vector<int>& candidates,
                                size_t max_scratch,
                                ScratchCalculator& calc) {
        TileConfig current;
        generate_recursive(candidates, max_scratch, calc, 0, current);

        std::sort(valid_configs_.begin(), valid_configs_.end(),
                  [](const TileConfig& a, const TileConfig& b) {
                      size_t vol_a = 1, vol_b = 1;
                      for (unsigned d = 0; d < Dim; ++d) {
                          vol_a *= a[d];
                          vol_b *= b[d];
                      }
                      return vol_a < vol_b;
                  });
    }

    template <typename ScratchCalculator>
    void generate_recursive(const std::vector<int>& candidates,
                            size_t max_scratch,
                            ScratchCalculator& calc,
                            unsigned dim,
                            TileConfig& current) {
        if (dim == Dim) {
            if (calc(current) <= max_scratch) {
                valid_configs_.push_back(current);
            }
            return;
        }
        for (int c : candidates) {
            current[dim] = c;
            generate_recursive(candidates, max_scratch, calc, dim + 1, current);
        }
    }

    size_t find_closest_config(const TileConfig& target) const {
        size_t best_idx  = 0;
        int best_dist    = std::numeric_limits<int>::max();

        for (size_t i = 0; i < valid_configs_.size(); ++i) {
            int dist = 0;
            for (unsigned d = 0; d < Dim; ++d) {
                int diff = valid_configs_[i][d] - target[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx  = i;
            }
        }
        return best_idx;
    }
};

}  // namespace ippl

#endif  // IPPL_TUNING_H