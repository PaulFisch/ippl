#ifndef IPPL_SCATTER_H
#define IPPL_SCATTER_H

#include "Utility/Tuning.h"

#include "Interpolation/Binning.h"
#include "Interpolation/Scatter/AtomicScatter.h"
#include "Interpolation/Scatter/GridParallelScatter.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"
#include "Interpolation/Scatter/ScatterConfig.h"
#include "Interpolation/Scatter/TiledScatter.h"
#include "Interpolation/WidthDispatcher.h"
#include "Particle/ParticleAttrib.h"

namespace ippl {

    namespace Interpolation::detail {

        struct UnsortedPolicy {
            static constexpr bool use_sorting      = false;
            static constexpr bool requires_binning = false;
        };

        struct SortedPolicy {
            static constexpr bool use_sorting      = true;
            static constexpr bool requires_binning = true;
        };

        template <typename Kernel, typename FieldType, typename PositionsType, typename ValuesType>
        struct DeduceScatterTypes {
            using FieldTr = ippl::detail::FieldTraits<std::decay_t<FieldType>>;
            using PosTr   = ippl::detail::AttribTraits<std::decay_t<PositionsType>>;
            using ValTr   = ippl::detail::AttribTraits<std::decay_t<ValuesType>>;
            using VecTr   = ippl::detail::VectorTraits<typename PosTr::value_type>;

            using type = ScatterTypes<FieldTr::dim, typename VecTr::real_type, std::decay_t<Kernel>,
                                      typename FieldTr::view_type, typename PosTr::view_type,
                                      typename ValTr::view_type>;
        };

        template <typename Kernel, typename FieldType, typename PositionsType, typename ValuesType>
        using DeducedScatterTypes =
            typename DeduceScatterTypes<Kernel, FieldType, PositionsType, ValuesType>::type;

        // ─────────────────────────────────────────────────────────────────────────
        // Tuner registry - one tuner per (method, Dim, RealType, is_complex) combo
        // ─────────────────────────────────────────────────────────────────────────
        template <template <int, class, class> class Impl, unsigned Dim, typename RealType,
                  bool IsComplex>
        TileSizeTuner<Dim, Vector<int, Dim>>& get_scatter_tuner() {
            static TileSizeTuner<Dim, Vector<int, Dim>> tuner;
            return tuner;
        }

    }  // namespace Interpolation::detail

    template <typename Kernel, unsigned Dim>
    class Scatter {
    public:
        Scatter(const Kernel& kernel, const Interpolation::ScatterConfig<Dim>& config = {})
            : kernel_m(kernel)
            , config_m(config) {}

        template <typename ValueT, typename FieldT, class Mesh, class Centering, class... ViewArgs,
                  typename ParticleT, class... PosProps, class... ValProps>
        void operator()(Field<FieldT, Dim, Mesh, Centering, ViewArgs...>& field,
                        const ParticleAttrib<Vector<ParticleT, Dim>, PosProps...>& positions,
                        const ParticleAttrib<ValueT, ValProps...>& values) {
            using Types =
                Interpolation::detail::DeducedScatterTypes<Kernel, decltype(field),
                                                           decltype(positions), decltype(values)>;

            const auto method = config_m.method;

            const bool run_atomic =
                (method == Interpolation::ScatterMethod::Atomic)
                || (method == Interpolation::ScatterMethod::OutputFocusedZBatch);

            if (run_atomic) {
                if (config_m.sort) {
                    dispatch<Interpolation::detail::AtomicScatter, Types,
                             Interpolation::detail::SortedPolicy>(field, positions, values);
                } else {
                    dispatch<Interpolation::detail::AtomicScatter, Types,
                             Interpolation::detail::UnsortedPolicy>(field, positions, values);
                }
                return;
            }

            if (method == Interpolation::ScatterMethod::Tiled) {
                dispatch<Interpolation::detail::TiledScatter, Types,
                         Interpolation::detail::SortedPolicy>(field, positions, values);
                return;
            }

            if (method == Interpolation::ScatterMethod::OutputFocused) {
                dispatch<Interpolation::detail::GridParallelScatter, Types,
                         Interpolation::detail::SortedPolicy>(field, positions, values);
            }
        }

    private:
        template <template <int, class, class> class Impl, class Types, class Policy, class Field,
                  class Positions, class Values>
        void dispatch(Field& field, const Positions& positions, const Values& values) {
            using memory_space    = typename Types::memory_space;
            // using execution_space = typename Types::execution_space;
            using RealType        = typename Types::RealType;
            using grid_value_t    = typename Field::value_type;

            constexpr bool is_complex = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

            const int width          = kernel_m.width();
            const size_t n_particles = positions.getParticleCount();

            // Determine tile size (with optional tuning)
            Vector<int, Dim> tile_size = config_m.get_tile_size();

            Interpolation::WidthDispatcher<1, 14>::dispatch(width, [&]<int W>() {
                // Initialize tuner and get tile size if method requires binning
                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    if (config_m.enable_tuning) {
                        tile_size = get_tuned_tile_size<Impl, W, Types, Policy, is_complex>(
                            field, tile_size);
                    }
                }

                // Create a mutable config copy with the (possibly tuned) tile size
                auto tuned_config = config_m;
                tuned_config.set_tile_size(tile_size);

                // Perform binning with the tile size
                Interpolation::detail::BinningResult<Dim, memory_space> binning;
                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    binning = performBinning<Types>(positions, field, tile_size);
                } else if (config_m.do_binning()) {
                    binning = performBinning<Types>(positions, field, tile_size);
                }

                // Create and run functor
                auto args = Impl<W, Types, Policy>::Arguments::create(
                    field, positions, values, kernel_m, tuned_config, binning);

                Impl<W, Types, Policy> functor{std::move(args)};

                field = 0.0;
                functor.run(n_particles);
                Kokkos::fence();

                // End tuning context
                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    if (config_m.enable_tuning) {
                        auto& tuner = Interpolation::detail::get_scatter_tuner<Impl, Dim, RealType,
                                                                               is_complex>();
                        tuner.end();
                    }
                }

                field.accumulateHalo();
            });
        }

        template <template <int, class, class> class Impl, int W, class Types, class Policy,
                  bool IsComplex, class Field>
        Vector<int, Dim> get_tuned_tile_size(const Field& /*field*/,
                                             const Vector<int, Dim>& default_tile) {
            using RealType        = typename Types::RealType;
            using execution_space = typename Types::execution_space;
            using team_policy     = Kokkos::TeamPolicy<execution_space>;

            auto& tuner =
                Interpolation::detail::get_scatter_tuner<Impl, Dim, RealType, IsComplex>();

            if (!tuner.is_initialized()) {
                const size_t max_scratch = team_policy(1, config_m.team_size).scratch_size_max(0);

                std::vector<int> candidates = {1, 2, 3, 4, 8, 16, 32};

                auto scratch_calc = [](const Vector<int, Dim>& tile) {
                    return Impl<W, Types, Policy>::template compute_scratch_size<IsComplex>(tile);
                };

                tuner.initialize("Scatter_" + std::string(typeid(Impl<W, Types, Policy>).name()),
                                 candidates, max_scratch, scratch_calc, default_tile);
            }

            return tuner.begin();
        }

        template <typename Types, typename Positions, typename Field>
        auto performBinning(const Positions& positions, const Field& field,
                            const Vector<int, Dim>& tile_size) {
            using memory_space = typename Types::memory_space;

            auto [permute, bin_offsets, num_tiles] = Interpolation::detail::bin_particles(
                positions, field.getLayout(), field.get_mesh(), tile_size, kernel_m.width());

            return Interpolation::detail::BinningResult<Dim, memory_space>{permute, bin_offsets,
                                                                           num_tiles};
        }

        Kernel kernel_m;
        Interpolation::ScatterConfig<Dim> config_m;
    };

}  // namespace ippl

#endif  // IPPL_SCATTER_H