// Interpolation/Scatter/Scatter.h
#ifndef IPPL_SCATTER_H
#define IPPL_SCATTER_H

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
            static constexpr bool requires_binning = false;  // algorithmically required
        };

        struct SortedPolicy {
            static constexpr bool use_sorting      = true;
            static constexpr bool requires_binning = true;  // algorithmically required
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
            using memory_space = typename Types::memory_space;

            Interpolation::detail::BinningResult<Dim, memory_space> binning;

            // If the implementation requires it, bin unconditionally. Otherwise bin if user
            // requested it.
            if constexpr (Impl<1, Types, Policy>::requires_binning) {
                binning = performBinning<Types>(positions, field);
            } else if (config_m.do_binning()) {
                binning = performBinning<Types>(positions, field);
            }

            const int width          = kernel_m.width();
            const size_t n_particles = positions.getParticleCount();

            Interpolation::WidthDispatcher<1, 14>::dispatch(width, [&]<int W>() {
                auto args = Impl<W, Types, Policy>::Arguments::create(field, positions, values,
                                                                      kernel_m, config_m, binning);

                Impl<W, Types, Policy> functor{std::move(args)};

                field = 0.0;
                functor.run(n_particles);
                field.accumulateHalo();
                Kokkos::fence();
            });
        }

        template <typename Types, typename Positions, typename Field>
        auto performBinning(const Positions& positions, const Field& field) {
            using memory_space = typename Types::memory_space;

            auto [permute, bin_offsets, num_tiles] =
                Interpolation::detail::bin_particles(positions, field.getLayout(), field.get_mesh(),
                                                     config_m.get_tile_size(), kernel_m.width());

            return Interpolation::detail::BinningResult<Dim, memory_space>{permute, bin_offsets,
                                                                           num_tiles};
        }

        Kernel kernel_m;
        Interpolation::ScatterConfig<Dim> config_m;
    };

}  // namespace ippl

#endif  // IPPL_SCATTER_H
