#ifndef IPPL_ATOMIC_SCATTER_H
#define IPPL_ATOMIC_SCATTER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

namespace ippl::Interpolation::detail {

    template <int W, class Types, bool UseSorting = false>
    struct AtomicScatter {
        static constexpr bool requires_binning = UseSorting;
        static constexpr unsigned Dim          = Types::Dim;

        using RealType        = Types::RealType;
        using ValueType       = Types::ValueType;
        using memory_space    = Types::memory_space;
        using execution_space = Types::execution_space;

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            using PermuteView = Kokkos::View<uint64_t*, memory_space>;
            PermuteView permute;  // Only used when UseSorting = true

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                    const Kernel& k, const ScatterConfig<Dim>&,
                                    const BinningResult<Dim, memory_space>& binning = {}) {
                Arguments a;
                a.initBase(field, pos, vals, k);
                if constexpr (UseSorting) {
                    a.permute = binning.permute;
                }
                return a;
            }
        };

        Arguments args;

        struct Stencil {
            Kokkos::Array<int, Dim> base;
            Kokkos::Array<Kokkos::Array<RealType, W>, Dim> kw;
        };

        KOKKOS_INLINE_FUNCTION void operator()(size_t j) const {
            using grid_value_t = typename decltype(args.grid)::non_const_value_type;

            // Potentially read permutation
            const size_t p = UseSorting ? args.permute(j) : j;

            const auto val = args.values(p);
            auto grid      = args.grid;

            CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};
            Stencil stencil{};

            // Build stencil
            for_constexpr(std::make_integer_sequence<int, Dim>{}, [&]<int d> {
                const RealType g_pos = transform.toGridCoordinate(args.x(p)[d], d);
                const int idx0       = transform.getStencilBase(g_pos, W);

                stencil.base[d] = idx0 - args.local_offset[d] + args.nghost;

                auto& kernel_vals = stencil.kw[d];
                for (int i = 0; i < W; ++i) {
                    kernel_vals[i] = args.kernel((g_pos - RealType(idx0 + i)) * args.inv_hw);
                }
            });

            // Scatter recursion
            auto rec = [&]<unsigned D>(auto&& self, RealType wprod, auto... idx) {
                const int bD   = stencil.base[D];
                const auto& kD = stencil.kw[D];

                for (int i = 0; i < W; ++i) {
                    const RealType w = wprod * kD[i];
                    if constexpr (D == 0) {
                        Kokkos::atomic_add(&grid(bD + i, idx...),
                                           static_cast<grid_value_t>(val * w));
                    } else {
                        self.template operator()<D - 1>(self, w, bD + i, idx...);
                    }
                }
            };
            rec.template operator()<Dim - 1>(rec, RealType(1));
        }

        void run(size_t n_particles) {
            Kokkos::parallel_for("AtomicScatter",
                                 Kokkos::RangePolicy<execution_space>(0, n_particles), *this);
        }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_ATOMIC_SCATTER_H
