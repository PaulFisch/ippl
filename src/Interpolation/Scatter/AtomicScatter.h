// #ifndef IPPL_ATOMIC_SCATTER_H
// #define IPPL_ATOMIC_SCATTER_H
//
// #include <Kokkos_Core.hpp>
//
// #include "Interpolation/CoordinateTransform.h"
// #include "Interpolation/Scatter/ScatterArgumentsBase.h"
//
// namespace ippl::Interpolation::detail {
//
//     template <int W, class Types, class Policy>
//     struct AtomicScatter {
//         static constexpr bool requires_binning = Policy::requires_binning;
//         static constexpr unsigned Dim          = Types::Dim;
//
//         using RealType        = Types::RealType;
//         using ValueType       = Types::ValueType;
//         using memory_space    = Types::memory_space;
//         using execution_space = Types::execution_space;
//
//         struct Arguments : ScatterArgumentsBase<Arguments, Types> {
//             using PermuteView = Kokkos::View<uint64_t*, memory_space>;
//             PermuteView permute;  // only used when Policy::use_sorting = true
//
//             template <class Field, class Positions, class Values, class Kernel>
//             static Arguments create(Field& field, const Positions& pos, const Values& vals,
//                                     const Kernel& k, const ScatterConfig<Dim>&,
//                                     const BinningResult<Dim, memory_space>& binning = {}) {
//                 Arguments a;
//                 a.initBase(field, pos, vals, k);
//                 if constexpr (Policy::use_sorting) {
//                     a.permute = binning.permute;
//                 }
//                 return a;
//             }
//         };
//
//         template <bool IsComplex>
//         static size_t compute_scratch_size(const Vector<int, Dim>& /*tile_size*/) {
//             return 0;
//         }
//
//         Arguments args;
//
//         struct Stencil {
//             Kokkos::Array<int, Dim> base;
//             Kokkos::Array<Kokkos::Array<RealType, W>, Dim> kw;
//         };
//
//         KOKKOS_INLINE_FUNCTION void operator()(size_t j) const {
//             using grid_value_t = decltype(args.grid)::non_const_value_type;
//
//             size_t p = j;
//             if constexpr (Policy::use_sorting) {
//                 p = args.permute(j);
//             }
//
//             const auto val = args.values(p);
//             auto grid      = args.grid;
//
//             CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};
//             Stencil stencil{};
//
//             for_constexpr(std::make_integer_sequence<int, Dim>{}, [&]<int d> {
//                 const RealType g_pos = transform.toGridCoordinate(args.x(p)[d], d);
//                 const int idx0       = transform.getStencilBase(g_pos, W);
//
//                 stencil.base[d] = idx0 - args.local_offset[d] + args.nghost;
//
//                 auto& kernel_vals = stencil.kw[d];
//                 for (int i = 0; i < W; ++i) {
//                     kernel_vals[i] = args.kernel((g_pos - RealType(idx0 + i)) * args.inv_hw);
//                 }
//             });
//
//             auto rec = [&]<unsigned D>(auto&& self, RealType wprod, auto... idx) {
//                 const int bD   = stencil.base[D];
//                 const auto& kD = stencil.kw[D];
//
//                 for (int i = 0; i < W; ++i) {
//                     const RealType w = wprod * kD[i];
//                     if constexpr (D == 0) {
//                         Kokkos::atomic_add(&grid(bD + i, idx...),
//                                            static_cast<grid_value_t>(val * w));
//                     } else {
//                         self.template operator()<D - 1>(self, w, bD + i, idx...);
//                     }
//                 }
//             };
//             rec.template operator()<Dim - 1>(rec, RealType(1));
//         }
//
//         void run(size_t n_particles) {
//             Kokkos::parallel_for("AtomicScatter",
//                                  Kokkos::RangePolicy<execution_space>(0, n_particles), *this);
//         }
//     };
//
// }  // namespace ippl::Interpolation::detail
//
// #endif  // IPPL_ATOMIC_SCATTER_H

#ifndef IPPL_TEAM_ATOMIC_SCATTER_H
#define IPPL_TEAM_ATOMIC_SCATTER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

namespace ippl::Interpolation::detail {

    template <int W, class Types, class Policy>
    struct AtomicScatter {
        static constexpr bool requires_binning = Policy::requires_binning;
        static constexpr unsigned Dim          = Types::Dim;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy = Kokkos::TeamPolicy<execution_space>;
        using team_member = typename team_policy::member_type;

        // Total number of stencil points = W^Dim
        static constexpr int total_stencil_points = []() {
            int result = 1;
            for (unsigned d = 0; d < Dim; ++d) result *= W;
            return result;
        }();

        // Scratch memory views
        using ScratchBaseView   = Kokkos::View<int[Dim], typename execution_space::scratch_memory_space,
                                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using ScratchWeightView = Kokkos::View<RealType[Dim][W], typename execution_space::scratch_memory_space,
                                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using ScratchValueView  = Kokkos::View<ValueType, typename execution_space::scratch_memory_space,
                                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            using PermuteView = Kokkos::View<uint64_t*, memory_space>;
            PermuteView permute;

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                    const Kernel& k, const ScatterConfig<Dim>&,
                                    const BinningResult<Dim, memory_space>& binning = {}) {
                Arguments a;
                a.initBase(field, pos, vals, k);
                if constexpr (Policy::use_sorting) {
                    a.permute = binning.permute;
                }
                return a;
            }
        };

        // Compute scratch memory size per team
        template <bool IsComplex>
        static size_t compute_scratch_size(const Vector<int, Dim>& /*tile_size*/) {
            return sizeof(int) * Dim + 8
                 + sizeof(RealType) * Dim * W
                 + sizeof(ValueType) + 12;
        }

        Arguments args;
        int team_size_;

        AtomicScatter(const Arguments& a)
            : args(a), team_size_(32) {}

        // Convert linear stencil index to multi-dimensional indices
        KOKKOS_INLINE_FUNCTION
        Kokkos::Array<int, Dim> linear_to_multi(int linear_idx) const {
            Kokkos::Array<int, Dim> idx;
            for (unsigned d = 0; d < Dim; ++d) {
                idx[d] = linear_idx % W;
                linear_idx /= W;
            }
            return idx;
        }

        // Compute product of weights for a given stencil point
        KOKKOS_INLINE_FUNCTION
        RealType compute_weight_product(const ScratchWeightView& kw,
                                        const Kokkos::Array<int, Dim>& stencil_idx) const {
            RealType w = RealType(1);
            for (unsigned d = 0; d < Dim; ++d) {
                w *= kw(d, stencil_idx[d]);
            }
            return w;
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t = decltype(args.grid)::non_const_value_type;

            const size_t j = team.league_rank();

            // Get scratch memory
            ScratchBaseView base(team.team_scratch(0));
            ScratchWeightView kw(team.team_scratch(0));
            ScratchValueView val_scratch(team.team_scratch(0));

            // Team leader (rank 0) loads particle data and computes stencil info
            if (team.team_rank() == 0) {
                size_t p = j;
                if constexpr (Policy::use_sorting) {
                    p = args.permute(j);
                }

                // Store particle value in shared memory
                val_scratch() = args.values(p);

                CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};

                // Compute and store stencil base indices and kernel weights
                for_constexpr(std::make_integer_sequence<int, Dim>{}, [&]<int d> {
                    const RealType g_pos = transform.toGridCoordinate(args.x(p)[d], d);
                    const int idx0       = transform.getStencilBase(g_pos, W);

                    base(d) = idx0 - args.local_offset[d] + args.nghost;

                    for (int i = 0; i < W; ++i) {
                        kw(d, i) = args.kernel((g_pos - RealType(idx0 + i)) * args.inv_hw);
                    }
                });
            }

            // Synchronize so all team members see the shared data
            team.team_barrier();

            // Read particle value from shared memory (avoid register spill)
            const auto val = val_scratch();
            auto grid      = args.grid;

            // Distribute stencil points across team members
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, total_stencil_points),
                [&](const int stencil_linear_idx) {
                    // Convert linear index to multi-dimensional stencil indices
                    const auto stencil_idx = linear_to_multi(stencil_linear_idx);

                    // Compute weight product from shared memory
                    const RealType w = compute_weight_product(kw, stencil_idx);

                    // Compute grid indices and perform atomic scatter
                    if constexpr (Dim == 1) {
                        const int i0 = base(0) + stencil_idx[0];
                        Kokkos::atomic_add(&grid(i0),
                                           static_cast<grid_value_t>(val * w));
                    } else if constexpr (Dim == 2) {
                        const int i0 = base(0) + stencil_idx[0];
                        const int i1 = base(1) + stencil_idx[1];
                        Kokkos::atomic_add(&grid(i0, i1),
                                           static_cast<grid_value_t>(val * w));
                    } else if constexpr (Dim == 3) {
                        const int i0 = base(0) + stencil_idx[0];
                        const int i1 = base(1) + stencil_idx[1];
                        const int i2 = base(2) + stencil_idx[2];
                        Kokkos::atomic_add(&grid(i0, i1, i2),
                                           static_cast<grid_value_t>(val * w));
                    }
                });
        }

        void run(size_t n_particles) {
            // Determine team size based on stencil points and hardware
            const int team_size = 32;

            const size_t scratch_size = compute_scratch_size<false>(Vector<int, Dim>{});

            team_policy policy(n_particles, team_size);
            policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

            Kokkos::parallel_for("TeamAtomicScatter", policy, *this);
        }
    };

    // // Alternative version with vector-level parallelism for better occupancy
    // template <int W, class Types, class Policy>
    // struct TeamVectorAtomicScatter {
    //     static constexpr bool requires_binning = Policy::requires_binning;
    //     static constexpr unsigned Dim          = Types::Dim;
    //
    //     using RealType        = typename Types::RealType;
    //     using ValueType       = typename Types::ValueType;
    //     using memory_space    = typename Types::memory_space;
    //     using execution_space = typename Types::execution_space;
    //
    //     using team_policy = Kokkos::TeamPolicy<execution_space>;
    //     using team_member = typename team_policy::member_type;
    //
    //     static constexpr int total_stencil_points = []() {
    //         int result = 1;
    //         for (unsigned d = 0; d < Dim; ++d) result *= W;
    //         return result;
    //     }();
    //
    //     // Level 0 scratch: per-team shared data
    //     using ScratchBaseView   = Kokkos::View<int[Dim], typename execution_space::scratch_memory_space,
    //                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    //     using ScratchWeightView = Kokkos::View<RealType[Dim][W], typename execution_space::scratch_memory_space,
    //                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    //     using ScratchValueView  = Kokkos::View<ValueType, typename execution_space::scratch_memory_space,
    //                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    //
    //     struct Arguments : ScatterArgumentsBase<Arguments, Types> {
    //         using PermuteView = Kokkos::View<uint64_t*, memory_space>;
    //         PermuteView permute;
    //
    //         template <class Field, class Positions, class Values, class Kernel>
    //         static Arguments create(Field& field, const Positions& pos, const Values& vals,
    //                                 const Kernel& k, const ScatterConfig<Dim>&,
    //                                 const BinningResult<Dim, memory_space>& binning = {}) {
    //             Arguments a;
    //             a.initBase(field, pos, vals, k);
    //             if constexpr (Policy::use_sorting) {
    //                 a.permute = binning.permute;
    //             }
    //             return a;
    //         }
    //     };
    //
    //     template <bool IsComplex>
    //     static size_t compute_scratch_size(const Vector<int, Dim>& /*tile_size*/) {
    //         return ScratchBaseView::shmem_size()
    //              + ScratchWeightView::shmem_size()
    //              + ScratchValueView::shmem_size();
    //     }
    //
    //     Arguments args;
    //     int particles_per_team_;
    //
    //     TeamVectorAtomicScatter(const Arguments& a, int ppt = 1)
    //         : args(a), particles_per_team_(ppt) {}
    //
    //     KOKKOS_INLINE_FUNCTION
    //     Kokkos::Array<int, Dim> linear_to_multi(int linear_idx) const {
    //         Kokkos::Array<int, Dim> idx;
    //         for (unsigned d = 0; d < Dim; ++d) {
    //             idx[d] = linear_idx % W;
    //             linear_idx /= W;
    //         }
    //         return idx;
    //     }
    //
    //     KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
    //         using grid_value_t = typename decltype(args.grid)::non_const_value_type;
    //
    //         const size_t team_idx     = team.league_rank();
    //         const size_t particle_idx = team_idx;  // One particle per team in this version
    //
    //         // Get scratch memory
    //         ScratchBaseView base(team.team_scratch(0));
    //         ScratchWeightView kw(team.team_scratch(0));
    //         ScratchValueView val_scratch(team.team_scratch(0));
    //
    //         // Single thread loads particle data
    //         Kokkos::single(Kokkos::PerTeam(team), [&]() {
    //             size_t p = particle_idx;
    //             if constexpr (Policy::use_sorting) {
    //                 p = args.permute(particle_idx);
    //             }
    //
    //             val_scratch() = args.values(p);
    //
    //             CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};
    //
    //             for_constexpr(std::make_integer_sequence<int, Dim>{}, [&]<int d> {
    //                 const RealType g_pos = transform.toGridCoordinate(args.x(p)[d], d);
    //                 const int idx0       = transform.getStencilBase(g_pos, W);
    //
    //                 base(d) = idx0 - args.local_offset[d] + args.nghost;
    //
    //                 for (int i = 0; i < W; ++i) {
    //                     kw(d, i) = args.kernel((g_pos - RealType(idx0 + i)) * args.inv_hw);
    //                 }
    //             });
    //         });
    //
    //         team.team_barrier();
    //
    //         const auto val = val_scratch();
    //         auto grid      = args.grid;
    //
    //         // Use TeamThreadRange for outer loop, ThreadVectorRange for inner
    //         // This provides two levels of parallelism
    //         if constexpr (Dim == 3) {
    //             // For 3D: parallelize over W^2 in team, W in vector
    //             constexpr int outer_points = W * W;  // i1, i2 combinations
    //
    //             Kokkos::parallel_for(
    //                 Kokkos::TeamThreadRange(team, outer_points),
    //                 [&](const int outer_idx) {
    //                     const int i1_local = outer_idx / W;
    //                     const int i2_local = outer_idx % W;
    //                     const int i1       = base(1) + i1_local;
    //                     const int i2       = base(2) + i2_local;
    //                     const RealType w12 = kw(1, i1_local) * kw(2, i2_local);
    //
    //                     Kokkos::parallel_for(
    //                         Kokkos::ThreadVectorRange(team, W),
    //                         [&](const int i0_local) {
    //                             const int i0       = base(0) + i0_local;
    //                             const RealType w   = w12 * kw(0, i0_local);
    //                             Kokkos::atomic_add(&grid(i0, i1, i2),
    //                                                static_cast<grid_value_t>(val * w));
    //                         });
    //                 });
    //         } else if constexpr (Dim == 2) {
    //             // For 2D: parallelize over W in team, W in vector
    //             Kokkos::parallel_for(
    //                 Kokkos::TeamThreadRange(team, W),
    //                 [&](const int i1_local) {
    //                     const int i1       = base(1) + i1_local;
    //                     const RealType w1  = kw(1, i1_local);
    //
    //                     Kokkos::parallel_for(
    //                         Kokkos::ThreadVectorRange(team, W),
    //                         [&](const int i0_local) {
    //                             const int i0       = base(0) + i0_local;
    //                             const RealType w   = w1 * kw(0, i0_local);
    //                             Kokkos::atomic_add(&grid(i0, i1),
    //                                                static_cast<grid_value_t>(val * w));
    //                         });
    //                 });
    //         } else {  // Dim == 1
    //             Kokkos::parallel_for(
    //                 Kokkos::TeamThreadRange(team, W),
    //                 [&](const int i0_local) {
    //                     const int i0       = base(0) + i0_local;
    //                     const RealType w   = kw(0, i0_local);
    //                     Kokkos::atomic_add(&grid(i0),
    //                                        static_cast<grid_value_t>(val * w));
    //                 });
    //         }
    //     }
    //
    //     void run(size_t n_particles) {
    //         // Choose team size to cover stencil points efficiently
    //         int team_size = []() {
    //             if constexpr (Dim == 3)
    //                 return W * W;  // Cover W^2 outer iterations
    //             else if constexpr (Dim == 2)
    //                 return W;
    //             else
    //                 return W;
    //         }();
    //
    //         // Vector length for innermost parallelism
    //         constexpr int vector_length = W;
    //
    //         const size_t scratch_size = compute_scratch_size<false>(Vector<int, Dim>{});
    //
    //         team_policy policy(n_particles, team_size, vector_length);
    //         policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
    //
    //         Kokkos::parallel_for("TeamVectorAtomicScatter", policy, *this);
    //     }
    // };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_TEAM_ATOMIC_SCATTER_H