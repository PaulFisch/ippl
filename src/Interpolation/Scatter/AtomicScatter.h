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

    // template <int W, class Types, class Policy>
    // struct AtomicScatter {
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
    //     // Total number of stencil points = W^Dim
    //     static constexpr int total_stencil_points = []() {
    //         int result = 1;
    //         for (unsigned d = 0; d < Dim; ++d) result *= W;
    //         return result;
    //     }();
    //
    //     // Scratch memory views
    //     using ScratchBaseView   = Kokkos::View<int[Dim], typename
    //     execution_space::scratch_memory_space,
    //                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    //     using ScratchWeightView = Kokkos::View<RealType[Dim][W], typename
    //     execution_space::scratch_memory_space,
    //                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    //     using ScratchValueView  = Kokkos::View<ValueType, typename
    //     execution_space::scratch_memory_space,
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
    //     // Compute scratch memory size per team
    //     template <bool IsComplex>
    //     static size_t compute_scratch_size(const Vector<int, Dim>& /*tile_size*/) {
    //         return sizeof(int) * Dim + 8
    //              + sizeof(RealType) * Dim * W
    //              + sizeof(ValueType) + 12;
    //     }
    //
    //     Arguments args;
    //     int team_size_;
    //
    //     AtomicScatter(const Arguments& a)
    //         : args(a), team_size_(32) {}
    //
    //     // Convert linear stencil index to multi-dimensional indices
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
    //     // Compute product of weights for a given stencil point
    //     KOKKOS_INLINE_FUNCTION
    //     RealType compute_weight_product(const ScratchWeightView& kw,
    //                                     const Kokkos::Array<int, Dim>& stencil_idx) const {
    //         RealType w = RealType(1);
    //         for (unsigned d = 0; d < Dim; ++d) {
    //             w *= kw(d, stencil_idx[d]);
    //         }
    //         return w;
    //     }
    //
    //     KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
    //         using grid_value_t = decltype(args.grid)::non_const_value_type;
    //
    //         const size_t j = team.league_rank();
    //
    //         // Get scratch memory
    //         ScratchBaseView base(team.team_scratch(0));
    //         ScratchWeightView kw(team.team_scratch(0));
    //         ScratchValueView val_scratch(team.team_scratch(0));
    //
    //         // Team leader (rank 0) loads particle data and computes stencil info
    //         if (team.team_rank() == 0) {
    //             size_t p = j;
    //             if constexpr (Policy::use_sorting) {
    //                 p = args.permute(j);
    //             }
    //
    //             // Store particle value in shared memory
    //             val_scratch() = args.values(p);
    //
    //             CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
    //             args.n_grid};
    //
    //             // Compute and store stencil base indices and kernel weights
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
    //         }
    //
    //         // Synchronize so all team members see the shared data
    //         team.team_barrier();
    //
    //         // Read particle value from shared memory (avoid register spill)
    //         const auto val = val_scratch();
    //         auto grid      = args.grid;
    //
    //         // Distribute stencil points across team members
    //         Kokkos::parallel_for(
    //             Kokkos::TeamThreadRange(team, total_stencil_points),
    //             [&](const int stencil_linear_idx) {
    //                 // Convert linear index to multi-dimensional stencil indices
    //                 const auto stencil_idx = linear_to_multi(stencil_linear_idx);
    //
    //                 // Compute weight product from shared memory
    //                 const RealType w = compute_weight_product(kw, stencil_idx);
    //
    //                 // Compute grid indices and perform atomic scatter
    //                 if constexpr (Dim == 1) {
    //                     const int i0 = base(0) + stencil_idx[0];
    //                     Kokkos::atomic_add(&grid(i0),
    //                                        static_cast<grid_value_t>(val * w));
    //                 } else if constexpr (Dim == 2) {
    //                     const int i0 = base(0) + stencil_idx[0];
    //                     const int i1 = base(1) + stencil_idx[1];
    //                     Kokkos::atomic_add(&grid(i0, i1),
    //                                        static_cast<grid_value_t>(val * w));
    //                 } else if constexpr (Dim == 3) {
    //                     const int i0 = base(0) + stencil_idx[0];
    //                     const int i1 = base(1) + stencil_idx[1];
    //                     const int i2 = base(2) + stencil_idx[2];
    //                     Kokkos::atomic_add(&grid(i0, i1, i2),
    //                                        static_cast<grid_value_t>(val * w));
    //                 }
    //             });
    //     }
    //
    //     void run(size_t n_particles) {
    //         // Determine team size based on stencil points and hardware
    //         const int team_size = 32;
    //
    //         const size_t scratch_size = compute_scratch_size<false>(Vector<int, Dim>{});
    //
    //         team_policy policy(n_particles, team_size);
    //         policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
    //
    //         Kokkos::parallel_for("TeamAtomicScatter", policy, *this);
    //     }
    // };

    template <int W, class Types, class Policy>
    struct AtomicScatter {
        static constexpr bool requires_binning = false;
        static constexpr unsigned Dim          = Types::Dim;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy = Kokkos::TeamPolicy<execution_space>;
        using team_member = typename team_policy::member_type;

        using scratch_space = typename execution_space::scratch_memory_space;
        using unmanaged     = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

        // Total number of stencil points = W^Dim
        static constexpr int total_stencil_points = []() {
            int result = 1;
            for (unsigned d = 0; d < Dim; ++d)
                result *= W;
            return result;
        }();

        // Warp size (vector length)
        static constexpr int vector_length = 32;

        // Scratch view types for N particles
        // base[N][Dim], weights[N][Dim][W], values[N]
        using ScratchBaseView    = Kokkos::View<int**, scratch_space, unmanaged>;
        using ScratchWeightsView = Kokkos::View<RealType***, scratch_space, unmanaged>;
        using ScratchValuesView  = Kokkos::View<ValueType*, scratch_space, unmanaged>;

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

        // Compute scratch size for N particles per team
        template <bool>
        static size_t compute_scratch_size(int particles_per_team) {
            return ScratchBaseView::shmem_size(particles_per_team, Dim)
                   + ScratchWeightsView::shmem_size(particles_per_team, Dim, W)
                   + ScratchValuesView::shmem_size(particles_per_team);
        }

        Arguments args;
        int particles_per_team_;

        AtomicScatter(const Arguments& a)
            : args(a)
            , particles_per_team_(4) {}

        // Convert linear stencil index to multi-dimensional indices
        KOKKOS_INLINE_FUNCTION Kokkos::Array<int, Dim> linear_to_multi(int linear_idx) const {
            Kokkos::Array<int, Dim> idx;
            for (unsigned d = 0; d < Dim; ++d) {
                idx[d] = linear_idx % W;
                linear_idx /= W;
            }
            return idx;
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t = typename decltype(args.grid)::non_const_value_type;

            const int team_rank = team.team_rank();
            const int team_size = team.team_size();

            // Allocate scratch views
            ScratchBaseView base(team.team_scratch(0), team_size, Dim);
            ScratchWeightsView kw(team.team_scratch(0), team_size, Dim, W);
            ScratchValuesView values(team.team_scratch(0), team_size);

            // Global particle index for this team member
            const size_t base_particle = team.league_rank() * team_size;
            const size_t p_global      = base_particle + team_rank;

            // Each team member loads its particle (only vector lane 0)
            if (p_global < args.n_particles) {
                size_t p = p_global;
                if constexpr (Policy::use_sorting) {
                    p = args.permute(p_global);
                }
                Kokkos::single(Kokkos::PerThread(team), [&]() {
                    values(team_rank) = args.values(p);
                });

                CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};

                // for (unsigned d = 0; d < Dim; ++d) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, Dim * W), [&](const int idx_linear) {
                        int d = idx_linear % Dim;
                        int i = idx_linear / Dim;
                        const RealType g_pos = transform.toGridCoordinate(args.x(p)[d], d);
                        const int idx0       = transform.getStencilBase(g_pos, W);

                        base(team_rank, d) = idx0 - args.local_offset[d] + args.nghost;

                        // for (int i = 0; i < W; ++i) {
                        if constexpr (Types::KernelType::has_width_template) {
                            kw(team_rank, d, i) =
                                args.kernel.eval<W>((g_pos - RealType(idx0 + i)) * args.inv_hw);
                        } else {
                            kw(team_rank, d, i) =
                                args.kernel((g_pos - RealType(idx0 + i)) * args.inv_hw);
                        }
                        // }
                        // }
                    });
            }

            // Synchronize so all threads see the shared data
            team.team_barrier();

            // Each team member processes its particle using vector parallelism
            if (p_global < args.n_particles) {
                const ValueType my_val = values(team_rank);
                auto grid              = args.grid;

                // Cache base indices in registers
                int my_base[Dim];
                for (unsigned d = 0; d < Dim; ++d) {
                    my_base[d] = base(team_rank, d);
                }

                // Distribute stencil points across vector lanes
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, total_stencil_points),
                    [&](const int stencil_idx_linear) {
                        const auto stencil_idx = linear_to_multi(stencil_idx_linear);

                        // Compute weight product
                        RealType w = RealType(1);
                        for (unsigned d = 0; d < Dim; ++d) {
                            w *= kw(team_rank, d, stencil_idx[d]);
                        }

                        const auto contribution = static_cast<grid_value_t>(my_val * w);

                        // Perform atomic scatter
                        if constexpr (Dim == 1) {
                            Kokkos::atomic_add(&grid(my_base[0] + stencil_idx[0]), contribution);
                        } else if constexpr (Dim == 2) {
                            Kokkos::atomic_add(
                                &grid(my_base[0] + stencil_idx[0], my_base[1] + stencil_idx[1]),
                                contribution);
                        } else if constexpr (Dim == 3) {
                            Kokkos::atomic_add(
                                &grid(my_base[0] + stencil_idx[0], my_base[1] + stencil_idx[1],
                                      my_base[2] + stencil_idx[2]),
                                contribution);
                        }
                    });
            }
        }

        void run(size_t) {
            const int team_size       = particles_per_team_;
            const size_t scratch_size = compute_scratch_size<true>(team_size);
            const size_t n_teams      = (args.n_particles + team_size - 1) / team_size;

            team_policy policy(n_teams, team_size, vector_length);
            policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

            Kokkos::parallel_for("AtomicScatterVectorized", policy, *this);
        }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_TEAM_ATOMIC_SCATTER_H