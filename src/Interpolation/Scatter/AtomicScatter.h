#ifndef IPPL_TEAM_ATOMIC_SCATTER_H
#define IPPL_TEAM_ATOMIC_SCATTER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

namespace ippl::Interpolation::detail {
    template <int W, class Types, class Policy>
    struct AtomicScatter {
        static constexpr bool requires_binning = false;
        static constexpr unsigned Dim          = Types::Dim;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy = Kokkos::TeamPolicy<execution_space, Kokkos::LaunchBounds<128, 8>>;
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
        using ScratchBaseView    = Kokkos::View<int**, scratch_space, unmanaged>;
        using ScratchWeightsView = Kokkos::View<RealType***, scratch_space, unmanaged>;
        using ScratchValuesView  = Kokkos::View<ValueType*, scratch_space, unmanaged>;
        using G0View             = Kokkos::View<RealType**, scratch_space, unmanaged>;

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
                   + G0View::shmem_size(particles_per_team, Dim);
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

            const size_t base_particle = size_t(team.league_rank()) * size_t(team_size);
            size_t p_global            = base_particle + size_t(team_rank);
            if (p_global >= args.n_particles)
                return;

            if constexpr (Policy::use_sorting)
                p_global = args.permute(p_global);

            CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};

            // -------------------------
            // base + g0 kept in registers (no scratch / no shared traffic)
            // -------------------------
            const RealType inv_hw = args.inv_hw;
            const auto xp         = args.x(p_global);

            Kokkos::Array<int, Dim> base;
            Kokkos::Array<RealType, Dim> g0;

#pragma unroll
            for (int d = 0; d < Dim; ++d) {
                const RealType g_pos = transform.toGridCoordinate(xp[d], d);
                const int idx0       = transform.getStencilBase(g_pos, W);

                base[d] = idx0 - args.local_offset[d] + args.nghost;
                g0[d]   = (g_pos - RealType(idx0)) * inv_hw;
            }

            // -------------------------
            // Weights in scratch (shared), padded to reduce worst-case bank conflicts
            // when multiple team_ranks are read by a warp (e.g. vector_length==1 setups).
            // Flattened: kw(team_rank, d*W+i)
            // -------------------------
            using ScratchSpace = typename team_member::scratch_memory_space;
            using Unmanaged    = Kokkos::MemoryUnmanaged;

            auto scratch = team.team_scratch(0);

            constexpr int KW_PAD    = 1;  // small, usually enough to break power-of-2 strides
            constexpr int KW_STRIDE = Dim * W + KW_PAD;

            using KwView = Kokkos::View<RealType**, scratch_space, unmanaged>;
            KwView kw(scratch, team_size, KW_STRIDE);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, Dim * W), [&](const int k) {
                const int d = k / W;
                const int i = k - d * W;

                const RealType xi = g0[d] - RealType(i) * inv_hw;

                RealType w;
                if constexpr (Types::KernelType::has_width_template) {
                    w = args.kernel.template eval<W>(xi);
                } else {
                    w = args.kernel(xi);
                }
                kw(team_rank, d * W + i) = w;
            });

            team.team_barrier();  // kw must be visible before stencil

            // -------------------------
            // Stencil: layout-aware decoding so the fastest-varying index matches the
            // contiguous dimension, improving locality / L2 behavior on atomics.
            // -------------------------
            const ValueType my_val = args.values(p_global);
            auto grid              = args.grid;

            constexpr int STENCIL_SIZE  = StaticPow<W, Dim>::value;
            using grid_layout           = typename decltype(args.grid)::array_layout;
            constexpr bool layout_right = std::is_same<grid_layout, Kokkos::LayoutRight>::value;

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, STENCIL_SIZE), [&](const int
                                                                                        flat) {
                if constexpr (Dim == 1) {
                    const int i0     = flat;
                    const RealType w = kw(team_rank, 0 * W + i0);

                    auto& cell = grid(base[0] + i0);
                    Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                       static_cast<grid_value_t>(my_val * w));

                } else if constexpr (Dim == 2) {
                    int i0, i1;
                    if constexpr (layout_right) {  // rightmost index contiguous: make i1 fastest
                        i1 = flat % W;
                        i0 = flat / W;
                    } else {  // leftmost index contiguous: make i0 fastest
                        i0 = flat % W;
                        i1 = flat / W;
                    }

                    const RealType w = kw(team_rank, 0 * W + i0) * kw(team_rank, 1 * W + i1);

                    auto& cell = grid(base[0] + i0, base[1] + i1);
                    Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                       static_cast<grid_value_t>(my_val * w));

                } else if constexpr (Dim == 3) {
                    int i0, i1, i2;
                    if constexpr (layout_right) {  // rightmost index contiguous: i2 fastest
                        i2          = flat % W;
                        const int t = flat / W;
                        i1          = t % W;
                        i0          = t / W;
                    } else {  // leftmost index contiguous: i0 fastest
                        i0          = flat % W;
                        const int t = flat / W;
                        i1          = t % W;
                        i2          = t / W;
                    }

                    const RealType w = kw(team_rank, 0 * W + i0) * kw(team_rank, 1 * W + i1)
                                       * kw(team_rank, 2 * W + i2);

                    auto& cell = grid(base[0] + i0, base[1] + i1, base[2] + i2);
                    Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                       static_cast<grid_value_t>(my_val * w));
                }
            });
        }

        // KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
        //     using grid_value_t = typename decltype(args.grid)::non_const_value_type;
        //
        //     const int team_rank = team.team_rank();
        //     const int team_size = team.team_size();
        //
        //     const size_t base_particle = team.league_rank() * team_size;
        //     size_t p_global            = base_particle + team_rank;
        //     if (p_global >= args.n_particles)
        //         return;
        //
        //     if constexpr (Policy::use_sorting) {
        //         p_global = args.permute(p_global);
        //     }
        //
        //     CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};
        //
        //     // ------------------------------------------------------------
        //     // Scratch allocation (IMPORTANT: reuse the same scratch "allocator")
        //     // ------------------------------------------------------------
        //     auto scratch = team.team_scratch(0);
        //
        //     ScratchBaseView base(scratch, team_size, Dim);      // int  [team_size, Dim]
        //     ScratchWeightsView kw(scratch, team_size, Dim, W);  // Real [team_size, Dim, W]
        //
        //     // Add one small scratch view to avoid recomputing idx0 per i:
        //     using ScratchSpace = typename team_member::scratch_memory_space;
        //     using Unmanaged    = Kokkos::MemoryUnmanaged;
        //     G0View g0(scratch, team_size, Dim);  // Real [team_size, Dim]
        //
        //     // ------------------------------------------------------------
        //     // 1) Precompute base and g0(d) = (g_pos - idx0) * inv_hw in warp-parallel way
        //     //    (Dim is small, but this avoids serial work and avoids redundant recompute
        //     later.)
        //     // ------------------------------------------------------------
        //     Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, Dim), [&](const int d) {
        //         const RealType g_pos = transform.toGridCoordinate(args.x(p_global)[d], d);
        //         const int idx0       = transform.getStencilBase(g_pos, W);
        //
        //         base(team_rank, d) = idx0 - args.local_offset[d] + args.nghost;
        //
        //         // Store scaled offset so weights become: kernel(g0 - i*inv_hw)
        //         g0(team_rank, d) = (g_pos - RealType(idx0)) * args.inv_hw;
        //     });
        //
        //     // ------------------------------------------------------------
        //     // 2) Fill kw in warp-parallel: N = Dim*W entries
        //     // ------------------------------------------------------------
        //     Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, Dim * W), [&](const int k) {
        //         const int d = k / W;
        //         const int i = k - d * W;
        //
        //         const RealType xi = g0(team_rank, d) - RealType(i) * args.inv_hw;
        //
        //         if constexpr (Types::KernelType::has_width_template) {
        //             kw(team_rank, d, i) = args.kernel.template eval<W>(xi);
        //         } else {
        //             kw(team_rank, d, i) = args.kernel(xi);
        //         }
        //     });
        //
        //     // ------------------------------------------------------------
        //     // 3) Flat vector loop over stencil points: W^Dim
        //     // ------------------------------------------------------------
        //     const ValueType my_val = args.values(p_global);
        //     auto grid              = args.grid;
        //
        //     constexpr int STENCIL_SIZE = StaticPow<W, Dim>::value;
        //
        //     Kokkos::parallel_for(
        //         Kokkos::ThreadVectorRange(team, STENCIL_SIZE), [&](const int flat) {
        //             if constexpr (Dim == 1) {
        //                 const int i0 = flat;
        //
        //                 const RealType w = kw(team_rank, 0, i0);
        //
        //                 auto& cell = grid(base(team_rank, 0) + i0);
        //                 Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
        //                                    static_cast<grid_value_t>(my_val * w));
        //
        //             } else if constexpr (Dim == 2) {
        //                 const int i0 = flat % W;
        //                 const int i1 = flat / W;
        //
        //                 const RealType w = kw(team_rank, 0, i0) * kw(team_rank, 1, i1);
        //
        //                 auto& cell = grid(base(team_rank, 0) + i0, base(team_rank, 1) + i1);
        //                 Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
        //                                    static_cast<grid_value_t>(my_val * w));
        //
        //             } else if constexpr (Dim == 3) {
        //                 const int i0 = flat % W;
        //                 const int t  = flat / W;
        //                 const int i1 = t % W;
        //                 const int i2 = t / W;
        //
        //                 const RealType w =
        //                     kw(team_rank, 0, i0) * kw(team_rank, 1, i1) * kw(team_rank, 2, i2);
        //
        //                 auto& cell = grid(base(team_rank, 0) + i0, base(team_rank, 1) + i1,
        //                                   base(team_rank, 2) + i2);
        //                 Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
        //                                    static_cast<grid_value_t>(my_val * w));
        //             }
        //         });
        // }

        // KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
        //     using grid_value_t = typename decltype(args.grid)::non_const_value_type;
        //
        //     const int team_rank = team.team_rank();
        //     const int team_size = team.team_size();
        //
        //     // Allocate scratch views
        //     ScratchBaseView base(team.team_scratch(0), team_size, Dim);
        //     ScratchWeightsView kw(team.team_scratch(0), team_size, Dim, W);
        //
        //     // Global particle index for this team member
        //     const size_t base_particle = team.league_rank() * team_size;
        //     size_t p_global            = base_particle + team_rank;
        //
        //     if (p_global >= args.n_particles) {
        //         return;
        //     }
        //
        //     // Each team member loads its particle
        //     if constexpr (Policy::use_sorting) {
        //         p_global = args.permute(p_global);
        //     }
        //
        //     CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};
        //
        //     constexpr auto dimension = Dim;
        //     constexpr auto width     = W;
        //
        //     Kokkos::parallel_for(
        //         Kokkos::ThreadVectorMDRange(team, dimension, width), [&](const int d, const int
        //         i) {
        //             const RealType g_pos = transform.toGridCoordinate(args.x(p_global)[d], d);
        //             const int idx0       = transform.getStencilBase(g_pos, W);
        //
        //             base(team_rank, d) = idx0 - args.local_offset[d] + args.nghost;
        //
        //             if constexpr (Types::KernelType::has_width_template) {
        //                 kw(team_rank, d, i) = args.kernel.template eval<W>(
        //                     (g_pos - RealType(idx0 + i)) * args.inv_hw);
        //             } else {
        //                 kw(team_rank, d, i) =
        //                     args.kernel((g_pos - RealType(idx0 + i)) * args.inv_hw);
        //             }
        //         });
        //
        //     // Synchronize so all threads see the shared data
        //     team.team_barrier();
        //
        //     // Each team member processes its particle using vector parallelism
        //     const ValueType my_val = args.values(p_global);
        //     auto grid              = args.grid;
        //
        //     auto stencil_extents = Kokkos::Array<int, Dim>{};
        //     for_constexpr(std::make_integer_sequence<int, Dim>{}, [&]<int d>() {
        //         stencil_extents[d] = W;
        //     });
        //     thread_vector_md_for<Dim>(team, stencil_extents, [&](auto... stencil_idx) {
        //         RealType w = product_over<Dim>([&]<int D>() {
        //             return kw(team_rank, D, get_arg<D>(stencil_idx...));
        //         });
        //
        //         auto& cell = grid_at(grid, base, team_rank, stencil_idx...);
        //         Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
        //                            static_cast<grid_value_t>(my_val * w));
        //     });
        // }

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