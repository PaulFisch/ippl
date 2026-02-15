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

            // Global particle index for this team member
            const size_t base_particle = team.league_rank() * team_size;
            size_t p_global            = base_particle + team_rank;

            if (p_global >= args.n_particles) {
                return;
            }

            // Each team member loads its particle
            if constexpr (Policy::use_sorting) {
                p_global = args.permute(p_global);
            }

            CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};

            constexpr auto dimension = Dim;
            constexpr auto width = W;

            Kokkos::parallel_for(
                Kokkos::ThreadVectorMDRange(team, dimension,  width), [&](const int d, const int  i) {
                    // int d                = idx_linear % Dim;
                    // int i                = idx_linear / Dim;
                    const RealType g_pos = transform.toGridCoordinate(args.x(p_global)[d], d);
                    const int idx0       = transform.getStencilBase(g_pos, W);

                    base(team_rank, d) = idx0 - args.local_offset[d] + args.nghost;

                    if constexpr (Types::KernelType::has_width_template) {
                        kw(team_rank, d, i) = args.kernel.template eval<W>(
                            (g_pos - RealType(idx0 + i)) * args.inv_hw);
                    } else {
                        kw(team_rank, d, i) =
                            args.kernel((g_pos - RealType(idx0 + i)) * args.inv_hw);
                    }
                });

            // Synchronize so all threads see the shared data
            team.team_barrier();

            // Each team member processes its particle using vector parallelism
            const ValueType my_val = args.values(p_global);
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

                    if constexpr (std::is_same_v<grid_value_t, Kokkos::complex<RealType>>
                                  && std::is_same_v<ValueType, RealType>) {
                        const auto contribution = my_val * w;

                        // Perform atomic scatter
                        if constexpr (Dim == 1) {
                            Kokkos::atomic_add(&grid(my_base[0] + stencil_idx[0]).real(),
                                               contribution);
                        } else if constexpr (Dim == 2) {
                            Kokkos::atomic_add(
                                &grid(my_base[0] + stencil_idx[0], my_base[1] + stencil_idx[1])
                                     .real(),
                                contribution);
                        } else if constexpr (Dim == 3) {
                            Kokkos::atomic_add(
                                &grid(my_base[0] + stencil_idx[0], my_base[1] + stencil_idx[1],
                                      my_base[2] + stencil_idx[2])
                                     .real(),
                                contribution);
                        }
                    } else {
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
                    }
                });
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