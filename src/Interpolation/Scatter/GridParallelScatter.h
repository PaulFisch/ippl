#ifndef IPPL_GRID_PARALLEL_SCATTER_H
#define IPPL_GRID_PARALLEL_SCATTER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

namespace ippl::Interpolation::detail {

    template <int W, class Types, class Policy>
    struct GridParallelScatter {
        static_assert(Policy::use_sorting,
                      "GridParallelScatter assumes sorted/bin-partitioned particles");

        static constexpr bool requires_binning = true;
        static constexpr unsigned Dim          = Types::Dim;
        static constexpr int half_left         = (W - 1) / 2;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy   = Kokkos::TeamPolicy<execution_space>;
        using team_member   = typename team_policy::member_type;
        using scratch_space = typename execution_space::scratch_memory_space;

        using scratch_real_view =
            Kokkos::View<RealType*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using scratch_int_view =
            Kokkos::View<int*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // ─────────────────────────────────────────────────────────────────────────
        // Scratch size:
        //   hist_r[htot] + hist_i[htot] (complex only)
        //   k_vals[Dim * W]
        //   base_s[Dim]
        //
        // ─────────────────────────────────────────────────────────────────────────
        template <bool IsComplex>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size) {
            size_t htot = 1;
            for (unsigned d = 0; d < Dim; ++d)
                htot *= static_cast<size_t>(tile_size[d] + W);

            size_t sz = (IsComplex ? 2 : 1) * scratch_real_view::shmem_size(htot)
                        + scratch_real_view::shmem_size(static_cast<int>(Dim) * W)
                        + scratch_int_view::shmem_size(static_cast<int>(Dim));
            return sz;
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Arguments
        // ─────────────────────────────────────────────────────────────────────────
        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            // Fixed: these are particle-index arrays (size_t), not uint64_t
            Kokkos::View<size_t*, memory_space> permute;
            Kokkos::View<size_t*, memory_space> bin_offsets;
            Vector<int, Dim> num_tiles;
            Vector<int, Dim> tile_size;
            int team_size;
            int oversubscription_factor;

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                    const Kernel& k, const ScatterConfig<Dim>& config,
                                    const BinningResult<Dim, memory_space>& binning) {
                Arguments a;
                a.initBase(field, pos, vals, k);
                a.permute                 = binning.permute;
                a.bin_offsets             = binning.bin_offsets;
                a.num_tiles               = binning.num_tiles;
                a.tile_size               = config.get_tile_size();
                a.team_size               = config.team_size;
                a.oversubscription_factor = config.oversubscription_factor;
                return a;
            }
        };

        Arguments args;

        // Sub-team decomposition: set in run(), read in operator()
        size_t sub_teams_per_tile_ = 1;
        size_t particles_per_team_ = 1;

        // ─────────────────────────────────────────────────────────────────────────
        // Geometry helpers
        // ─────────────────────────────────────────────────────────────────────────
        KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
            Vector<int, Dim> hs;
            for (unsigned d = 0; d < Dim; ++d)
                hs[d] = args.tile_size[d] + W;
            return hs;
        }

        KOKKOS_INLINE_FUNCTION size_t hist_total() const {
            size_t n = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n *= static_cast<size_t>(args.tile_size[d] + W);
            return n;
        }

        KOKKOS_INLINE_FUNCTION Vector<int, Dim> decode_tile_base(size_t tile_id) const {
            Vector<int, Dim> tile_base;
            for (size_t t = tile_id, d = Dim; d-- > 0;) {
                tile_base[d] = static_cast<int>(t % static_cast<size_t>(args.num_tiles[d]))
                               * args.tile_size[d];
                t /= static_cast<size_t>(args.num_tiles[d]);
            }
            return tile_base;
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Kernel operator
        // ─────────────────────────────────────────────────────────────────────────
        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
            constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

            // ── Sub-team decomposition ──────────────────────────────────────────
            const size_t league_r  = static_cast<size_t>(team.league_rank());
            const size_t tile_id   = league_r / sub_teams_per_tile_;
            const size_t sub_id    = league_r % sub_teams_per_tile_;
            const size_t bin_start = args.bin_offsets(tile_id);
            const size_t bin_end   = args.bin_offsets(tile_id + 1);
            const size_t bin_size  = bin_end - bin_start;
            const size_t particles_per_sub =
                (bin_size + sub_teams_per_tile_ - 1) / sub_teams_per_tile_;

            const size_t pstart = bin_start + sub_id * particles_per_sub;
            if (pstart >= bin_end)
                return;

            const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);

            if (pstart >= bin_end)
                return;  // this sub-team has no work

            const auto tile_base = decode_tile_base(tile_id);
            const auto hs        = hist_size();
            const size_t htot    = hist_total();

            // ── Histogram scratch ───────────────────────────────────────────────
            scratch_real_view hist_r(team.team_scratch(0), htot);
            scratch_real_view hist_i;
            if constexpr (gcplx)
                hist_i = scratch_real_view(team.team_scratch(0), htot);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, htot), [&](size_t i) {
                hist_r(i) = RealType(0);
                if constexpr (gcplx)
                    hist_i(i) = RealType(0);
            });
            team.team_barrier();

            // ── Per-particle scratch ─────────────────────────────────────────────
            // k_vals: Dim*W kernel weights, filled in parallel once per particle
            // base_s: Dim stencil base indices, set as a side effect of k_vals loop
            // (i == 0 thread for each dimension writes base_s[d]; no race since d is unique)
            scratch_real_view k_vals(team.team_scratch(0), static_cast<int>(Dim) * W);
            scratch_int_view base_s(team.team_scratch(0), static_cast<int>(Dim));

            const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
                                                               args.n_grid};

            for (size_t ip = pstart; ip < pend; ++ip) {
                const size_t p = args.permute(ip);

                // ── Broadcast particle value (no scratch needed) ────────────────
                // Each call to Kokkos::single(PerTeam) broadcasts the result to all
                // threads in the team via the second argument.
                RealType val_r = RealType(0);
                Kokkos::single(
                    Kokkos::PerTeam(team),
                    [&](RealType& v) {
                        if constexpr (vcplx)
                            v = args.values(p).real();
                        else
                            v = static_cast<RealType>(args.values(p));
                    },
                    val_r);

                RealType val_i = RealType(0);
                if constexpr (gcplx) {
                    Kokkos::single(
                        Kokkos::PerTeam(team),
                        [&](RealType& v) {
                            if constexpr (vcplx)
                                v = args.values(p).imag();
                            // else stays 0 for real->complex promotion
                        },
                        val_i);
                }

                // ── Kernel weights + stencil base in parallel ───────────────────
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, static_cast<int>(Dim) * W), [&](int flat) {
                        const int d       = flat / W;
                        const int i       = flat % W;
                        const RealType gp = transform.toGridCoordinate(args.x(p)[d], d);
                        const int idx0    = transform.getStencilBase(gp, W);
                        k_vals(flat)      = args.kernel((gp - RealType(idx0 + i)) * args.inv_hw);
                        // Thread with i==0 is the only writer for base_s[d]
                        if (i == 0)
                            base_s(d) = idx0 - args.local_offset[d];
                    });
                team.team_barrier();

                // ── Specialized scatter to local histogram ──────────────────────
                // All index arithmetic is explicit; no runtime index decomposition.
                if constexpr (Dim == 1) {
                    const int bh0 = base_s(0) + half_left - tile_base[0];

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W), [&](int i0) {
                        const RealType w  = k_vals(i0);
                        const size_t hidx = static_cast<size_t>(bh0 + i0);
                        hist_r(hidx) += val_r * w;
                        if constexpr (gcplx)
                            hist_i(hidx) += val_i * w;
                    });

                } else if constexpr (Dim == 2) {
                    const int bh0 = base_s(0) + half_left - tile_base[0];
                    const int bh1 = base_s(1) + half_left - tile_base[1];

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W * W), [&](int flat) {
                        const int i0     = flat % W;
                        const int i1     = flat / W;
                        const RealType w = k_vals(i0) * k_vals(W + i1);
                        const size_t hidx =
                            static_cast<size_t>(bh0 + i0)
                            + static_cast<size_t>(hs[0]) * static_cast<size_t>(bh1 + i1);
                        hist_r(hidx) += val_r * w;
                        if constexpr (gcplx)
                            hist_i(hidx) += val_i * w;
                    });

                } else if constexpr (Dim == 3) {
                    const int bh0 = base_s(0) + half_left - tile_base[0];
                    const int bh1 = base_s(1) + half_left - tile_base[1];
                    const int bh2 = base_s(2) + half_left - tile_base[2];

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W * W * W), [&](int flat) {
                        const int i0     = flat % W;
                        const int i1     = (flat / W) % W;
                        const int i2     = flat / (W * W);
                        const RealType w = k_vals(i0) * k_vals(W + i1) * k_vals(2 * W + i2);
                        const size_t hidx =
                            static_cast<size_t>(bh0 + i0)
                            + static_cast<size_t>(hs[0])
                                  * (static_cast<size_t>(bh1 + i1)
                                     + static_cast<size_t>(hs[1]) * static_cast<size_t>(bh2 + i2));
                        hist_r(hidx) += val_r * w;
                        if constexpr (gcplx)
                            hist_i(hidx) += val_i * w;
                    });
                }
                team.team_barrier();
            }

            // ── Flush local histogram → global grid (atomic) ────────────────────
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, htot), [&](size_t idx) {
                // Decode flat histogram index to per-dim histogram coordinates
                size_t tmp = idx;
                Kokkos::Array<int, Dim> hc{};
                for (unsigned d = 0; d < Dim; ++d) {
                    hc[d] = static_cast<int>(tmp % static_cast<size_t>(hs[d]));
                    tmp /= static_cast<size_t>(hs[d]);
                }

                // Map histogram coords to local grid coords; skip out-of-bounds cells
                Kokkos::Array<int, Dim> gc{};
                for (unsigned d = 0; d < Dim; ++d) {
                    const int local = tile_base[d] + hc[d] - half_left;
                    if (local < -args.nghost || local >= args.n_grid_local[d] + args.nghost)
                        return;
                    gc[d] = local + args.nghost;
                }

                [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    if constexpr (gcplx) {
                        RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                        Kokkos::atomic_add(&ptr[0], hist_r(idx));
                        Kokkos::atomic_add(&ptr[1], hist_i(idx));
                    } else {
                        Kokkos::atomic_add(&args.grid(gc[Is]...),
                                           static_cast<grid_value_t>(hist_r(idx)));
                    }
                }(std::make_index_sequence<Dim>{});
            });
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Run
        //
        // Sub-team strategy
        // -----------------
        // avg = n_particles / n_tiles  (average bin size)
        // particles_per_team_ = avg
        // sub_teams_per_tile_ = oversubscription_factor  (default 4)
        //
        // ─────────────────────────────────────────────────────────────────────────
        void run(size_t n_particles) {
            using grid_value_t  = typename decltype(args.grid)::non_const_value_type;
            constexpr bool cplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

            size_t n_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n_tiles *= static_cast<size_t>(args.num_tiles[d]);

            if (n_tiles == 0 || n_particles == 0)
                return;

            particles_per_team_ = std::max(size_t(1), n_particles / n_tiles);
            sub_teams_per_tile_ = std::max(size_t(1), size_t(args.oversubscription_factor));

            const size_t scratch = compute_scratch_size<cplx>(args.tile_size);

            Kokkos::parallel_for("GridParallelScatter",
                                 team_policy(n_tiles * sub_teams_per_tile_, args.team_size)
                                     .set_scratch_size(0, Kokkos::PerTeam(scratch)),
                                 *this);
        }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_GRID_PARALLEL_SCATTER_H