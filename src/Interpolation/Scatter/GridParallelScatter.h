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
    // Scratch size calculation
    // ─────────────────────────────────────────────────────────────────────────
    template <bool IsComplex>
    static size_t compute_scratch_size(const Vector<int, Dim>& tile_size) {
        size_t htot = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            htot *= static_cast<size_t>(tile_size[d] + W);
        }

        const size_t real_count = (IsComplex ? 2 : 1) * htot
                                  + static_cast<size_t>(Dim) * W
                                  + (IsComplex ? 2 : 1);
        const size_t int_count = static_cast<size_t>(Dim);

        return real_count * sizeof(RealType) + int_count * sizeof(int) + 4;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Arguments
    // ─────────────────────────────────────────────────────────────────────────
    struct Arguments : ScatterArgumentsBase<Arguments, Types> {
        Kokkos::View<uint64_t*, memory_space> permute;
        Kokkos::View<uint64_t*, memory_space> bin_offsets;
        Vector<int, Dim> num_tiles;
        Vector<int, Dim> tile_size;
        int team_size;

        template <class Field, class Positions, class Values, class Kernel>
        static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                const Kernel& k, const ScatterConfig<Dim>& config,
                                const BinningResult<Dim, memory_space>& binning) {
            Arguments a;
            a.initBase(field, pos, vals, k);
            a.permute     = binning.permute;
            a.bin_offsets = binning.bin_offsets;
            a.num_tiles   = binning.num_tiles;
            a.tile_size   = config.get_tile_size();
            a.team_size   = config.team_size;
            return a;
        }
    };

    Arguments args;

    // ─────────────────────────────────────────────────────────────────────────
    // Histogram helper
    // ─────────────────────────────────────────────────────────────────────────
    template <typename GridValueT>
    struct Histogram {
        static constexpr bool is_complex =
            std::is_same_v<GridValueT, Kokkos::complex<RealType>>;

        scratch_real_view data_r;
        scratch_real_view data_i;
        Vector<int, Dim> size;
        size_t total;

        KOKKOS_INLINE_FUNCTION void init(const team_member& team, scratch_real_view r,
                                         scratch_real_view i, Vector<int, Dim> sz, size_t n) {
            data_r = r;
            if constexpr (is_complex) data_i = i;
            size  = sz;
            total = n;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total), [&](size_t idx) {
                data_r(idx) = RealType(0);
                if constexpr (is_complex) data_i(idx) = RealType(0);
            });
        }

        KOKKOS_INLINE_FUNCTION size_t to_flat(const Kokkos::Array<int, Dim>& c) const {
            size_t idx = 0, stride = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                idx += static_cast<size_t>(c[d]) * stride;
                stride *= static_cast<size_t>(size[d]);
            }
            return idx;
        }

        KOKKOS_INLINE_FUNCTION Kokkos::Array<int, Dim> from_flat(size_t idx) const {
            Kokkos::Array<int, Dim> c{};
            for (unsigned d = 0; d < Dim; ++d) {
                c[d] = static_cast<int>(idx % static_cast<size_t>(size[d]));
                idx /= static_cast<size_t>(size[d]);
            }
            return c;
        }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Geometry helpers
    // ─────────────────────────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
        Vector<int, Dim> hs;
        for (unsigned d = 0; d < Dim; ++d) hs[d] = args.tile_size[d] + W;
        return hs;
    }

    KOKKOS_INLINE_FUNCTION size_t hist_total() const {
        size_t n = 1;
        for (unsigned d = 0; d < Dim; ++d) n *= static_cast<size_t>(args.tile_size[d] + W);
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

    KOKKOS_INLINE_FUNCTION static constexpr size_t stencil_total() {
        size_t n = 1;
        for (unsigned d = 0; d < Dim; ++d) n *= static_cast<size_t>(W);
        return n;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kernel operator
    // ─────────────────────────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
        using grid_value_t = typename decltype(args.grid)::non_const_value_type;

        constexpr bool grid_is_complex =
            std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
        constexpr bool val_is_complex =
            std::is_same_v<ValueType, Kokkos::complex<RealType>>;

        const size_t tile_id = team.league_rank();
        const auto tile_base = decode_tile_base(tile_id);

        const auto hs     = hist_size();
        const size_t htot = hist_total();

        scratch_real_view scratch_r(team.team_scratch(0), htot);
        scratch_real_view scratch_i;
        if constexpr (Histogram<grid_value_t>::is_complex) {
            scratch_i = scratch_real_view(team.team_scratch(0), htot);
        }

        Histogram<grid_value_t> hist;
        hist.init(team, scratch_r, scratch_i, hs, htot);
        team.team_barrier();

        scratch_real_view k_vals(team.team_scratch(0), Dim * W);
        scratch_int_view base(team.team_scratch(0), Dim);
        scratch_real_view vparts(team.team_scratch(0), grid_is_complex ? 2 : 1);

        const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};

        const size_t pstart = args.bin_offsets(tile_id);
        const size_t pend   = args.bin_offsets(tile_id + 1);

        for (size_t ip = pstart; ip < pend; ++ip) {
            Kokkos::single(Kokkos::PerTeam(team), [&] {
                const size_t p  = args.permute(ip);
                const auto v_in = args.values(p);

                if constexpr (grid_is_complex) {
                    if constexpr (val_is_complex) {
                        vparts(0) = v_in.real();
                        vparts(1) = v_in.imag();
                    } else {
                        vparts(0) = static_cast<RealType>(v_in);
                        vparts(1) = RealType(0);
                    }
                } else {
                    if constexpr (val_is_complex) {
                        vparts(0) = v_in.real();
                    } else {
                        vparts(0) = static_cast<RealType>(v_in);
                    }
                }

                for (unsigned d = 0; d < Dim; ++d) {
                    const RealType g = transform.toGridCoordinate(args.x(p)[d], d);
                    const int idx0   = transform.getStencilBase(g, W);
                    base(d)          = idx0 - args.local_offset[d];
                }
            });

            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, Dim * W), [&](int flat) {
                const int d      = flat / W;
                const int i      = flat % W;
                const size_t p   = args.permute(ip);
                const RealType g = transform.toGridCoordinate(args.x(p)[d], d);
                const int idx0   = base(d) + args.local_offset[d];
                k_vals(flat)     = args.kernel((g - RealType(idx0 + i)) * args.inv_hw);
            });

            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, stencil_total()), [&](size_t t) {
                size_t tmp     = t;
                RealType wprod = RealType(1);
                Kokkos::Array<int, Dim> hc{};

                for (unsigned d = 0; d < Dim; ++d) {
                    const int off = static_cast<int>(tmp % static_cast<size_t>(W));
                    tmp /= static_cast<size_t>(W);
                    wprod *= k_vals(d * W + off);
                    hc[d] = base(d) + off + half_left - tile_base[d];
                }

                const size_t hidx = hist.to_flat(hc);

                if constexpr (Histogram<grid_value_t>::is_complex) {
                    hist.data_r(hidx) += vparts(0) * wprod;
                    hist.data_i(hidx) += vparts(1) * wprod;
                } else {
                    hist.data_r(hidx) += vparts(0) * wprod;
                }
            });

            team.team_barrier();
        }

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, htot), [&](size_t idx) {
            const auto hc = hist.from_flat(idx);

            Kokkos::Array<int, Dim> gc{};
            for (unsigned d = 0; d < Dim; ++d) {
                const int local = tile_base[d] + hc[d] - half_left;
                if (local < -args.nghost || local >= args.n_grid_local[d] + args.nghost)
                    return;
                gc[d] = local + args.nghost;
            }

            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                if constexpr (Histogram<grid_value_t>::is_complex) {
                    RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                    Kokkos::atomic_add(&ptr[0], hist.data_r(idx));
                    Kokkos::atomic_add(&ptr[1], hist.data_i(idx));
                } else {
                    Kokkos::atomic_add(&args.grid(gc[Is]...),
                                       static_cast<grid_value_t>(hist.data_r(idx)));
                }
            }(std::make_index_sequence<Dim>{});

        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Run
    // ─────────────────────────────────────────────────────────────────────────
    void run(size_t) {
        using grid_value_t  = typename decltype(args.grid)::non_const_value_type;
        constexpr bool cplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

        const size_t scratch = compute_scratch_size<cplx>(args.tile_size);

        size_t n_tiles = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            n_tiles *= static_cast<size_t>(args.num_tiles[d]);
        }

        Kokkos::parallel_for(
            "GridParallelScatter",
            team_policy(n_tiles, args.team_size).set_scratch_size(0, Kokkos::PerTeam(scratch)),
            *this);
    }
};

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_GRID_PARALLEL_SCATTER_H