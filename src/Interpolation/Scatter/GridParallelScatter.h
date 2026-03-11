// #ifndef IPPL_GRID_PARALLEL_SCATTER_H
// #define IPPL_GRID_PARALLEL_SCATTER_H
//
// namespace ippl::Interpolation::detail {
//
//     template <int Base, int Exp>
//     struct StaticPow {
//         static constexpr int value = Base * StaticPow<Base, Exp - 1>::value;
//     };
//     template <int Base>
//     struct StaticPow<Base, 0> {
//         static constexpr int value = 1;
//     };
//
//     template <int W, class Types, class Policy>
//     struct GridParallelScatter {
//         static_assert(Policy::use_sorting,
//                       "GridParallelScatter assumes sorted/bin-partitioned particles");
//
//         static constexpr bool requires_binning = true;
//         static constexpr unsigned Dim          = Types::Dim;
//         static constexpr int half_left         = (W + 1) / 2;
//         static constexpr int padded_extra      = 2 * half_left;
//         static constexpr int stencil_total     = StaticPow<W, Dim>::value;
//
//         using RealType        = typename Types::RealType;
//         using ValueType       = typename Types::ValueType;
//         using memory_space    = typename Types::memory_space;
//         using execution_space = typename Types::execution_space;
//
//         using team_policy   = Kokkos::TeamPolicy<execution_space>;
//         using team_member   = typename team_policy::member_type;
//         using scratch_space = typename execution_space::scratch_memory_space;
//
//         using scratch_real_view =
//             Kokkos::View<RealType*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
//         using scratch_int_view =
//             Kokkos::View<int*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
//
//         struct Arguments : ScatterArgumentsBase<Arguments, Types> {
//             Kokkos::View<size_t*, memory_space> permute;
//             Kokkos::View<size_t*, memory_space> bin_offsets;
//             Vector<int, Dim> num_tiles;
//             Vector<int, Dim> tile_size;
//             int team_size;
//             int oversubscription_factor;
//             int batch_np;
//
//             template <class Field, class Positions, class Values, class Kernel>
//             static Arguments create(Field& field, const Positions& pos, const Values& vals,
//                                     const Kernel& k, const ScatterConfig<Dim>& config,
//                                     const BinningResult<Dim, memory_space>& binning) {
//                 Arguments a;
//                 a.initBase(field, pos, vals, k);
//                 a.permute                 = binning.permute;
//                 a.bin_offsets             = binning.bin_offsets;
//                 a.num_tiles               = binning.num_tiles;
//                 a.tile_size               = config.get_tile_size();
//                 a.team_size               = config.team_size;
//                 a.oversubscription_factor = config.oversubscription_factor;
//                 a.batch_np                = config.z_batches > 0 ? config.z_batches : 1;
//                 return a;
//             }
//         };
//
//         Arguments args;
//         size_t hist_total_         = 1;
//         size_t sub_teams_per_tile_ = 1;
//
//         KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
//             Vector<int, Dim> hs;
//             for (unsigned d = 0; d < Dim; ++d)
//                 hs[d] = args.tile_size[d] + padded_extra;
//             return hs;
//         }
//
//         KOKKOS_INLINE_FUNCTION Vector<int, Dim> decode_tile_base(size_t tile_id) const {
//             Vector<int, Dim> tile_base;
//             for (size_t t = tile_id, d = Dim; d-- > 0;) {
//                 tile_base[d] = static_cast<int>(t % static_cast<size_t>(args.num_tiles[d]))
//                                * args.tile_size[d];
//                 t /= static_cast<size_t>(args.num_tiles[d]);
//             }
//             return tile_base;
//         }
//
//         template <bool NeedsImag>
//         KOKKOS_INLINE_FUNCTION static void scatter_particle_fast_3d(
//             const team_member& team, const int sx, const int sy, const int sz, const size_t s1,
//             const size_t s2, const RealType* kw, const RealType vr, const RealType vi,
//             const scratch_real_view& local_r, const scratch_real_view& local_i) {
//             constexpr int plane = W * W;
//             constexpr int total = W * W * W;
//
//             const RealType* kwx = kw + 0 * W;
//             const RealType* kwy = kw + 1 * W;
//             const RealType* kwz = kw + 2 * W;
//
//             for (int idx = team.team_rank(); idx < total; idx += team.team_size()) {
//                 const int zz  = idx / plane;
//                 const int rem = idx - zz * plane;
//                 const int yy  = rem / W;
//                 const int xx  = rem - yy * W;
//
//                 const size_t hidx = static_cast<size_t>(sx + xx) + static_cast<size_t>(sy + yy) *
//                 s1
//                                     + static_cast<size_t>(sz + zz) * s2;
//
//                 const RealType w = kwx[xx] * kwy[yy] * kwz[zz];
//                 local_r(hidx) += vr * w;
//                 if constexpr (NeedsImag)
//                     local_i(hidx) += vi * w;
//             }
//         }
//
//         template <bool NeedsImag>
//         KOKKOS_INLINE_FUNCTION static void scatter_particle_checked_3d(
//             const team_member& team, const int sx, const int sy, const int sz, const int hs0,
//             const int hs1, const int hs2, const size_t s1, const size_t s2, const RealType* kw,
//             const RealType vr, const RealType vi, const scratch_real_view& local_r,
//             const scratch_real_view& local_i) {
//             constexpr int plane = W * W;
//             constexpr int total = W * W * W;
//
//             const RealType* kwx = kw + 0 * W;
//             const RealType* kwy = kw + 1 * W;
//             const RealType* kwz = kw + 2 * W;
//
//             for (int idx = team.team_rank(); idx < total; idx += team.team_size()) {
//                 const int zz  = idx / plane;
//                 const int rem = idx - zz * plane;
//                 const int yy  = rem / W;
//                 const int xx  = rem - yy * W;
//
//                 const int hx = sx + xx;
//                 const int hy = sy + yy;
//                 const int hz = sz + zz;
//
//                 if (static_cast<unsigned>(hx) >= static_cast<unsigned>(hs0)
//                     || static_cast<unsigned>(hy) >= static_cast<unsigned>(hs1)
//                     || static_cast<unsigned>(hz) >= static_cast<unsigned>(hs2))
//                     continue;
//
//                 const size_t hidx = static_cast<size_t>(hx) + static_cast<size_t>(hy) * s1
//                                     + static_cast<size_t>(hz) * s2;
//
//                 const RealType w = kwx[xx] * kwy[yy] * kwz[zz];
//                 local_r(hidx) += vr * w;
//                 if constexpr (NeedsImag)
//                     local_i(hidx) += vi * w;
//             }
//         }
//
//         template <bool NeedsImag>
//         KOKKOS_INLINE_FUNCTION static void scatter_particle_fast_nd(
//             const team_member& team, const Kokkos::Array<int, Dim>& shift,
//             const Kokkos::Array<size_t, Dim>& stride, const RealType* kw, const RealType vr,
//             const RealType vi, const scratch_real_view& local_r, const scratch_real_view&
//             local_i) { for (int idx = team.team_rank(); idx < stencil_total; idx +=
//             team.team_size()) {
//                 int tmp     = idx;
//                 size_t hidx = 0;
//                 RealType w  = RealType(1);
//
//                 for (unsigned d = 0; d < Dim; ++d) {
//                     const int wi = tmp % W;
//                     tmp /= W;
//                     hidx += static_cast<size_t>(shift[d] + wi) * stride[d];
//                     w *= kw[d * W + wi];
//                 }
//
//                 local_r(hidx) += vr * w;
//                 if constexpr (NeedsImag)
//                     local_i(hidx) += vi * w;
//             }
//         }
//
//         template <bool NeedsImag>
//         KOKKOS_INLINE_FUNCTION static void scatter_particle_checked_nd(
//             const team_member& team, const Kokkos::Array<int, Dim>& shift,
//             const Kokkos::Array<int, Dim>& hs, const Kokkos::Array<size_t, Dim>& stride,
//             const RealType* kw, const RealType vr, const RealType vi,
//             const scratch_real_view& local_r, const scratch_real_view& local_i) {
//             for (int idx = team.team_rank(); idx < stencil_total; idx += team.team_size()) {
//                 int tmp     = idx;
//                 size_t hidx = 0;
//                 RealType w  = RealType(1);
//
//                 bool ok = true;
//                 for (unsigned d = 0; d < Dim; ++d) {
//                     const int wi    = tmp % W;
//                     const int coord = shift[d] + wi;
//                     tmp /= W;
//
//                     if (static_cast<unsigned>(coord) >= static_cast<unsigned>(hs[d])) {
//                         ok = false;
//                         break;
//                     }
//
//                     hidx += static_cast<size_t>(coord) * stride[d];
//                     w *= kw[d * W + wi];
//                 }
//
//                 if (!ok)
//                     continue;
//
//                 local_r(hidx) += vr * w;
//                 if constexpr (NeedsImag)
//                     local_i(hidx) += vi * w;
//             }
//         }
//
//         KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
//             using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
//             constexpr bool grid_complex  = std::is_same_v<grid_value_t,
//             Kokkos::complex<RealType>>; constexpr bool value_complex = std::is_same_v<ValueType,
//             Kokkos::complex<RealType>>; constexpr bool needs_imag    = grid_complex &&
//             value_complex;
//
//             const size_t league_r = static_cast<size_t>(team.league_rank());
//             const size_t tile_id  = league_r / sub_teams_per_tile_;
//             const size_t sub_id   = league_r % sub_teams_per_tile_;
//
//             const size_t bin_start = args.bin_offsets(tile_id);
//             const size_t bin_end   = args.bin_offsets(tile_id + 1);
//             const size_t bin_size  = bin_end - bin_start;
//             if (bin_size == 0)
//                 return;
//
//             const size_t target_particles_per_subteam =
//                 static_cast<size_t>(Kokkos::max(128, 4 * args.team_size));
//
//             const size_t active_subteams =
//                 Kokkos::min(sub_teams_per_tile_,
//                             Kokkos::max<size_t>(1, (bin_size + target_particles_per_subteam - 1)
//                                                        / target_particles_per_subteam));
//
//             if (sub_id >= active_subteams)
//                 return;
//
//             const size_t particles_per_sub = (bin_size + active_subteams - 1) / active_subteams;
//             const size_t pstart            = bin_start + sub_id * particles_per_sub;
//             if (pstart >= bin_end)
//                 return;
//             const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);
//
//             const auto tile_base = decode_tile_base(tile_id);
//             const auto hs_vec    = hist_size();
//             const size_t htot    = hist_total_;
//             const int batch_np   = args.batch_np;
//
//             Kokkos::Array<int, Dim> hs{};
//             Kokkos::Array<size_t, Dim> stride{};
//             stride[0] = 1;
//             for (unsigned d = 0; d < Dim; ++d) {
//                 hs[d] = hs_vec[d];
//                 if (d > 0)
//                     stride[d] = stride[d - 1] * static_cast<size_t>(hs[d - 1]);
//             }
//
//             const int hs0   = hs[0];
//             const int hs1   = (Dim > 1 ? hs[1] : 1);
//             const int hs2   = (Dim > 2 ? hs[2] : 1);
//             const size_t s1 = (Dim > 1 ? stride[1] : 1);
//             const size_t s2 = (Dim > 2 ? stride[2] : 1);
//
//             auto scratch = team.team_scratch(0);
//
//             scratch_real_view local_r(scratch, htot);
//             scratch_real_view local_i;
//             if constexpr (needs_imag)
//                 local_i = scratch_real_view(scratch, htot);
//
//             scratch_real_view kerevals(scratch, static_cast<size_t>(batch_np) * Dim * W);
//             scratch_real_view vals_r(scratch, batch_np);
//             scratch_real_view vals_i;
//             if constexpr (needs_imag)
//                 vals_i = scratch_real_view(scratch, batch_np);
//
//             scratch_int_view shifts(scratch, static_cast<size_t>(batch_np) * Dim);
//
//             for (size_t i = static_cast<size_t>(team.team_rank()); i < htot;
//                  i += static_cast<size_t>(team.team_size())) {
//                 local_r(i) = RealType(0);
//                 if constexpr (needs_imag)
//                     local_i(i) = RealType(0);
//             }
//             team.team_barrier();
//
//             const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
//                                                                args.n_grid};
//
//             for (size_t batch_begin = pstart; batch_begin < pend;
//                  batch_begin += static_cast<size_t>(batch_np)) {
//                 const int batch_size = static_cast<int>(
//                     Kokkos::min(pend - batch_begin, static_cast<size_t>(batch_np)));
//
//                 Kokkos::parallel_for(Kokkos::TeamThreadRange(team, batch_size), [&](const int bi)
//                 {
//                     const size_t p = args.permute(batch_begin + static_cast<size_t>(bi));
//
//                     if constexpr (Dim == 3) {
//                         const RealType gp0 = transform.template
//                         toGridCoordinate<0>(args.x(p)[0]); const RealType gp1 =
//                         transform.template toGridCoordinate<1>(args.x(p)[1]); const RealType gp2
//                         = transform.template toGridCoordinate<2>(args.x(p)[2]);
//
//                         const int idx0 = transform.template getStencilBase<W>(gp0 -
//                         RealType(0.5)); const int idx1 = transform.template getStencilBase<W>(gp1
//                         - RealType(0.5)); const int idx2 = transform.template
//                         getStencilBase<W>(gp2 - RealType(0.5));
//
//                         for (int wi = 0; wi < W; ++wi) {
//                             kerevals(static_cast<size_t>(bi) * Dim * W + 0 * W + wi) =
//                             args.kernel(
//                                 (gp0 - (RealType(idx0 + wi) + RealType(0.5))) * args.inv_hw);
//                             kerevals(static_cast<size_t>(bi) * Dim * W + 1 * W + wi) =
//                             args.kernel(
//                                 (gp1 - (RealType(idx1 + wi) + RealType(0.5))) * args.inv_hw);
//                             kerevals(static_cast<size_t>(bi) * Dim * W + 2 * W + wi) =
//                             args.kernel(
//                                 (gp2 - (RealType(idx2 + wi) + RealType(0.5))) * args.inv_hw);
//                         }
//
//                         shifts(static_cast<size_t>(bi) * Dim + 0) =
//                             idx0 - args.local_offset[0] + half_left - tile_base[0];
//                         shifts(static_cast<size_t>(bi) * Dim + 1) =
//                             idx1 - args.local_offset[1] + half_left - tile_base[1];
//                         shifts(static_cast<size_t>(bi) * Dim + 2) =
//                             idx2 - args.local_offset[2] + half_left - tile_base[2];
//                     } else {
//                         for (unsigned d = 0; d < Dim; ++d) {
//                             const RealType gp = transform.toGridCoordinate(args.x(p)[d], d);
//                             const int idx0    = transform.getStencilBase(gp - RealType(0.5), W);
//
//                             for (int wi = 0; wi < W; ++wi) {
//                                 kerevals(static_cast<size_t>(bi) * Dim * W + d * W + wi) =
//                                     args.kernel((gp - (RealType(idx0 + wi) + RealType(0.5)))
//                                                 * args.inv_hw);
//                             }
//
//                             shifts(static_cast<size_t>(bi) * Dim + d) =
//                                 idx0 - args.local_offset[d] + half_left - tile_base[d];
//                         }
//                     }
//
//                     if constexpr (value_complex) {
//                         vals_r(bi) = args.values(p).real();
//                         if constexpr (needs_imag)
//                             vals_i(bi) = args.values(p).imag();
//                     } else {
//                         vals_r(bi) = static_cast<RealType>(args.values(p));
//                     }
//                 });
//                 team.team_barrier();
//
//                 for (int bi = 0; bi < batch_size; ++bi) {
//                     const RealType* kw = kerevals.data() + static_cast<size_t>(bi) * Dim * W;
//                     const RealType vr  = vals_r(bi);
//                     const RealType vi  = (needs_imag ? vals_i(bi) : RealType(0));
//
//                     if constexpr (Dim == 3) {
//                         const int sx = shifts(static_cast<size_t>(bi) * Dim + 0);
//                         const int sy = shifts(static_cast<size_t>(bi) * Dim + 1);
//                         const int sz = shifts(static_cast<size_t>(bi) * Dim + 2);
//
//                         const bool full_inside = (sx >= 0 && sx <= hs0 - W)
//                                                  && (sy >= 0 && sy <= hs1 - W)
//                                                  && (sz >= 0 && sz <= hs2 - W);
//
//                         if (full_inside) {
//                             scatter_particle_fast_3d<needs_imag>(team, sx, sy, sz, s1, s2, kw,
//                             vr,
//                                                                  vi, local_r, local_i);
//                         } else {
//                             scatter_particle_checked_3d<needs_imag>(team, sx, sy, sz, hs0, hs1,
//                             hs2,
//                                                                     s1, s2, kw, vr, vi, local_r,
//                                                                     local_i);
//                         }
//                     } else {
//                         Kokkos::Array<int, Dim> shift{};
//                         bool full_inside = true;
//                         for (unsigned d = 0; d < Dim; ++d) {
//                             shift[d] = shifts(static_cast<size_t>(bi) * Dim + d);
//                             if (shift[d] < 0 || shift[d] > hs[d] - W)
//                                 full_inside = false;
//                         }
//
//                         if (full_inside) {
//                             scatter_particle_fast_nd<needs_imag>(team, shift, stride, kw, vr, vi,
//                                                                  local_r, local_i);
//                         } else {
//                             scatter_particle_checked_nd<needs_imag>(team, shift, hs, stride, kw,
//                             vr,
//                                                                     vi, local_r, local_i);
//                         }
//                     }
//                     team.team_barrier();
//                 }
//             }
//
//             for (size_t idx = static_cast<size_t>(team.team_rank()); idx < htot;
//                  idx += static_cast<size_t>(team.team_size())) {
//                 const RealType rr = local_r(idx);
//
//                 if constexpr (needs_imag) {
//                     const RealType ii = local_i(idx);
//                     if (rr == RealType(0) && ii == RealType(0))
//                         continue;
//                 } else {
//                     if (rr == RealType(0))
//                         continue;
//                 }
//
//                 size_t tmp = idx;
//                 Kokkos::Array<int, Dim> hc{};
//                 for (unsigned d = 0; d < Dim; ++d) {
//                     hc[d] = static_cast<int>(tmp % static_cast<size_t>(hs_vec[d]));
//                     tmp /= static_cast<size_t>(hs_vec[d]);
//                 }
//
//                 Kokkos::Array<int, Dim> gc{};
//                 for (unsigned d = 0; d < Dim; ++d) {
//                     const int local = tile_base[d] + hc[d] - half_left;
//                     if (local < -args.nghost || local >= args.n_grid_local[d] + args.nghost)
//                         goto flush_next;
//                     gc[d] = local + args.nghost;
//                 }
//
//                 [&]<std::size_t... Is>(std::index_sequence<Is...>) {
//                     if constexpr (grid_complex) {
//                         RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
//                         Kokkos::atomic_add(&ptr[0], rr);
//                         if constexpr (needs_imag)
//                             Kokkos::atomic_add(&ptr[1], local_i(idx));
//                     } else {
//                         Kokkos::atomic_add(&args.grid(gc[Is]...), static_cast<grid_value_t>(rr));
//                     }
//                 }(std::make_index_sequence<Dim>{});
//
// flush_next:;
//             }
//         }
//
//         template <bool NeedsImag>
//         static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /*team_size*/,
//                                            int z_batches) {
//             const int batch_np = z_batches > 0 ? z_batches : 1;
//
//             size_t htot = 1;
//             for (unsigned d = 0; d < Dim; ++d)
//                 htot *= static_cast<size_t>(tile_size[d] + padded_extra);
//
//             size_t s = 0;
//             s += scratch_real_view::shmem_size(htot);
//             if constexpr (NeedsImag)
//                 s += scratch_real_view::shmem_size(htot);
//
//             s += scratch_real_view::shmem_size(static_cast<size_t>(batch_np) * Dim * W);
//             s += scratch_real_view::shmem_size(batch_np);
//             if constexpr (NeedsImag)
//                 s += scratch_real_view::shmem_size(batch_np);
//
//             s += scratch_int_view::shmem_size(static_cast<size_t>(batch_np) * Dim);
//             return s;
//         }
//
//         void run(size_t n_particles) {
//             using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
//             constexpr bool grid_complex  = std::is_same_v<grid_value_t,
//             Kokkos::complex<RealType>>; constexpr bool value_complex = std::is_same_v<ValueType,
//             Kokkos::complex<RealType>>; constexpr bool needs_imag    = grid_complex &&
//             value_complex;
//
//             size_t n_tiles = 1;
//             for (unsigned d = 0; d < Dim; ++d)
//                 n_tiles *= static_cast<size_t>(args.num_tiles[d]);
//
//             if (n_tiles == 0 || n_particles == 0)
//                 return;
//
//             hist_total_ = 1;
//             for (unsigned d = 0; d < Dim; ++d)
//                 hist_total_ *= static_cast<size_t>(args.tile_size[d] + padded_extra);
//
//             sub_teams_per_tile_ =
//                 std::max<size_t>(1, static_cast<size_t>(args.oversubscription_factor));
//
//             const size_t scratch =
//                 compute_scratch_size<needs_imag>(args.tile_size, args.team_size, args.batch_np);
//
//             auto policy = team_policy(n_tiles * sub_teams_per_tile_, args.team_size, 1)
//                               .set_scratch_size(0, Kokkos::PerTeam(scratch));
//
//             Kokkos::parallel_for("GridParallelScatterOutputDrivenBatched", policy, *this);
//         }
//     };
//
// }  // namespace ippl::Interpolation::detail
//
// #endif

#ifndef IPPL_GRID_PARALLEL_SCATTER_H
#define IPPL_GRID_PARALLEL_SCATTER_H

#include <type_traits>

namespace ippl::Interpolation::detail {

    // ─────────────────────────────────────────────────────────────────────────────
    // Compile-time helpers
    // ─────────────────────────────────────────────────────────────────────────────

    template <int Base, int Exp>
    struct StaticPow {
        static constexpr int value = Base * StaticPow<Base, Exp - 1>::value;
    };
    template <int Base>
    struct StaticPow<Base, 0> {
        static constexpr int value = 1;
    };

    template <int A, int B>
    struct StaticGCD {
        static constexpr int value = StaticGCD<B, (A % B)>::value;
    };
    template <int A>
    struct StaticGCD<A, 0> {
        static constexpr int value = A;
    };

    // Policy hook:
    // - if Policy::fixed_oversubscription exists, use it
    // - otherwise default to true
    template <class Policy, class = void>
    struct GridParallelScatterTuning {
        static constexpr bool fixed_oversubscription = true;
    };

    template <class Policy>
    struct GridParallelScatterTuning<Policy,
                                     std::void_t<decltype(Policy::fixed_oversubscription)>> {
        static constexpr bool fixed_oversubscription = Policy::fixed_oversubscription;
    };

    // Shared-memory padding helper.
    // Choose the smallest padding so the physical pitch is coprime to the
    // effective bank count for RealType.
    //
    // float  -> effective_bank_count = 32
    // double -> effective_bank_count = 16
    template <class RealType>
    struct SharedBankLayout {
        static constexpr int bank_count      = 32;
        static constexpr int bank_word_bytes = 4;
        static constexpr int elem_words =
            (static_cast<int>(sizeof(RealType)) + bank_word_bytes - 1) / bank_word_bytes;
        static constexpr int effective_bank_count =
            bank_count / StaticGCD<bank_count, elem_words>::value;

        KOKKOS_INLINE_FUNCTION static int gcd_runtime(int a, int b) {
            while (b != 0) {
                const int t = a % b;
                a           = b;
                b           = t;
            }
            return a;
        }

        KOKKOS_INLINE_FUNCTION static int minimal_coprime_pad(int logical_extent) {
            int pad = 0;
            while (gcd_runtime(logical_extent + pad, effective_bank_count) != 1)
                ++pad;
            return pad;
        }
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // 3D optimized implementation
    // ─────────────────────────────────────────────────────────────────────────────

    template <int W, class Types, class Policy>
    struct GridParallelScatterImpl3D {
        static_assert(Policy::use_sorting,
                      "GridParallelScatter assumes sorted/bin-partitioned particles");
        static_assert(Types::Dim == 3, "GridParallelScatterImpl3D requires Dim == 3");

        static constexpr bool requires_binning = true;
        static constexpr unsigned Dim          = Types::Dim;
        static constexpr int half_left         = (W + 1) / 2;
        static constexpr int padded_extra      = 2 * half_left;
        static constexpr int stencil_total     = W * W * W;
        static constexpr bool fixed_oversubscription =
            GridParallelScatterTuning<Policy>::fixed_oversubscription;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy   = Kokkos::TeamPolicy<execution_space>;
        using team_member   = typename team_policy::member_type;
        using scratch_space = typename execution_space::scratch_memory_space;

        using scratch_real_view =
            Kokkos::View<RealType*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        struct alignas(16) Shift3 {
            int x, y, z, pad;
        };

        using scratch_shift_view =
            Kokkos::View<Shift3*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            Kokkos::View<size_t*, memory_space> permute;
            Kokkos::View<size_t*, memory_space> bin_offsets;
            Vector<int, Dim> num_tiles;
            Vector<int, Dim> tile_size;
            int team_size;
            int oversubscription_factor;
            int batch_np;

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
                a.batch_np                = config.z_batches > 0 ? config.z_batches : 1;
                return a;
            }
        };

        Arguments args;
        size_t logical_total_      = 1;
        size_t physical_total_     = 1;
        size_t sub_teams_per_tile_ = 1;

        // IMPORTANT:
        // binning flattens with dimension 2 fastest:
        //   bin = tx * (ny*nz) + ty * nz + tz
        // so decoding must reverse in order z, y, x.
        KOKKOS_INLINE_FUNCTION void decode_tile_base(const size_t tile_id_in, int& tx, int& ty,
                                                     int& tz) const {
            size_t t = tile_id_in;

            const size_t nt2 = static_cast<size_t>(args.num_tiles[2]);
            const size_t nt1 = static_cast<size_t>(args.num_tiles[1]);

            const int iz = static_cast<int>(t % nt2);
            t /= nt2;
            const int iy = static_cast<int>(t % nt1);
            t /= nt1;
            const int ix = static_cast<int>(t);

            tx = ix * args.tile_size[0];
            ty = iy * args.tile_size[1];
            tz = iz * args.tile_size[2];
        }

        template <bool NeedsImag>
        KOKKOS_FORCEINLINE_FUNCTION static void scatter_particle_fast(
            const team_member& team, const int sx, const int sy, const int sz, const size_t pitch0,
            const size_t pitch1, const RealType* __restrict__ kw, const RealType vr,
            const RealType vi, RealType* __restrict__ local_r, RealType* __restrict__ local_i) {
            const RealType* __restrict__ kwx = kw + 0 * W;
            const RealType* __restrict__ kwy = kw + 1 * W;
            const RealType* __restrict__ kwz = kw + 2 * W;

            constexpr int plane = W * W;

#pragma unroll 1
            for (int idx = team.team_rank(); idx < stencil_total; idx += team.team_size()) {
                const int zz  = idx / plane;
                const int rem = idx - zz * plane;
                const int yy  = rem / W;
                const int xx  = rem - yy * W;

                const size_t hidx = static_cast<size_t>(sx + xx)
                                    + static_cast<size_t>(sy + yy) * pitch0
                                    + static_cast<size_t>(sz + zz) * pitch1;

                const RealType w = kwx[xx] * kwy[yy] * kwz[zz];
                local_r[hidx] += vr * w;
                if constexpr (NeedsImag)
                    local_i[hidx] += vi * w;
            }
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            const size_t league_r = static_cast<size_t>(team.league_rank());

            size_t tile_id;
            size_t sub_id;
            if constexpr (fixed_oversubscription) {
                tile_id = league_r;
                sub_id  = 0;
            } else {
                tile_id = league_r / sub_teams_per_tile_;
                sub_id  = league_r - tile_id * sub_teams_per_tile_;
            }

            const size_t bin_start = args.bin_offsets(tile_id);
            const size_t bin_end   = args.bin_offsets(tile_id + 1);
            const size_t bin_size  = bin_end - bin_start;
            if (bin_size == 0)
                return;

            size_t active_subteams = 1;
            if constexpr (!fixed_oversubscription) {
                const size_t target_particles_per_subteam =
                    static_cast<size_t>(Kokkos::max(128, 4 * args.team_size));

                active_subteams =
                    Kokkos::min(sub_teams_per_tile_,
                                Kokkos::max<size_t>(1, (bin_size + target_particles_per_subteam - 1)
                                                           / target_particles_per_subteam));

                if (sub_id >= active_subteams)
                    return;
            }

            const size_t particles_per_sub = (bin_size + active_subteams - 1) / active_subteams;
            const size_t pstart            = bin_start + sub_id * particles_per_sub;
            if (pstart >= bin_end)
                return;
            const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);

            const int tile_x = args.tile_size[0];
            const int tile_y = args.tile_size[1];
            const int tile_z = args.tile_size[2];

            const int hs0 = tile_x + padded_extra;
            const int hs1 = tile_y + padded_extra;
            const int hs2 = tile_z + padded_extra;

            const int pad0 = SharedBankLayout<RealType>::minimal_coprime_pad(hs0);
            const int pad1 = SharedBankLayout<RealType>::minimal_coprime_pad(hs1);

            const size_t pitch0 = static_cast<size_t>(hs0 + pad0);
            const size_t phys_y = static_cast<size_t>(hs1 + pad1);
            const size_t pitch1 = pitch0 * phys_y;

            const size_t logical_total = logical_total_;
            const size_t phys_total    = physical_total_;
            const int batch_np         = args.batch_np;

            int tile_base_x, tile_base_y, tile_base_z;
            decode_tile_base(tile_id, tile_base_x, tile_base_y, tile_base_z);

            auto scratch = team.team_scratch(0);

            scratch_real_view local_r_v(scratch, phys_total);
            RealType* const local_r = local_r_v.data();

            RealType* local_i = nullptr;
            scratch_real_view local_i_v;
            if constexpr (needs_imag) {
                local_i_v = scratch_real_view(scratch, phys_total);
                local_i   = local_i_v.data();
            }

            scratch_real_view kerevals_v(scratch, static_cast<size_t>(batch_np) * 3 * W);
            RealType* const kerevals = kerevals_v.data();

            scratch_real_view vals_r_v(scratch, batch_np);
            RealType* const vals_r = vals_r_v.data();

            RealType* vals_i = nullptr;
            scratch_real_view vals_i_v;
            if constexpr (needs_imag) {
                vals_i_v = scratch_real_view(scratch, batch_np);
                vals_i   = vals_i_v.data();
            }

            scratch_shift_view shifts_v(scratch, batch_np);
            Shift3* const shifts = shifts_v.data();

            for (size_t i = static_cast<size_t>(team.team_rank()); i < phys_total;
                 i += static_cast<size_t>(team.team_size())) {
                local_r[i] = RealType(0);
                if constexpr (needs_imag)
                    local_i[i] = RealType(0);
            }
            team.team_barrier();

            const CoordinateTransform<RealType, 3> transform{args.origin, args.invdx, args.n_grid};

            for (size_t batch_begin = pstart; batch_begin < pend;
                 batch_begin += static_cast<size_t>(batch_np)) {
                const int batch_size = static_cast<int>(
                    Kokkos::min(pend - batch_begin, static_cast<size_t>(batch_np)));

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, batch_size), [&](const int bi) {
                    const size_t p = args.permute(batch_begin + static_cast<size_t>(bi));

                    const RealType gp0 = transform.template toGridCoordinate<0>(args.x(p)[0]);
                    const RealType gp1 = transform.template toGridCoordinate<1>(args.x(p)[1]);
                    const RealType gp2 = transform.template toGridCoordinate<2>(args.x(p)[2]);

                    const int idx0 = transform.template getStencilBase<W>(gp0 - RealType(0.5));
                    const int idx1 = transform.template getStencilBase<W>(gp1 - RealType(0.5));
                    const int idx2 = transform.template getStencilBase<W>(gp2 - RealType(0.5));

                    RealType* const kw = kerevals + static_cast<size_t>(bi) * 3 * W;

#pragma unroll
                    for (int wi = 0; wi < W; ++wi) {
                        kw[0 * W + wi] = args.kernel((gp0 - (RealType(idx0 + wi) + RealType(0.5)))
                                                     * args.inv_hw);
                        kw[1 * W + wi] = args.kernel((gp1 - (RealType(idx1 + wi) + RealType(0.5)))
                                                     * args.inv_hw);
                        kw[2 * W + wi] = args.kernel((gp2 - (RealType(idx2 + wi) + RealType(0.5)))
                                                     * args.inv_hw);
                    }

                    shifts[bi] = Shift3{idx0 - args.local_offset[0] + half_left - tile_base_x,
                                        idx1 - args.local_offset[1] + half_left - tile_base_y,
                                        idx2 - args.local_offset[2] + half_left - tile_base_z, 0};

                    if constexpr (value_complex) {
                        const auto v = args.values(p);
                        vals_r[bi]   = v.real();
                        if constexpr (needs_imag)
                            vals_i[bi] = v.imag();
                    } else {
                        vals_r[bi] = static_cast<RealType>(args.values(p));
                    }
                });
                team.team_barrier();

                // With the current binning (center-based) and local particle ownership,
                // shift is guaranteed to satisfy 0 <= shift <= tile_size.
                for (int bi = 0; bi < batch_size; ++bi) {
                    const Shift3 sh          = shifts[bi];
                    const RealType* const kw = kerevals + static_cast<size_t>(bi) * 3 * W;
                    const RealType vr        = vals_r[bi];
                    const RealType vi        = (needs_imag ? vals_i[bi] : RealType(0));

                    scatter_particle_fast<needs_imag>(team, sh.x, sh.y, sh.z, pitch0, pitch1, kw,
                                                      vr, vi, local_r, local_i);
                    team.team_barrier();
                }
            }

            const size_t plane_logical = static_cast<size_t>(hs0) * static_cast<size_t>(hs1);

            for (size_t lid = static_cast<size_t>(team.team_rank()); lid < logical_total;
                 lid += static_cast<size_t>(team.team_size())) {
                const int kz   = static_cast<int>(lid / plane_logical);
                const size_t r = lid - static_cast<size_t>(kz) * plane_logical;
                const int jy   = static_cast<int>(r / static_cast<size_t>(hs0));
                const int ix =
                    static_cast<int>(r - static_cast<size_t>(jy) * static_cast<size_t>(hs0));

                const size_t sidx = static_cast<size_t>(ix) + static_cast<size_t>(jy) * pitch0
                                    + static_cast<size_t>(kz) * pitch1;

                const RealType rr = local_r[sidx];
                if constexpr (needs_imag) {
                    const RealType ii = local_i[sidx];
                    if (rr == RealType(0) && ii == RealType(0))
                        continue;
                } else {
                    if (rr == RealType(0))
                        continue;
                }

                const int local_x = tile_base_x + ix - half_left;
                const int local_y = tile_base_y + jy - half_left;
                const int local_z = tile_base_z + kz - half_left;

                if (local_x < -args.nghost || local_x >= args.n_grid_local[0] + args.nghost
                    || local_y < -args.nghost || local_y >= args.n_grid_local[1] + args.nghost
                    || local_z < -args.nghost || local_z >= args.n_grid_local[2] + args.nghost)
                    continue;

                const int gx = local_x + args.nghost;
                const int gy = local_y + args.nghost;
                const int gz = local_z + args.nghost;

                if constexpr (grid_complex) {
                    RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gx, gy, gz));
                    Kokkos::atomic_add(&ptr[0], rr);
                    if constexpr (needs_imag)
                        Kokkos::atomic_add(&ptr[1], local_i[sidx]);
                } else {
                    Kokkos::atomic_add(&args.grid(gx, gy, gz), static_cast<grid_value_t>(rr));
                }
            }
        }

        template <bool NeedsImag>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /*team_size*/,
                                           int z_batches) {
            const int batch_np = z_batches > 0 ? z_batches : 1;

            const int hs0 = tile_size[0] + padded_extra;
            const int hs1 = tile_size[1] + padded_extra;
            const int hs2 = tile_size[2] + padded_extra;

            const int pad0 = SharedBankLayout<RealType>::minimal_coprime_pad(hs0);
            const int pad1 = SharedBankLayout<RealType>::minimal_coprime_pad(hs1);

            const size_t pitch0     = static_cast<size_t>(hs0 + pad0);
            const size_t phys_y     = static_cast<size_t>(hs1 + pad1);
            const size_t phys_total = pitch0 * phys_y * static_cast<size_t>(hs2);

            size_t s = 0;
            s += scratch_real_view::shmem_size(phys_total);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(phys_total);

            s += scratch_real_view::shmem_size(static_cast<size_t>(batch_np) * 3 * W);
            s += scratch_real_view::shmem_size(batch_np);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(batch_np);

            s += scratch_shift_view::shmem_size(batch_np);
            return s;
        }

        void run(size_t n_particles) {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            size_t n_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n_tiles *= static_cast<size_t>(args.num_tiles[d]);

            if (n_tiles == 0 || n_particles == 0)
                return;

            const int hs0 = args.tile_size[0] + padded_extra;
            const int hs1 = args.tile_size[1] + padded_extra;
            const int hs2 = args.tile_size[2] + padded_extra;

            const int pad0 = SharedBankLayout<RealType>::minimal_coprime_pad(hs0);
            const int pad1 = SharedBankLayout<RealType>::minimal_coprime_pad(hs1);

            logical_total_ =
                static_cast<size_t>(hs0) * static_cast<size_t>(hs1) * static_cast<size_t>(hs2);
            physical_total_ = static_cast<size_t>(hs0 + pad0) * static_cast<size_t>(hs1 + pad1)
                              * static_cast<size_t>(hs2);

            size_t league_size;
            if constexpr (fixed_oversubscription) {
                sub_teams_per_tile_ = 1;
                league_size         = n_tiles;
            } else {
                sub_teams_per_tile_ =
                    std::max<size_t>(1, static_cast<size_t>(args.oversubscription_factor));
                league_size = n_tiles * sub_teams_per_tile_;
            }

            const size_t scratch =
                compute_scratch_size<needs_imag>(args.tile_size, args.team_size, args.batch_np);

            auto policy = team_policy(league_size, args.team_size, 1)
                              .set_scratch_size(0, Kokkos::PerTeam(scratch));

            Kokkos::parallel_for("GridParallelScatterOutputDrivenBatched3D", policy, *this);
        }
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // Generic Dim != 3 fallback
    // ─────────────────────────────────────────────────────────────────────────────

    template <int W, class Types, class Policy>
    struct GridParallelScatterImplND {
        static_assert(Policy::use_sorting,
                      "GridParallelScatter assumes sorted/bin-partitioned particles");
        static_assert(Types::Dim != 3, "GridParallelScatterImplND is the non-3D fallback");

        static constexpr bool requires_binning = true;
        static constexpr unsigned Dim          = Types::Dim;
        static constexpr int half_left         = (W + 1) / 2;
        static constexpr int padded_extra      = 2 * half_left;
        static constexpr int stencil_total     = StaticPow<W, Dim>::value;
        static constexpr bool fixed_oversubscription =
            GridParallelScatterTuning<Policy>::fixed_oversubscription;

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

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            Kokkos::View<size_t*, memory_space> permute;
            Kokkos::View<size_t*, memory_space> bin_offsets;
            Vector<int, Dim> num_tiles;
            Vector<int, Dim> tile_size;
            int team_size;
            int oversubscription_factor;
            int batch_np;

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
                a.batch_np                = config.z_batches > 0 ? config.z_batches : 1;
                return a;
            }
        };

        Arguments args;
        size_t logical_total_      = 1;
        size_t physical_total_     = 1;
        size_t sub_teams_per_tile_ = 1;

        KOKKOS_INLINE_FUNCTION Vector<int, Dim> logical_hist_size() const {
            Vector<int, Dim> hs;
            for (unsigned d = 0; d < Dim; ++d)
                hs[d] = args.tile_size[d] + padded_extra;
            return hs;
        }

        KOKKOS_INLINE_FUNCTION Vector<int, Dim> decode_tile_base(size_t tile_id) const {
            Vector<int, Dim> tile_base;
            for (size_t t = tile_id, d = Dim; d-- > 0;) {
                const int tile_coord = static_cast<int>(t % static_cast<size_t>(args.num_tiles[d]));
                tile_base[d]         = tile_coord * args.tile_size[d];
                t /= static_cast<size_t>(args.num_tiles[d]);
            }
            return tile_base;
        }

        template <bool NeedsImag>
        KOKKOS_FORCEINLINE_FUNCTION static void scatter_particle_fast(
            const team_member& team, const int* __restrict__ shift,
            const size_t* __restrict__ stride, const RealType* __restrict__ kw, const RealType vr,
            const RealType vi, RealType* __restrict__ local_r, RealType* __restrict__ local_i) {
            for (int idx = team.team_rank(); idx < stencil_total; idx += team.team_size()) {
                int tmp     = idx;
                size_t hidx = 0;
                RealType w  = RealType(1);

                for (unsigned d = 0; d < Dim; ++d) {
                    const int wi = tmp % W;
                    tmp /= W;
                    hidx += static_cast<size_t>(shift[d] + wi) * stride[d];
                    w *= kw[d * W + wi];
                }

                local_r[hidx] += vr * w;
                if constexpr (NeedsImag)
                    local_i[hidx] += vi * w;
            }
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            const size_t league_r = static_cast<size_t>(team.league_rank());

            size_t tile_id;
            size_t sub_id;
            if constexpr (fixed_oversubscription) {
                tile_id = league_r;
                sub_id  = 0;
            } else {
                tile_id = league_r / sub_teams_per_tile_;
                sub_id  = league_r - tile_id * sub_teams_per_tile_;
            }

            const size_t bin_start = args.bin_offsets(tile_id);
            const size_t bin_end   = args.bin_offsets(tile_id + 1);
            const size_t bin_size  = bin_end - bin_start;
            if (bin_size == 0)
                return;

            size_t active_subteams = 1;
            if constexpr (!fixed_oversubscription) {
                const size_t target_particles_per_subteam =
                    static_cast<size_t>(Kokkos::max(128, 4 * args.team_size));

                active_subteams =
                    Kokkos::min(sub_teams_per_tile_,
                                Kokkos::max<size_t>(1, (bin_size + target_particles_per_subteam - 1)
                                                           / target_particles_per_subteam));

                if (sub_id >= active_subteams)
                    return;
            }

            const size_t particles_per_sub = (bin_size + active_subteams - 1) / active_subteams;
            const size_t pstart            = bin_start + sub_id * particles_per_sub;
            if (pstart >= bin_end)
                return;
            const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);

            const auto hs_logical      = logical_hist_size();
            const auto tile_base       = decode_tile_base(tile_id);
            const size_t logical_total = logical_total_;
            const size_t phys_total    = physical_total_;
            const int batch_np         = args.batch_np;

            int hs[Dim];
            int phys_extent[Dim];
            size_t stride[Dim];

            for (unsigned d = 0; d < Dim; ++d) {
                hs[d]          = hs_logical[d];
                phys_extent[d] = hs[d];
            }
            if constexpr (Dim >= 1)
                phys_extent[0] += SharedBankLayout<RealType>::minimal_coprime_pad(hs[0]);
            if constexpr (Dim >= 2)
                phys_extent[1] += SharedBankLayout<RealType>::minimal_coprime_pad(hs[1]);

            stride[0] = 1;
            for (unsigned d = 1; d < Dim; ++d)
                stride[d] = stride[d - 1] * static_cast<size_t>(phys_extent[d - 1]);

            auto scratch = team.team_scratch(0);

            scratch_real_view local_r_v(scratch, phys_total);
            RealType* const local_r = local_r_v.data();

            RealType* local_i = nullptr;
            scratch_real_view local_i_v;
            if constexpr (needs_imag) {
                local_i_v = scratch_real_view(scratch, phys_total);
                local_i   = local_i_v.data();
            }

            scratch_real_view kerevals_v(scratch, static_cast<size_t>(batch_np) * Dim * W);
            RealType* const kerevals = kerevals_v.data();

            scratch_real_view vals_r_v(scratch, batch_np);
            RealType* const vals_r = vals_r_v.data();

            RealType* vals_i = nullptr;
            scratch_real_view vals_i_v;
            if constexpr (needs_imag) {
                vals_i_v = scratch_real_view(scratch, batch_np);
                vals_i   = vals_i_v.data();
            }

            scratch_int_view shifts_v(scratch, static_cast<size_t>(batch_np) * Dim);
            int* const shifts = shifts_v.data();

            for (size_t i = static_cast<size_t>(team.team_rank()); i < phys_total;
                 i += static_cast<size_t>(team.team_size())) {
                local_r[i] = RealType(0);
                if constexpr (needs_imag)
                    local_i[i] = RealType(0);
            }
            team.team_barrier();

            const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
                                                               args.n_grid};

            for (size_t batch_begin = pstart; batch_begin < pend;
                 batch_begin += static_cast<size_t>(batch_np)) {
                const int batch_size = static_cast<int>(
                    Kokkos::min(pend - batch_begin, static_cast<size_t>(batch_np)));

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, batch_size), [&](const int bi) {
                    const size_t p = args.permute(batch_begin + static_cast<size_t>(bi));

                    RealType* const kw = kerevals + static_cast<size_t>(bi) * Dim * W;

                    for (unsigned d = 0; d < Dim; ++d) {
                        const RealType gp = transform.toGridCoordinate(args.x(p)[d], d);
                        const int idx0    = transform.getStencilBase(gp - RealType(0.5), W);

                        for (int wi = 0; wi < W; ++wi) {
                            kw[d * W + wi] = args.kernel(
                                (gp - (RealType(idx0 + wi) + RealType(0.5))) * args.inv_hw);
                        }

                        shifts[static_cast<size_t>(bi) * Dim + d] =
                            idx0 - args.local_offset[d] + half_left - tile_base[d];
                    }

                    if constexpr (value_complex) {
                        const auto v = args.values(p);
                        vals_r[bi]   = v.real();
                        if constexpr (needs_imag)
                            vals_i[bi] = v.imag();
                    } else {
                        vals_r[bi] = static_cast<RealType>(args.values(p));
                    }
                });
                team.team_barrier();

                for (int bi = 0; bi < batch_size; ++bi) {
                    int shift[Dim];
                    for (unsigned d = 0; d < Dim; ++d)
                        shift[d] = shifts[static_cast<size_t>(bi) * Dim + d];

                    const RealType* const kw = kerevals + static_cast<size_t>(bi) * Dim * W;
                    const RealType vr        = vals_r[bi];
                    const RealType vi        = (needs_imag ? vals_i[bi] : RealType(0));

                    scatter_particle_fast<needs_imag>(team, shift, stride, kw, vr, vi, local_r,
                                                      local_i);
                    team.team_barrier();
                }
            }

            for (size_t lid = static_cast<size_t>(team.team_rank()); lid < logical_total;
                 lid += static_cast<size_t>(team.team_size())) {
                size_t tmp = lid;
                int coord[Dim];
                size_t sidx = 0;

                for (unsigned d = 0; d < Dim; ++d) {
                    coord[d] = static_cast<int>(tmp % static_cast<size_t>(hs[d]));
                    tmp /= static_cast<size_t>(hs[d]);
                    sidx += static_cast<size_t>(coord[d]) * stride[d];
                }

                const RealType rr = local_r[sidx];
                if constexpr (needs_imag) {
                    const RealType ii = local_i[sidx];
                    if (rr == RealType(0) && ii == RealType(0))
                        continue;
                } else {
                    if (rr == RealType(0))
                        continue;
                }

                Kokkos::Array<int, Dim> gc{};
                for (unsigned d = 0; d < Dim; ++d) {
                    const int local = tile_base[d] + coord[d] - half_left;
                    if (local < -args.nghost || local >= args.n_grid_local[d] + args.nghost)
                        goto skip_flush_nd;
                    gc[d] = local + args.nghost;
                }

                [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    if constexpr (grid_complex) {
                        RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                        Kokkos::atomic_add(&ptr[0], rr);
                        if constexpr (needs_imag)
                            Kokkos::atomic_add(&ptr[1], local_i[sidx]);
                    } else {
                        Kokkos::atomic_add(&args.grid(gc[Is]...), static_cast<grid_value_t>(rr));
                    }
                }(std::make_index_sequence<Dim>{});

skip_flush_nd:;
            }
        }

        template <bool NeedsImag>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /*team_size*/,
                                           int z_batches) {
            const int batch_np = z_batches > 0 ? z_batches : 1;

            int hs[Dim];
            int phys_extent[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                hs[d]          = tile_size[d] + padded_extra;
                phys_extent[d] = hs[d];
            }
            if constexpr (Dim >= 1)
                phys_extent[0] += SharedBankLayout<RealType>::minimal_coprime_pad(hs[0]);
            if constexpr (Dim >= 2)
                phys_extent[1] += SharedBankLayout<RealType>::minimal_coprime_pad(hs[1]);

            size_t phys_total = 1;
            for (unsigned d = 0; d < Dim; ++d)
                phys_total *= static_cast<size_t>(phys_extent[d]);

            size_t s = 0;
            s += scratch_real_view::shmem_size(phys_total);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(phys_total);

            s += scratch_real_view::shmem_size(static_cast<size_t>(batch_np) * Dim * W);
            s += scratch_real_view::shmem_size(batch_np);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(batch_np);

            s += scratch_int_view::shmem_size(static_cast<size_t>(batch_np) * Dim);
            return s;
        }

        void run(size_t n_particles) {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            size_t n_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n_tiles *= static_cast<size_t>(args.num_tiles[d]);

            if (n_tiles == 0 || n_particles == 0)
                return;

            const auto hs_logical = logical_hist_size();

            logical_total_  = 1;
            physical_total_ = 1;
            for (unsigned d = 0; d < Dim; ++d)
                logical_total_ *= static_cast<size_t>(hs_logical[d]);

            for (unsigned d = 0; d < Dim; ++d) {
                int phys_d = hs_logical[d];
                if (d == 0)
                    phys_d += SharedBankLayout<RealType>::minimal_coprime_pad(hs_logical[d]);
                else if (d == 1)
                    phys_d += SharedBankLayout<RealType>::minimal_coprime_pad(hs_logical[d]);
                physical_total_ *= static_cast<size_t>(phys_d);
            }

            size_t league_size;
            if constexpr (fixed_oversubscription) {
                sub_teams_per_tile_ = 1;
                league_size         = n_tiles;
            } else {
                sub_teams_per_tile_ =
                    std::max<size_t>(1, static_cast<size_t>(args.oversubscription_factor));
                league_size = n_tiles * sub_teams_per_tile_;
            }

            const size_t scratch =
                compute_scratch_size<needs_imag>(args.tile_size, args.team_size, args.batch_np);

            auto policy = team_policy(league_size, args.team_size, 1)
                              .set_scratch_size(0, Kokkos::PerTeam(scratch));

            Kokkos::parallel_for("GridParallelScatterOutputDrivenBatchedND", policy, *this);
        }
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // Public selector
    // ─────────────────────────────────────────────────────────────────────────────

    template <int W, class Types, class Policy>
    using GridParallelScatter =
        std::conditional_t<(Types::Dim == 3), GridParallelScatterImpl3D<W, Types, Policy>,
                           GridParallelScatterImplND<W, Types, Policy>>;

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_GRID_PARALLEL_SCATTER_H