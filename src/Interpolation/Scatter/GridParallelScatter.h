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
//             int batch_np;  // runtime batch size, driven by config.z_batches
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
//                 hs[d] = args.tile_size[d] + W + 1;
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
//         KOKKOS_INLINE_FUNCTION static void scatter_particle_fast(
//             const team_member& team, const Kokkos::Array<int, Dim>& shift,
//             const Kokkos::Array<size_t, Dim>& stride, const RealType* kw, const RealType vr,
//             const RealType vi, const scratch_real_view& local_r, const scratch_real_view&
//             local_i) { if constexpr (Dim == 3) {
//                 constexpr int plane = W * W;
//                 constexpr int total = W * W * W;
//
//                 Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total), [&](const int idx) {
//                     const int zz  = idx / plane;
//                     const int rem = idx - zz * plane;
//                     const int yy  = rem / W;
//                     const int xx  = rem - yy * W;
//
//                     const size_t hidx = static_cast<size_t>(shift[0] + xx) * stride[0]
//                                         + static_cast<size_t>(shift[1] + yy) * stride[1]
//                                         + static_cast<size_t>(shift[2] + zz) * stride[2];
//
//                     const RealType w = kw[0 * W + xx] * kw[1 * W + yy] * kw[2 * W + zz];
//                     local_r(hidx) += vr * w;
//                     if constexpr (NeedsImag)
//                         local_i(hidx) += vi * w;
//                 });
//             } else {
//                 Kokkos::parallel_for(Kokkos::TeamThreadRange(team, stencil_total),
//                                      [&](const int idx) {
//                                          int tmp     = idx;
//                                          size_t hidx = 0;
//                                          RealType w  = RealType(1);
//
//                                          for (unsigned d = 0; d < Dim; ++d) {
//                                              const int wi = tmp % W;
//                                              tmp /= W;
//                                              hidx += static_cast<size_t>(shift[d] + wi) *
//                                              stride[d]; w *= kw[d * W + wi];
//                                          }
//
//                                          local_r(hidx) += vr * w;
//                                          if constexpr (NeedsImag)
//                                              local_i(hidx) += vi * w;
//                                      });
//             }
//         }
//
//         template <bool NeedsImag>
//         KOKKOS_INLINE_FUNCTION static void scatter_particle_checked(
//             const team_member& team, const Kokkos::Array<int, Dim>& shift,
//             const Kokkos::Array<int, Dim>& hs, const Kokkos::Array<size_t, Dim>& stride,
//             const RealType* kw, const RealType vr, const RealType vi,
//             const scratch_real_view& local_r, const scratch_real_view& local_i) {
//             if constexpr (Dim == 3) {
//                 constexpr int plane = W * W;
//                 constexpr int total = W * W * W;
//
//                 Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total), [&](const int idx) {
//                     const int zz  = idx / plane;
//                     const int rem = idx - zz * plane;
//                     const int yy  = rem / W;
//                     const int xx  = rem - yy * W;
//
//                     const int hx = shift[0] + xx;
//                     const int hy = shift[1] + yy;
//                     const int hz = shift[2] + zz;
//
//                     if (static_cast<unsigned>(hx) >= static_cast<unsigned>(hs[0])
//                         || static_cast<unsigned>(hy) >= static_cast<unsigned>(hs[1])
//                         || static_cast<unsigned>(hz) >= static_cast<unsigned>(hs[2]))
//                         return;
//
//                     const size_t hidx = static_cast<size_t>(hx) * stride[0]
//                                         + static_cast<size_t>(hy) * stride[1]
//                                         + static_cast<size_t>(hz) * stride[2];
//
//                     const RealType w = kw[0 * W + xx] * kw[1 * W + yy] * kw[2 * W + zz];
//                     local_r(hidx) += vr * w;
//                     if constexpr (NeedsImag)
//                         local_i(hidx) += vi * w;
//                 });
//             } else {
//                 Kokkos::parallel_for(
//                     Kokkos::TeamThreadRange(team, stencil_total), [&](const int idx) {
//                         int tmp     = idx;
//                         size_t hidx = 0;
//                         RealType w  = RealType(1);
//
//                         for (unsigned d = 0; d < Dim; ++d) {
//                             const int wi    = tmp % W;
//                             const int coord = shift[d] + wi;
//                             tmp /= W;
//
//                             if (static_cast<unsigned>(coord) >= static_cast<unsigned>(hs[d]))
//                                 return;
//
//                             hidx += static_cast<size_t>(coord) * stride[d];
//                             w *= kw[d * W + wi];
//                         }
//
//                         local_r(hidx) += vr * w;
//                         if constexpr (NeedsImag)
//                             local_i(hidx) += vi * w;
//                     });
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
//             Kokkos::parallel_for(Kokkos::TeamThreadRange(team, htot), [&](const size_t i) {
//                 local_r(i) = RealType(0);
//                 if constexpr (needs_imag)
//                     local_i(i) = RealType(0);
//             });
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
//                     for (unsigned d = 0; d < Dim; ++d) {
//                         const RealType gp = transform.toGridCoordinate(args.x(p)[d], d);
//                         const int idx0    = transform.getStencilBase(gp - RealType(0.5), W);
//
//                         for (int wi = 0; wi < W; ++wi) {
//                             kerevals(static_cast<size_t>(bi) * Dim * W + d * W + wi) =
//                             args.kernel(
//                                 (gp - (RealType(idx0 + wi) + RealType(0.5))) * args.inv_hw);
//                         }
//
//                         shifts(static_cast<size_t>(bi) * Dim + d) =
//                             idx0 - args.local_offset[d] + half_left - tile_base[d];
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
//                     Kokkos::Array<int, Dim> shift{};
//                     bool full_inside = true;
//                     for (unsigned d = 0; d < Dim; ++d) {
//                         shift[d] = shifts(static_cast<size_t>(bi) * Dim + d);
//                         if (shift[d] < 0 || shift[d] > hs[d] - W)
//                             full_inside = false;
//                     }
//
//                     const RealType* kw = kerevals.data() + static_cast<size_t>(bi) * Dim * W;
//                     const RealType vr  = vals_r(bi);
//                     const RealType vi  = (needs_imag ? vals_i(bi) : RealType(0));
//
//                     if (full_inside) {
//                         scatter_particle_fast<needs_imag>(team, shift, stride, kw, vr, vi,
//                         local_r,
//                                                           local_i);
//                     } else {
//                         scatter_particle_checked<needs_imag>(team, shift, hs, stride, kw, vr, vi,
//                                                              local_r, local_i);
//                     }
//                     team.team_barrier();
//                 }
//             }
//
//             Kokkos::parallel_for(Kokkos::TeamThreadRange(team, htot), [&](const size_t idx) {
//                 const RealType rr = local_r(idx);
//
//                 if constexpr (needs_imag) {
//                     const RealType ii = local_i(idx);
//                     if (rr == RealType(0) && ii == RealType(0))
//                         return;
//                 } else {
//                     if (rr == RealType(0))
//                         return;
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
//                         return;
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
//             });
//         }
//
//         template <bool NeedsImag>
//         static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /*team_size*/,
//                                            int z_batches) {
//             const int batch_np = z_batches > 0 ? z_batches : 1;
//
//             size_t htot = 1;
//             for (unsigned d = 0; d < Dim; ++d)
//                 htot *= static_cast<size_t>(tile_size[d] + W + 1);
//
//             size_t s = 0;
//             s += scratch_real_view::shmem_size(htot);
//             if constexpr (NeedsImag)
//                 s += scratch_real_view::shmem_size(htot);
//
//             s +=
//                 scratch_real_view::shmem_size(static_cast<size_t>(batch_np) * Dim * W);  //
//                 kerevals
//             s += scratch_real_view::shmem_size(batch_np);                                //
//             vals_r if constexpr (NeedsImag)
//                 s += scratch_real_view::shmem_size(batch_np);  // vals_i
//
//             s += scratch_int_view::shmem_size(static_cast<size_t>(batch_np) * Dim);  // shifts
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
//                 hist_total_ *= static_cast<size_t>(args.tile_size[d] + W + 1);
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
// #endif  // IPPL_GRID_PARALLEL_SCATTER_H

#ifndef IPPL_GRID_PARALLEL_SCATTER_H
#define IPPL_GRID_PARALLEL_SCATTER_H

namespace ippl::Interpolation::detail {

    template <int Base, int Exp>
    struct StaticPow {
        static constexpr int value = Base * StaticPow<Base, Exp - 1>::value;
    };
    template <int Base>
    struct StaticPow<Base, 0> {
        static constexpr int value = 1;
    };

    template <int W, class Types, class Policy>
    struct GridParallelScatter {
        static_assert(Policy::use_sorting,
                      "GridParallelScatter assumes sorted/bin-partitioned particles");

        static constexpr bool requires_binning = true;
        static constexpr unsigned Dim          = Types::Dim;
        static constexpr int half_left         = (W + 1) / 2;
        static constexpr int padded_extra      = 2 * half_left;
        static constexpr int stencil_total     = StaticPow<W, Dim>::value;

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
        size_t hist_total_         = 1;
        size_t sub_teams_per_tile_ = 1;

        KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
            Vector<int, Dim> hs;
            for (unsigned d = 0; d < Dim; ++d)
                hs[d] = args.tile_size[d] + padded_extra;
            return hs;
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

        template <bool NeedsImag>
        KOKKOS_INLINE_FUNCTION static void scatter_particle_fast_3d(
            const team_member& team, const int sx, const int sy, const int sz, const size_t s1,
            const size_t s2, const RealType* kw, const RealType vr, const RealType vi,
            const scratch_real_view& local_r, const scratch_real_view& local_i) {
            constexpr int plane = W * W;
            constexpr int total = W * W * W;

            const RealType* kwx = kw + 0 * W;
            const RealType* kwy = kw + 1 * W;
            const RealType* kwz = kw + 2 * W;

            for (int idx = team.team_rank(); idx < total; idx += team.team_size()) {
                const int zz  = idx / plane;
                const int rem = idx - zz * plane;
                const int yy  = rem / W;
                const int xx  = rem - yy * W;

                const size_t hidx = static_cast<size_t>(sx + xx) + static_cast<size_t>(sy + yy) * s1
                                    + static_cast<size_t>(sz + zz) * s2;

                const RealType w = kwx[xx] * kwy[yy] * kwz[zz];
                local_r(hidx) += vr * w;
                if constexpr (NeedsImag)
                    local_i(hidx) += vi * w;
            }
        }

        template <bool NeedsImag>
        KOKKOS_INLINE_FUNCTION static void scatter_particle_checked_3d(
            const team_member& team, const int sx, const int sy, const int sz, const int hs0,
            const int hs1, const int hs2, const size_t s1, const size_t s2, const RealType* kw,
            const RealType vr, const RealType vi, const scratch_real_view& local_r,
            const scratch_real_view& local_i) {
            constexpr int plane = W * W;
            constexpr int total = W * W * W;

            const RealType* kwx = kw + 0 * W;
            const RealType* kwy = kw + 1 * W;
            const RealType* kwz = kw + 2 * W;

            for (int idx = team.team_rank(); idx < total; idx += team.team_size()) {
                const int zz  = idx / plane;
                const int rem = idx - zz * plane;
                const int yy  = rem / W;
                const int xx  = rem - yy * W;

                const int hx = sx + xx;
                const int hy = sy + yy;
                const int hz = sz + zz;

                if (static_cast<unsigned>(hx) >= static_cast<unsigned>(hs0)
                    || static_cast<unsigned>(hy) >= static_cast<unsigned>(hs1)
                    || static_cast<unsigned>(hz) >= static_cast<unsigned>(hs2))
                    continue;

                const size_t hidx = static_cast<size_t>(hx) + static_cast<size_t>(hy) * s1
                                    + static_cast<size_t>(hz) * s2;

                const RealType w = kwx[xx] * kwy[yy] * kwz[zz];
                local_r(hidx) += vr * w;
                if constexpr (NeedsImag)
                    local_i(hidx) += vi * w;
            }
        }

        template <bool NeedsImag>
        KOKKOS_INLINE_FUNCTION static void scatter_particle_fast_nd(
            const team_member& team, const Kokkos::Array<int, Dim>& shift,
            const Kokkos::Array<size_t, Dim>& stride, const RealType* kw, const RealType vr,
            const RealType vi, const scratch_real_view& local_r, const scratch_real_view& local_i) {
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

                local_r(hidx) += vr * w;
                if constexpr (NeedsImag)
                    local_i(hidx) += vi * w;
            }
        }

        template <bool NeedsImag>
        KOKKOS_INLINE_FUNCTION static void scatter_particle_checked_nd(
            const team_member& team, const Kokkos::Array<int, Dim>& shift,
            const Kokkos::Array<int, Dim>& hs, const Kokkos::Array<size_t, Dim>& stride,
            const RealType* kw, const RealType vr, const RealType vi,
            const scratch_real_view& local_r, const scratch_real_view& local_i) {
            for (int idx = team.team_rank(); idx < stencil_total; idx += team.team_size()) {
                int tmp     = idx;
                size_t hidx = 0;
                RealType w  = RealType(1);

                bool ok = true;
                for (unsigned d = 0; d < Dim; ++d) {
                    const int wi    = tmp % W;
                    const int coord = shift[d] + wi;
                    tmp /= W;

                    if (static_cast<unsigned>(coord) >= static_cast<unsigned>(hs[d])) {
                        ok = false;
                        break;
                    }

                    hidx += static_cast<size_t>(coord) * stride[d];
                    w *= kw[d * W + wi];
                }

                if (!ok)
                    continue;

                local_r(hidx) += vr * w;
                if constexpr (NeedsImag)
                    local_i(hidx) += vi * w;
            }
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            const size_t league_r = static_cast<size_t>(team.league_rank());
            const size_t tile_id  = league_r / sub_teams_per_tile_;
            const size_t sub_id   = league_r % sub_teams_per_tile_;

            const size_t bin_start = args.bin_offsets(tile_id);
            const size_t bin_end   = args.bin_offsets(tile_id + 1);
            const size_t bin_size  = bin_end - bin_start;
            if (bin_size == 0)
                return;

            const size_t target_particles_per_subteam =
                static_cast<size_t>(Kokkos::max(128, 4 * args.team_size));

            const size_t active_subteams =
                Kokkos::min(sub_teams_per_tile_,
                            Kokkos::max<size_t>(1, (bin_size + target_particles_per_subteam - 1)
                                                       / target_particles_per_subteam));

            if (sub_id >= active_subteams)
                return;

            const size_t particles_per_sub = (bin_size + active_subteams - 1) / active_subteams;
            const size_t pstart            = bin_start + sub_id * particles_per_sub;
            if (pstart >= bin_end)
                return;
            const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);

            const auto tile_base = decode_tile_base(tile_id);
            const auto hs_vec    = hist_size();
            const size_t htot    = hist_total_;
            const int batch_np   = args.batch_np;

            Kokkos::Array<int, Dim> hs{};
            Kokkos::Array<size_t, Dim> stride{};
            stride[0] = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                hs[d] = hs_vec[d];
                if (d > 0)
                    stride[d] = stride[d - 1] * static_cast<size_t>(hs[d - 1]);
            }

            const int hs0   = hs[0];
            const int hs1   = (Dim > 1 ? hs[1] : 1);
            const int hs2   = (Dim > 2 ? hs[2] : 1);
            const size_t s1 = (Dim > 1 ? stride[1] : 1);
            const size_t s2 = (Dim > 2 ? stride[2] : 1);

            auto scratch = team.team_scratch(0);

            scratch_real_view local_r(scratch, htot);
            scratch_real_view local_i;
            if constexpr (needs_imag)
                local_i = scratch_real_view(scratch, htot);

            scratch_real_view kerevals(scratch, static_cast<size_t>(batch_np) * Dim * W);
            scratch_real_view vals_r(scratch, batch_np);
            scratch_real_view vals_i;
            if constexpr (needs_imag)
                vals_i = scratch_real_view(scratch, batch_np);

            scratch_int_view shifts(scratch, static_cast<size_t>(batch_np) * Dim);

            for (size_t i = static_cast<size_t>(team.team_rank()); i < htot;
                 i += static_cast<size_t>(team.team_size())) {
                local_r(i) = RealType(0);
                if constexpr (needs_imag)
                    local_i(i) = RealType(0);
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

                    if constexpr (Dim == 3) {
                        const RealType gp0 = transform.template toGridCoordinate<0>(args.x(p)[0]);
                        const RealType gp1 = transform.template toGridCoordinate<1>(args.x(p)[1]);
                        const RealType gp2 = transform.template toGridCoordinate<2>(args.x(p)[2]);

                        const int idx0 = transform.template getStencilBase<W>(gp0 - RealType(0.5));
                        const int idx1 = transform.template getStencilBase<W>(gp1 - RealType(0.5));
                        const int idx2 = transform.template getStencilBase<W>(gp2 - RealType(0.5));

                        for (int wi = 0; wi < W; ++wi) {
                            kerevals(static_cast<size_t>(bi) * Dim * W + 0 * W + wi) = args.kernel(
                                (gp0 - (RealType(idx0 + wi) + RealType(0.5))) * args.inv_hw);
                            kerevals(static_cast<size_t>(bi) * Dim * W + 1 * W + wi) = args.kernel(
                                (gp1 - (RealType(idx1 + wi) + RealType(0.5))) * args.inv_hw);
                            kerevals(static_cast<size_t>(bi) * Dim * W + 2 * W + wi) = args.kernel(
                                (gp2 - (RealType(idx2 + wi) + RealType(0.5))) * args.inv_hw);
                        }

                        shifts(static_cast<size_t>(bi) * Dim + 0) =
                            idx0 - args.local_offset[0] + half_left - tile_base[0];
                        shifts(static_cast<size_t>(bi) * Dim + 1) =
                            idx1 - args.local_offset[1] + half_left - tile_base[1];
                        shifts(static_cast<size_t>(bi) * Dim + 2) =
                            idx2 - args.local_offset[2] + half_left - tile_base[2];
                    } else {
                        for (unsigned d = 0; d < Dim; ++d) {
                            const RealType gp = transform.toGridCoordinate(args.x(p)[d], d);
                            const int idx0    = transform.getStencilBase(gp - RealType(0.5), W);

                            for (int wi = 0; wi < W; ++wi) {
                                kerevals(static_cast<size_t>(bi) * Dim * W + d * W + wi) =
                                    args.kernel((gp - (RealType(idx0 + wi) + RealType(0.5)))
                                                * args.inv_hw);
                            }

                            shifts(static_cast<size_t>(bi) * Dim + d) =
                                idx0 - args.local_offset[d] + half_left - tile_base[d];
                        }
                    }

                    if constexpr (value_complex) {
                        vals_r(bi) = args.values(p).real();
                        if constexpr (needs_imag)
                            vals_i(bi) = args.values(p).imag();
                    } else {
                        vals_r(bi) = static_cast<RealType>(args.values(p));
                    }
                });
                team.team_barrier();

                for (int bi = 0; bi < batch_size; ++bi) {
                    const RealType* kw = kerevals.data() + static_cast<size_t>(bi) * Dim * W;
                    const RealType vr  = vals_r(bi);
                    const RealType vi  = (needs_imag ? vals_i(bi) : RealType(0));

                    if constexpr (Dim == 3) {
                        const int sx = shifts(static_cast<size_t>(bi) * Dim + 0);
                        const int sy = shifts(static_cast<size_t>(bi) * Dim + 1);
                        const int sz = shifts(static_cast<size_t>(bi) * Dim + 2);

                        const bool full_inside = (sx >= 0 && sx <= hs0 - W)
                                                 && (sy >= 0 && sy <= hs1 - W)
                                                 && (sz >= 0 && sz <= hs2 - W);

                        if (full_inside) {
                            scatter_particle_fast_3d<needs_imag>(team, sx, sy, sz, s1, s2, kw, vr,
                                                                 vi, local_r, local_i);
                        } else {
                            scatter_particle_checked_3d<needs_imag>(team, sx, sy, sz, hs0, hs1, hs2,
                                                                    s1, s2, kw, vr, vi, local_r,
                                                                    local_i);
                        }
                    } else {
                        Kokkos::Array<int, Dim> shift{};
                        bool full_inside = true;
                        for (unsigned d = 0; d < Dim; ++d) {
                            shift[d] = shifts(static_cast<size_t>(bi) * Dim + d);
                            if (shift[d] < 0 || shift[d] > hs[d] - W)
                                full_inside = false;
                        }

                        if (full_inside) {
                            scatter_particle_fast_nd<needs_imag>(team, shift, stride, kw, vr, vi,
                                                                 local_r, local_i);
                        } else {
                            scatter_particle_checked_nd<needs_imag>(team, shift, hs, stride, kw, vr,
                                                                    vi, local_r, local_i);
                        }
                    }
                    team.team_barrier();
                }
            }

            for (size_t idx = static_cast<size_t>(team.team_rank()); idx < htot;
                 idx += static_cast<size_t>(team.team_size())) {
                const RealType rr = local_r(idx);

                if constexpr (needs_imag) {
                    const RealType ii = local_i(idx);
                    if (rr == RealType(0) && ii == RealType(0))
                        continue;
                } else {
                    if (rr == RealType(0))
                        continue;
                }

                size_t tmp = idx;
                Kokkos::Array<int, Dim> hc{};
                for (unsigned d = 0; d < Dim; ++d) {
                    hc[d] = static_cast<int>(tmp % static_cast<size_t>(hs_vec[d]));
                    tmp /= static_cast<size_t>(hs_vec[d]);
                }

                Kokkos::Array<int, Dim> gc{};
                for (unsigned d = 0; d < Dim; ++d) {
                    const int local = tile_base[d] + hc[d] - half_left;
                    if (local < -args.nghost || local >= args.n_grid_local[d] + args.nghost)
                        goto flush_next;
                    gc[d] = local + args.nghost;
                }

                [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    if constexpr (grid_complex) {
                        RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                        Kokkos::atomic_add(&ptr[0], rr);
                        if constexpr (needs_imag)
                            Kokkos::atomic_add(&ptr[1], local_i(idx));
                    } else {
                        Kokkos::atomic_add(&args.grid(gc[Is]...), static_cast<grid_value_t>(rr));
                    }
                }(std::make_index_sequence<Dim>{});

flush_next:;
            }
        }

        template <bool NeedsImag>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /*team_size*/,
                                           int z_batches) {
            const int batch_np = z_batches > 0 ? z_batches : 1;

            size_t htot = 1;
            for (unsigned d = 0; d < Dim; ++d)
                htot *= static_cast<size_t>(tile_size[d] + padded_extra);

            size_t s = 0;
            s += scratch_real_view::shmem_size(htot);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(htot);

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

            hist_total_ = 1;
            for (unsigned d = 0; d < Dim; ++d)
                hist_total_ *= static_cast<size_t>(args.tile_size[d] + padded_extra);

            sub_teams_per_tile_ =
                std::max<size_t>(1, static_cast<size_t>(args.oversubscription_factor));

            const size_t scratch =
                compute_scratch_size<needs_imag>(args.tile_size, args.team_size, args.batch_np);

            auto policy = team_policy(n_tiles * sub_teams_per_tile_, args.team_size, 1)
                              .set_scratch_size(0, Kokkos::PerTeam(scratch));

            Kokkos::parallel_for("GridParallelScatterOutputDrivenBatched", policy, *this);
        }
    };

}  // namespace ippl::Interpolation::detail

#endif