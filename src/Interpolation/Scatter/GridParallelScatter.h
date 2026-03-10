// #ifndef IPPL_GRID_PARALLEL_SCATTER_H
// #define IPPL_GRID_PARALLEL_SCATTER_H
//
// #include <Kokkos_Core.hpp>
//
// #include "Interpolation/CoordinateTransform.h"
// #include "Interpolation/Scatter/ScatterArgumentsBase.h"
//
// // ============================================================================
// //  GridParallelScatter  –  vectorized shared-memory histogramming
// // ============================================================================
// //
// // Algorithm overview
// // ------------------
// // The team is composed of `team_size` Kokkos "threads" (warps), each of which
// // maps to one GPU warp of `vector_length` (= 32) CUDA threads.
// //
// //   • Every vector owns a private slice of the shared-memory histogram:
// //       hist_r[ vec_id * htot + local_entry ]
// //     Only one vector writes to its own slice during the particle loop,
// //     so no atomics are needed at this stage.
// //
// //   • Particles in the bin are distributed across vectors in round-robin
// //     fashion. For each particle, the vector's 32 lanes cooperate via
// //     ThreadVectorRange to:
// //       1. Compute Dim*W kernel weights in parallel.
// //       2. Scatter W^Dim stencil contributions into the private histogram.
// //
// //   • After a team barrier, the reduction phase merges all per-vector copies:
// //     a nested (TeamThreadRange × ThreadVectorRange) loop sums each histogram
// //     entry across all vectors and issues a single atomic add to global memory.
// //
// // ============================================================================
//
// namespace ippl::Interpolation::detail {
//
//     template <int W, class Types, class Policy>
//     struct GridParallelScatter {
//         static_assert(Policy::use_sorting,
//                       "GridParallelScatter assumes sorted/bin-partitioned particles");
//
//         static constexpr bool     requires_binning = true;
//         static constexpr unsigned Dim              = Types::Dim;
//         static constexpr int      half_left        = (W + 1) / 2;
//
//         static constexpr int vector_length = 32;
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
//         // ── Scratch layout (per team) ────────────────────────────────────────
//         //
//         //  [A]  nv * htot   hist_r  (real part)
//         //  [B]  nv * htot   hist_i  (imag, complex only)
//         //  [C]  nv * Dim*W  kw      (kernel weights per vector)
//         //  [D]  nv * Dim    base_s  (stencil bases per vector)
//         //
//         //  When z_batches > 1, the z-dimension of the histogram is reduced from
//         //  (tile_z + W + 1) to (tile_z + z_batch_size + 1).
//         // ────────────────────────────────────────────────────────────────────
//         template <bool IsComplex>
//         static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int team_size,
//                                            int z_batches = 1) {
//             const int z_batch_size = (W + z_batches - 1) / z_batches;
//
//             size_t htot = 1;
//             for (unsigned d = 0; d < Dim; ++d) {
//                 const int hist_dim = (Dim == 3 && d == 2 && z_batches > 1)
//                                          ? tile_size[d] + z_batch_size + 1
//                                          : tile_size[d] + W + 1;
//                 htot *= static_cast<size_t>(hist_dim);
//             }
//
//             const int nv = std::max(1, team_size);
//
//             return (IsComplex ? 2 : 1) * scratch_real_view::shmem_size(nv * htot)
//                    + scratch_real_view::shmem_size(nv * static_cast<int>(Dim) * W)
//                    + scratch_int_view::shmem_size(nv * static_cast<int>(Dim));
//         }
//
//         // ── Arguments ───────────────────────────────────────────────────────
//         struct Arguments : ScatterArgumentsBase<Arguments, Types> {
//             Kokkos::View<size_t*, memory_space> permute;
//             Kokkos::View<size_t*, memory_space> bin_offsets;
//             Vector<int, Dim> num_tiles;
//             Vector<int, Dim> tile_size;
//             int team_size;
//             int oversubscription_factor;
//             int z_batches;
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
//                 a.z_batches               = config.z_batches;
//                 return a;
//             }
//         };
//
//         Arguments args;
//
//         size_t sub_teams_per_tile_ = 1;
//         size_t hist_total_         = 1;
//
//         // Z-batching state
//         int z_batch_size_ = W;
//         int z_start_      = 0;
//         int z_end_        = W;
//
//         // ── Geometry helpers ─────────────────────────────────────────────────
//         KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
//             Vector<int, Dim> hs;
//             for (unsigned d = 0; d < Dim; ++d)
//                 hs[d] = args.tile_size[d] + ((Dim == 3 && d == 2) ? z_batch_size_ : W) + 1;
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
//         // ── Kernel operator ──────────────────────────────────────────────────
//         KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
//             using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
//             constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
//             constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
//
//             const int vec_id = team.team_rank();
//             const int nv     = team.team_size();
//
//             const size_t league_r  = static_cast<size_t>(team.league_rank());
//             const size_t tile_id   = league_r / sub_teams_per_tile_;
//             const size_t sub_id    = league_r % sub_teams_per_tile_;
//             const size_t bin_start = args.bin_offsets(tile_id);
//             const size_t bin_end   = args.bin_offsets(tile_id + 1);
//             const size_t bin_size  = bin_end - bin_start;
//
//             const size_t particles_per_sub =
//                 (bin_size + sub_teams_per_tile_ - 1) / sub_teams_per_tile_;
//             const size_t pstart = bin_start + sub_id * particles_per_sub;
//             if (pstart >= bin_end)
//                 return;
//             const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);
//
//             const auto   tile_base = decode_tile_base(tile_id);
//             const auto   hs        = hist_size();
//             const size_t htot      = hist_total_;
//
//             // ── Scratch allocation ───────────────────────────────────────────
//             scratch_real_view hist_r(team.team_scratch(0), nv * htot);
//             scratch_real_view hist_i;
//             if constexpr (gcplx)
//                 hist_i = scratch_real_view(team.team_scratch(0), nv * htot);
//
//             scratch_real_view kw(team.team_scratch(0), nv * static_cast<int>(Dim) * W);
//             scratch_int_view  base_s(team.team_scratch(0), nv * static_cast<int>(Dim));
//
//             RealType* my_kw   = kw.data()    + vec_id * (static_cast<int>(Dim) * W);
//             int*      my_base = base_s.data() + vec_id * static_cast<int>(Dim);
//
//             // ── Zero this warp's histogram slice ─────────────────────────────
//             Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, htot), [&](size_t i) {
//                 hist_r(vec_id * htot + i) = RealType(0);
//                 if constexpr (gcplx)
//                     hist_i(vec_id * htot + i) = RealType(0);
//             });
//             team.team_barrier();
//
//             // ── Particle loop ────────────────────────────────────────────────
//             const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
//                                                                args.n_grid};
//
//             for (size_t ip = pstart + static_cast<size_t>(vec_id); ip < pend;
//                  ip += static_cast<size_t>(nv)) {
//
//                 const size_t p = args.permute(ip);
//
//                 RealType val_r = RealType(0);
//                 RealType val_i = RealType(0);
//                 if constexpr (vcplx) {
//                     val_r = args.values(p).real();
//                     if constexpr (gcplx)
//                         val_i = args.values(p).imag();
//                 } else {
//                     val_r = static_cast<RealType>(args.values(p));
//                 }
//
//                 // ── Step 1: kernel weights ────────────────────────────────────
//                 Kokkos::parallel_for(
//                     Kokkos::ThreadVectorRange(team, static_cast<int>(Dim) * W),
//                     [&](int flat) {
//                         const int      d    = flat / W;
//                         const int      i    = flat % W;
//                         const RealType gp   = transform.toGridCoordinate(args.x(p)[d], d);
//                         const int      idx0 = transform.getStencilBase(gp - RealType(0.5), W);
//                         my_kw[d * W + i] =
//                             args.kernel((gp - (RealType(idx0 + i) + RealType(0.5))) *
//                             args.inv_hw);
//                         if (i == 0)
//                             my_base[d] = idx0 - args.local_offset[d];
//                     });
//
//                 // ── Step 2: scatter to private histogram ──────────────────────
//                 RealType* h_r = hist_r.data() + vec_id * htot;
//                 RealType* h_i = gcplx ? hist_i.data() + vec_id * htot : nullptr;
//
//                 if constexpr (Dim == 1) {
//                     const int bh0 = my_base[0] + half_left - tile_base[0];
//
//                     Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W), [&](int i0) {
//                         const size_t   hidx = static_cast<size_t>(bh0 + i0);
//                         const RealType w    = my_kw[i0];
//                         h_r[hidx] += val_r * w;
//                         if constexpr (gcplx)
//                             h_i[hidx] += val_i * w;
//                     });
//
//                 } else if constexpr (Dim == 2) {
//                     const int bh0 = my_base[0] + half_left - tile_base[0];
//                     const int bh1 = my_base[1] + half_left - tile_base[1];
//
//                     Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W * W),
//                                          [&](int flat) {
//                         const int      i0   = flat % W;
//                         const int      i1   = flat / W;
//                         const RealType w    = my_kw[i0] * my_kw[W + i1];
//                         const size_t   hidx = static_cast<size_t>(bh0 + i0)
//                                             + static_cast<size_t>(hs[0])
//                                                   * static_cast<size_t>(bh1 + i1);
//                         h_r[hidx] += val_r * w;
//                         if constexpr (gcplx)
//                             h_i[hidx] += val_i * w;
//                     });
//
//                 } else if constexpr (Dim == 3) {
//                     const int bh0      = my_base[0] + half_left - tile_base[0];
//                     const int bh1      = my_base[1] + half_left - tile_base[1];
//                     const int bh2      = my_base[2] + half_left - tile_base[2];
//                     const int z_range  = z_end_ - z_start_;
//
//                     Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W * W * z_range),
//                                          [&](int flat) {
//                         const int      i0   = flat % W;
//                         const int      i1   = (flat / W) % W;
//                         const int      i2   = z_start_ + flat / (W * W);
//                         const RealType w    = my_kw[i0] * my_kw[W + i1] * my_kw[2 * W + i2];
//                         const size_t   hidx =
//                             static_cast<size_t>(bh0 + i0)
//                             + static_cast<size_t>(hs[0])
//                                   * (static_cast<size_t>(bh1 + i1)
//                                      + static_cast<size_t>(hs[1])
//                                            * static_cast<size_t>(bh2 + i2 - z_start_));
//                         h_r[hidx] += val_r * w;
//                         if constexpr (gcplx)
//                             h_i[hidx] += val_i * w;
//                     });
//                 }
//             }
//
//             team.team_barrier();
//
//             // ── Reduction: merge nv histogram copies → global grid ────────────
//             const size_t chunks = (htot + static_cast<size_t>(vector_length) - 1)
//                                   / static_cast<size_t>(vector_length);
//
//             Kokkos::parallel_for(Kokkos::TeamThreadRange(team, chunks), [&](size_t chunk) {
//                 Kokkos::parallel_for(
//                     Kokkos::ThreadVectorRange(team, vector_length), [&](int lane) {
//                         const size_t idx = chunk * static_cast<size_t>(vector_length)
//                                          + static_cast<size_t>(lane);
//                         if (idx >= htot)
//                             return;
//
//                         RealType sum_r = RealType(0);
//                         RealType sum_i = RealType(0);
//                         for (int v = 0; v < nv; ++v) {
//                             sum_r += hist_r(v * htot + idx);
//                             if constexpr (gcplx)
//                                 sum_i += hist_i(v * htot + idx);
//                         }
//
//                         // Decode flat histogram index → per-dim histogram coordinates
//                         size_t tmp = idx;
//                         Kokkos::Array<int, Dim> hc{};
//                         for (unsigned d = 0; d < Dim; ++d) {
//                             hc[d] = static_cast<int>(tmp % static_cast<size_t>(hs[d]));
//                             tmp /= static_cast<size_t>(hs[d]);
//                         }
//
//                         // Map histogram coords → local grid coords; skip ghost overflow
//                         Kokkos::Array<int, Dim> gc{};
//                         for (unsigned d = 0; d < Dim; ++d) {
//                             int hc_adjusted = hc[d];
//                             if constexpr (Dim == 3) {
//                                 if (d == 2)
//                                     hc_adjusted += z_start_;
//                             }
//                             const int local = tile_base[d] + hc_adjusted - half_left;
//                             if (local < -args.nghost
//                                 || local >= args.n_grid_local[d] + args.nghost)
//                                 return;
//                             gc[d] = local + args.nghost;
//                         }
//
//                         [&]<std::size_t... Is>(std::index_sequence<Is...>) {
//                             if constexpr (gcplx) {
//                                 RealType* ptr =
//                                     reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
//                                 Kokkos::atomic_add(&ptr[0], sum_r);
//                                 Kokkos::atomic_add(&ptr[1], sum_i);
//                             } else {
//                                 Kokkos::atomic_add(&args.grid(gc[Is]...),
//                                                    static_cast<grid_value_t>(sum_r));
//                             }
//                         }(std::make_index_sequence<Dim>{});
//                     });
//             });
//         }
//
//         // ── run() ────────────────────────────────────────────────────────────
//         void run(size_t n_particles) {
//             using grid_value_t  = typename decltype(args.grid)::non_const_value_type;
//             constexpr bool cplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
//
//             size_t n_tiles = 1;
//             for (unsigned d = 0; d < Dim; ++d)
//                 n_tiles *= static_cast<size_t>(args.num_tiles[d]);
//
//             if (n_tiles == 0 || n_particles == 0)
//                 return;
//
//             sub_teams_per_tile_ = std::max(size_t(1), size_t(args.oversubscription_factor));
//
//             const int z_batches = (Dim == 3) ? std::max(1, args.z_batches) : 1;
//             z_batch_size_       = (W + z_batches - 1) / z_batches;
//
//             // Compute histogram total size
//             hist_total_ = 1;
//             for (unsigned d = 0; d < Dim; ++d) {
//                 const int hist_dim = args.tile_size[d]
//                                      + ((Dim == 3 && d == 2) ? z_batch_size_ : W) + 1;
//                 hist_total_ *= static_cast<size_t>(hist_dim);
//             }
//
//             const size_t scratch =
//                 compute_scratch_size<cplx>(args.tile_size, args.team_size, z_batches);
//
//             for (int batch = 0; batch < z_batches; ++batch) {
//                 z_start_ = batch * z_batch_size_;
//                 z_end_   = std::min((batch + 1) * z_batch_size_, W);
//
//                 Kokkos::parallel_for(
//                     "GridParallelScatterVectorized",
//                     team_policy(n_tiles * sub_teams_per_tile_, args.team_size, vector_length)
//                         .set_scratch_size(0, Kokkos::PerTeam(scratch)),
//                     *this);
//             }
//         }
//     };
//
// }  // namespace ippl::Interpolation::detail
//
// #endif  // IPPL_GRID_PARALLEL_SCATTER_H

// namespace ippl::Interpolation::detail {
//
//     template <int W, class Types, class Policy>
//     struct GridParallelScatter {
//         static_assert(Policy::use_sorting,
//                       "GridParallelScatter assumes sorted/bin-partitioned particles");
//
//         static constexpr bool requires_binning = true;
//         static constexpr unsigned Dim          = Types::Dim;
//         static constexpr int half_left         = (W + 1) / 2;
//         static constexpr int vector_length     = 32;
//
//         // Particles loaded per batch.  Increasing this raises occupancy up to the
//         // point where scratch spills; 16 is a safe starting value to tune from.
//         // static constexpr int batch_np = 16;
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
//         // ── Scratch layout (per team) ─────────────────────────────────────────
//         //
//         //  [A]  htot                local_r    real part of local subgrid
//         //  [B]  htot                local_i    imag part (complex grids only)
//         //  [C]  batch_np * Dim * W  kerevals   kernel weights [bi][d][wi]
//         //  [D]  batch_np            vals_r     particle real values
//         //  [E]  batch_np            vals_i     particle imag values (complex only)
//         //  [F]  batch_np * Dim      shifts     stencil base relative to tile [bi][d]
//         //
//         //  Total ≈ htot + batch_np*(Dim*W + 1 + Dim) floats + Dim*batch_np ints.
//         //  Previous layout needed nv*htot floats for histograms; this is ~25× smaller
//         //  for typical nv=32, eliminating the reduction pass entirely.
//         // ─────────────────────────────────────────────────────────────────────
//         template <bool IsComplex>
//         static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /*team_size*/,
//                                            int batches) {
//             size_t htot = 1;
//             for (unsigned d = 0; d < Dim; ++d)
//                 htot *= static_cast<size_t>(tile_size[d] + W + 1);
//
//             return (IsComplex ? 2 : 1) * scratch_real_view::shmem_size(htot)
//                    + scratch_real_view::shmem_size(batches * static_cast<int>(Dim) * W)
//                    + (IsComplex ? scratch_real_view::shmem_size(batches) : 0)
//                    + scratch_real_view::shmem_size(batches)
//                    + scratch_int_view::shmem_size(batches * static_cast<int>(Dim));
//         }
//
//         // ── Arguments ─────────────────────────────────────────────────────────
//         struct Arguments : ScatterArgumentsBase<Arguments, Types> {
//             Kokkos::View<size_t*, memory_space> permute;
//             Kokkos::View<size_t*, memory_space> bin_offsets;
//             Vector<int, Dim> num_tiles;
//             Vector<int, Dim> tile_size;
//             int team_size;
//             int oversubscription_factor;
//             int batches;
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
//                 a.batches                 = config.z_batches;
//                 return a;
//             }
//         };
//
//         Arguments args;
//         size_t hist_total_         = 1;
//         size_t sub_teams_per_tile_ = 1;
//
//         // ── Geometry helpers ───────────────────────────────────────────────────
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
//         // ── Kernel operator ────────────────────────────────────────────────────
//         KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
//             using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
//             constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
//             constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
//
//             const size_t league_r  = static_cast<size_t>(team.league_rank());
//             const size_t tile_id   = league_r / sub_teams_per_tile_;
//             const size_t sub_id    = league_r % sub_teams_per_tile_;
//             const size_t bin_start = args.bin_offsets(tile_id);
//             const size_t bin_end   = args.bin_offsets(tile_id + 1);
//             const int batches = args.batches;
//             const size_t bin_size  = bin_end - bin_start;
//
//             const size_t particles_per_sub =
//                 (bin_size + sub_teams_per_tile_ - 1) / sub_teams_per_tile_;
//             const size_t pstart = bin_start + sub_id * particles_per_sub;
//             if (pstart >= bin_end)
//                 return;
//             const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);
//
//             const auto tile_base = decode_tile_base(tile_id);
//             const auto hs        = hist_size();
//             const size_t htot    = hist_total_;
//
//             // ── Scratch allocation ─────────────────────────────────────────────
//             scratch_real_view local_r(team.team_scratch(0), htot);
//             scratch_real_view local_i;
//             if constexpr (gcplx)
//                 local_i = scratch_real_view(team.team_scratch(0), htot);
//
//             scratch_real_view kerevals(team.team_scratch(0), batches * static_cast<int>(Dim) *
//             W); scratch_real_view vals_r(team.team_scratch(0), batches); scratch_real_view
//             vals_i; if constexpr (vcplx && gcplx)
//                 vals_i = scratch_real_view(team.team_scratch(0), batches);
//             scratch_int_view shifts(team.team_scratch(0), batches * static_cast<int>(Dim));
//
//             // ── Zero the local subgrid ─────────────────────────────────────────
//             Kokkos::parallel_for(Kokkos::TeamVectorRange(team, htot), [&](size_t i) {
//                 local_r(i) = RealType(0);
//                 if constexpr (gcplx)
//                     local_i(i) = RealType(0);
//             });
//             team.team_barrier();
//
//             const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
//                                                                args.n_grid};
//
//             // ── Batch loop ─────────────────────────────────────────────────────
//             for (size_t batch_begin = pstart; batch_begin < pend; batch_begin += batches) {
//                 const int batch_size = static_cast<int>(
//                     Kokkos::min(pend - batch_begin, static_cast<size_t>(batches)));
//
//                 // ── Load phase ─────────────────────────────────────────────────
//                 //
//                 //  Flat index covers (bi, d, wi) in row-major order so each lane
//                 //  touches a unique kerevals slot.  Multiple lanes from the same
//                 //  (bi, d) write the same value to shifts — an idempotent race
//                 //  that is benign on all current Kokkos backends.
//                 Kokkos::parallel_for(
//                     Kokkos::TeamVectorRange(team, batch_size * static_cast<int>(Dim) * W),
//                     [&](int flat) {
//                         const int wi = flat % W;
//                         const int d  = (flat / W) % static_cast<int>(Dim);
//                         const int bi = flat / (W * static_cast<int>(Dim));
//
//                         const size_t p    = args.permute(batch_begin + static_cast<size_t>(bi));
//                         const RealType gp = transform.toGridCoordinate(args.x(p)[d], d);
//                         const int idx0    = transform.getStencilBase(gp - RealType(0.5), W);
//
//                         kerevals(bi * static_cast<int>(Dim) * W + d * W + wi) =
//                             args.kernel((gp - (RealType(idx0 + wi) + RealType(0.5))) *
//                             args.inv_hw);
//
//                         shifts(bi * static_cast<int>(Dim) + d) =
//                             idx0 - args.local_offset[d] + half_left - tile_base[d];
//                     });
//
//                 Kokkos::parallel_for(Kokkos::TeamVectorRange(team, batch_size), [&](int bi) {
//                     const size_t p = args.permute(batch_begin + static_cast<size_t>(bi));
//                     if constexpr (vcplx) {
//                         vals_r(bi) = args.values(p).real();
//                         if constexpr (gcplx)
//                             vals_i(bi) = args.values(p).imag();
//                     } else {
//                         vals_r(bi) = static_cast<RealType>(args.values(p));
//                     }
//                 });
//
//                 team.team_barrier();  // all scratch writes visible before gather
//
//                 // ── Gather phase (output-driven) ───────────────────────────────
//                 //
//                 //  Each lane is assigned a contiguous slice of subgrid cells.
//                 //  The flat→(i0,i1,…) decode is performed ONCE per cell, outside
//                 //  the particle loop, eliminating repeated multiply/modulo chains.
//                 //
//                 //  Hot-loop body per (cell, particle):
//                 //    Dim subtracts          wi = ic[d] - sh[d]
//                 //    Dim unsigned compares  (wi<0 || wi≥W) fused into one op each
//                 //    Dim FP loads + (Dim-1) FP multiplies  (separable weight)
//                 //    1-2 FMAs               (accumulate r, i)
//                 //
//                 Kokkos::parallel_for(Kokkos::TeamVectorRange(team, htot), [&](size_t idx) {
//                     // Decode flat subgrid index → per-dim coordinates.
//                     // This is hoisted outside the particle loop — the key saving
//                     // vs the input-driven version where hidx was recomputed inside.
//                     Kokkos::Array<int, Dim> ic{};
//                     {
//                         size_t tmp = idx;
//                         for (unsigned d = 0; d < Dim; ++d) {
//                             ic[d] = static_cast<int>(tmp % static_cast<size_t>(hs[d]));
//                             tmp /= static_cast<size_t>(hs[d]);
//                         }
//                     }
//
//                     RealType acc_r = RealType(0);
//                     RealType acc_i = RealType(0);
//
//                     for (int bi = 0; bi < batch_size; ++bi) {
//                         const int* sh      = shifts.data() + bi * static_cast<int>(Dim);
//                         const RealType* kw = kerevals.data() + bi * static_cast<int>(Dim) * W;
//
//                         // Casting to unsigned fuses the (wi < 0 || wi >= W) test
//                         // into a single comparison per dimension.
//                         RealType w = RealType(1);
//                         bool ok    = true;
//                         for (unsigned d = 0; d < Dim; ++d) {
//                             const auto wi = static_cast<unsigned>(ic[d] - sh[d]);
//                             if (wi >= static_cast<unsigned>(W)) {
//                                 ok = false;
//                                 break;
//                             }
//                             w *= kw[d * W + static_cast<int>(wi)];
//                         }
//                         if (!ok)
//                             continue;
//
//                         acc_r += vals_r(bi) * w;
//                         if constexpr (gcplx)
//                             acc_i += vals_i(bi) * w;
//                     }
//
//                     // Each lane writes to a unique idx — no atomics needed here.
//                     local_r(idx) += acc_r;
//                     if constexpr (gcplx)
//                         local_i(idx) += acc_i;
//                 });
//
//                 team.team_barrier();  // local_r writes must land before next load
//             }
//
//             // ── Write local subgrid → global grid ─────────────────────────────
//             //
//             //  One atomic add per live cell, issued once per tile regardless of
//             //  particle count.  The previous version issued one add per particle
//             //  per stencil point after the intra-team reduction.
//             Kokkos::parallel_for(Kokkos::TeamVectorRange(team, htot), [&](size_t idx) {
//                 size_t tmp = idx;
//                 Kokkos::Array<int, Dim> hc{};
//                 for (unsigned d = 0; d < Dim; ++d) {
//                     hc[d] = static_cast<int>(tmp % static_cast<size_t>(hs[d]));
//                     tmp /= static_cast<size_t>(hs[d]);
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
//                     if constexpr (gcplx) {
//                         RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
//                         Kokkos::atomic_add(&ptr[0], local_r(idx));
//                         Kokkos::atomic_add(&ptr[1], local_i(idx));
//                     } else {
//                         Kokkos::atomic_add(&args.grid(gc[Is]...),
//                                            static_cast<grid_value_t>(local_r(idx)));
//                     }
//                 }(std::make_index_sequence<Dim>{});
//             });
//         }
//
//         // ── run() ──────────────────────────────────────────────────────────────
//         void run(size_t n_particles) {
//             using grid_value_t  = typename decltype(args.grid)::non_const_value_type;
//             constexpr bool cplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
//
//             size_t n_tiles = 1;
//             for (unsigned d = 0; d < Dim; ++d)
//                 n_tiles *= static_cast<size_t>(args.num_tiles[d]);
//
//             if (n_tiles == 0 || n_particles == 0)
//                 return;
//
//             sub_teams_per_tile_ = std::max(size_t(1), size_t(args.oversubscription_factor));
//
//             hist_total_ = 1;
//             for (unsigned d = 0; d < Dim; ++d)
//                 hist_total_ *= static_cast<size_t>(args.tile_size[d] + W + 1);
//
//             const size_t scratch = compute_scratch_size<cplx>(args.tile_size, args.team_size, 1);
//
//             Kokkos::parallel_for(
//                 "GridParallelScatterOutputDriven",
//                 team_policy(n_tiles * sub_teams_per_tile_, args.team_size, vector_length)
//                     .set_scratch_size(0, Kokkos::PerTeam(scratch)),
//                 *this);
//         }
//     };
//
// }  // namespace ippl::Interpolation::detail

#ifndef IPPL_GRID_PARALLEL_SCATTER_H
#define IPPL_GRID_PARALLEL_SCATTER_H

namespace ippl::Interpolation::detail {

// ── Forward declaration (needed so the __global__ kernel can reference the struct) ──
template <int W, class Types, class Policy>
struct GridParallelScatter;

// ── Native GPU kernel (free function; __global__ cannot be a member) ─────────────
//
//  Parses extern __shared__ into the same scratch layout used by the Kokkos path,
//  then delegates to GridParallelScatter::run_tile with threadIdx.x / blockIdx.x.
//  One block per tile; blockDim.x is args.team_size.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <int W, class Types, class Policy>
__global__ void grid_parallel_scatter_native(GridParallelScatter<W, Types, Policy> self) {
    using Scatter    = GridParallelScatter<W, Types, Policy>;
    using RealType   = typename Scatter::RealType;
    using ValueType  = typename Scatter::ValueType;
    using grid_val_t = typename decltype(self.args.grid)::non_const_value_type;
    constexpr bool gcplx = std::is_same_v<grid_val_t, Kokkos::complex<RealType>>;
    constexpr bool vcplx = std::is_same_v<ValueType,  Kokkos::complex<RealType>>;
    constexpr int  Dim   = static_cast<int>(Scatter::Dim);

    extern __shared__ char sbuf[];

    // ── Carve shared memory ───────────────────────────────────────────────────
    //  All RealType arrays are laid out first (naturally aligned); the int array
    //  is placed last with an explicit alignment step.
    const size_t htot = self.hist_total_;
    const int    np   = self.args.batch_np;

    size_t off = 0;
    auto alloc_r = [&](size_t n) -> RealType* {
        RealType* p  = reinterpret_cast<RealType*>(sbuf + off);
        off         += n * sizeof(RealType);
        return p;
    };
    auto alloc_i = [&](size_t n) -> int* {
        off         = (off + sizeof(int) - 1) & ~(sizeof(int) - 1);
        int* p       = reinterpret_cast<int*>(sbuf + off);
        off         += n * sizeof(int);
        return p;
    };

    RealType* local_r  = alloc_r(htot);
    RealType* local_i  = nullptr;
    if constexpr (gcplx) local_i = alloc_r(htot);
    RealType* kerevals = alloc_r(static_cast<size_t>(np * Dim * W));
    RealType* vals_r   = alloc_r(static_cast<size_t>(np));
    RealType* vals_i   = nullptr;
    if constexpr (gcplx && vcplx) vals_i = alloc_r(static_cast<size_t>(np));
    int*      shifts   = alloc_i(static_cast<size_t>(np * Dim));

    self.run_tile(
        static_cast<int>(threadIdx.x),
        static_cast<int>(blockDim.x),
        static_cast<size_t>(blockIdx.x),
        local_r, local_i, kerevals, vals_r, vals_i, shifts,
        typename Scatter::NativeBarrier{});
}
#endif // KOKKOS_ENABLE_CUDA || KOKKOS_ENABLE_HIP


// ─────────────────────────────────────────────────────────────────────────────
template <int W, class Types, class Policy>
struct GridParallelScatter {
    static_assert(Policy::use_sorting,
                  "GridParallelScatter assumes sorted/bin-partitioned particles");

    static constexpr bool     requires_binning = true;
    static constexpr unsigned Dim              = Types::Dim;
    static constexpr int      half_left        = (W + 1) / 2;

    using RealType        = typename Types::RealType;
    using ValueType       = typename Types::ValueType;
    using memory_space    = typename Types::memory_space;
    using execution_space = typename Types::execution_space;

    // True when we can bypass Kokkos launch and use <<<>>> directly.
    static constexpr bool use_native_gpu =
        false
#if defined(KOKKOS_ENABLE_CUDA)
        || std::is_same_v<execution_space, Kokkos::Cuda>
#endif
#if defined(KOKKOS_ENABLE_HIP)
        || std::is_same_v<execution_space, Kokkos::HIP>
#endif
        ;

    using team_policy   = Kokkos::TeamPolicy<execution_space>;
    using team_member   = typename team_policy::member_type;
    using scratch_space = typename execution_space::scratch_memory_space;

    using scratch_real_view =
        Kokkos::View<RealType*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using scratch_int_view =
        Kokkos::View<int*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // ── Scratch layout (one copy per team / block) ────────────────────────────
    //
    //  [A]  htot          local_r    real part of local subgrid
    //  [B]  htot          local_i    imag part (complex grids only)
    //  [C]  np * Dim * W  kerevals   kernel weights per batch particle [bi][d][wi]
    //  [D]  np            vals_r     particle real values
    //  [E]  np            vals_i     particle imag values (complex only)
    //  [F]  np * Dim      shifts     stencil bases relative to tile [bi][d]
    //
    //  np = args.batch_np (from config.z_batches).
    //  No shared-memory atomics: particles are scattered one at a time so all
    //  threads write disjoint local_r cells within each particle pass.
    // ─────────────────────────────────────────────────────────────────────────
    // ── Public scratch-size query (called by the dispatch layer) ─────────────
    //
    //  IsComplex refers to the grid value type (real vs complex<RealType>).
    //  np is derived from config.z_batches (the batch-particle count).
    //  ValueType complexity is not visible here, so we conservatively allocate
    //  vals_i when IsComplex is true (same condition used in operator()).
    template <bool IsComplex>
    static size_t compute_scratch_size(const Vector<int, Dim>& tile_size,
                                       int /*team_size*/, int z_batches) {
        const int np = std::max(1, z_batches);
        return kokkos_scratch_size<IsComplex, IsComplex>(tile_size, np);
    }

    template <bool GCplx, bool VCplx>
    static size_t kokkos_scratch_size(const Vector<int, Dim>& tile_size, int np) {
        size_t htot = 1;
        for (unsigned d = 0; d < Dim; ++d)
            htot *= static_cast<size_t>(tile_size[d] + W + 1);

        return (GCplx ? 2 : 1) * scratch_real_view::shmem_size(htot)
             + scratch_real_view::shmem_size(static_cast<size_t>(np * static_cast<int>(Dim) * W))
             + scratch_real_view::shmem_size(static_cast<size_t>(np))
             + (GCplx && VCplx ? scratch_real_view::shmem_size(static_cast<size_t>(np)) : 0)
             + scratch_int_view::shmem_size(static_cast<size_t>(np * static_cast<int>(Dim)));
    }

    // ── Arguments ─────────────────────────────────────────────────────────────
    struct Arguments : ScatterArgumentsBase<Arguments, Types> {
        Kokkos::View<size_t*, memory_space> permute;
        Kokkos::View<size_t*, memory_space> bin_offsets;
        Vector<int, Dim> num_tiles;
        Vector<int, Dim> tile_size;
        int team_size;
        int batch_np;  // particle batch size; taken from config.z_batches

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
            a.batch_np    = std::max(1, config.z_batches);
            return a;
        }
    };

    Arguments args;
    size_t    hist_total_ = 1;

    // ── Geometry ──────────────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
        Vector<int, Dim> hs;
        for (unsigned d = 0; d < Dim; ++d)
            hs[d] = args.tile_size[d] + W + 1;
        return hs;
    }

    KOKKOS_INLINE_FUNCTION Vector<int, Dim> decode_tile_base(size_t tile_id) const {
        Vector<int, Dim> tb;
        for (size_t t = tile_id, d = Dim; d-- > 0;) {
            tb[d] = static_cast<int>(t % static_cast<size_t>(args.num_tiles[d]))
                    * args.tile_size[d];
            t    /= static_cast<size_t>(args.num_tiles[d]);
        }
        return tb;
    }

    // ── Sync functors ─────────────────────────────────────────────────────────

    // Used by the native GPU kernel path
    struct NativeBarrier {
        KOKKOS_FORCEINLINE_FUNCTION void operator()() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            __syncthreads();
#endif
        }
    };

    // Used by the Kokkos team operator path
    struct KokkosBarrier {
        const team_member& team_;
        KOKKOS_FORCEINLINE_FUNCTION void operator()() const { team_.team_barrier(); }
    };

    // ── Tile body ─────────────────────────────────────────────────────────────
    //
    //  Shared between the Kokkos operator (KokkosBarrier, team.team_rank() as tid)
    //  and the native __global__ kernel (NativeBarrier, threadIdx.x as tid).
    //
    //  Algorithm matches spread_3d_output_driven:
    //   1. Zero local subgrid (parallel, strided by block_size).
    //   2. For each particle batch of size np:
    //      a. Load kerevals, vals, shifts → shared (parallel over bi).
    //      b. For each particle bi in batch (sequential outer loop):
    //           All threads collectively scatter the W^Dim stencil into local_r.
    //           sync() after each particle — no shared-memory atomics since all
    //           threads work on the same particle and each owns unique cells.
    //   3. Flush local subgrid → global grid (Kokkos::atomic_add, parallel).
    //
    //  local_i / vals_i_sm are nullptr when the grid / particles are real.
    // ─────────────────────────────────────────────────────────────────────────
    template <class SyncFn>
    KOKKOS_FORCEINLINE_FUNCTION
    void run_tile(int tid, int block_size, size_t tile_id,
                  RealType* __restrict__ local_r,
                  RealType* __restrict__ local_i,
                  RealType* __restrict__ kerevals_sm,  // [bi * Dim * W + d * W + wi]
                  RealType* __restrict__ vals_r_sm,
                  RealType* __restrict__ vals_i_sm,
                  int*      __restrict__ shifts_sm,    // [bi * Dim + d]
                  SyncFn                 sync) const {

        using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
        constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
        constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

        const size_t htot      = hist_total_;
        const auto   hs        = hist_size();
        const auto   tb        = decode_tile_base(tile_id);
        const size_t bin_start = args.bin_offsets(tile_id);
        const size_t bin_end   = args.bin_offsets(tile_id + 1);
        const int    nupts     = static_cast<int>(bin_end - bin_start);
        const int    np        = args.batch_np;

        const CoordinateTransform<RealType, Dim> transform{
            args.origin, args.invdx, args.n_grid};

        // ── 1. Zero local subgrid ─────────────────────────────────────────────
        for (int i = tid; i < static_cast<int>(htot); i += block_size) {
            local_r[i] = RealType(0);
            if constexpr (gcplx) local_i[i] = RealType(0);
        }
        sync();

        // ── 2. Batch loop ─────────────────────────────────────────────────────
        for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
            const int batch_size = Kokkos::min(np, nupts - batch_begin);

            // ── 2a. Load ──────────────────────────────────────────────────────
            for (int bi = tid; bi < batch_size; bi += block_size) {
                const size_t p =
                    args.permute(bin_start + static_cast<size_t>(batch_begin + bi));

                if constexpr (vcplx) {
                    vals_r_sm[bi] = args.values(p).real();
                    if constexpr (gcplx) vals_i_sm[bi] = args.values(p).imag();
                } else {
                    vals_r_sm[bi] = static_cast<RealType>(args.values(p));
                }

                for (unsigned d = 0; d < Dim; ++d) {
                    const RealType gp   = transform.toGridCoordinate(args.x(p)[d], d);
                    const int      idx0 = transform.getStencilBase(gp - RealType(0.5), W);
                    shifts_sm[bi * static_cast<int>(Dim) + static_cast<int>(d)] =
                        idx0 - args.local_offset[d] + half_left - tb[d];
                    for (int wi = 0; wi < W; ++wi)
                        kerevals_sm[bi * static_cast<int>(Dim) * W
                                    + static_cast<int>(d) * W + wi] =
                            args.kernel((gp - (RealType(idx0 + wi) + RealType(0.5)))
                                        * args.inv_hw);
                }
            }
            sync();

            // ── 2b. Scatter ───────────────────────────────────────────────────
            //
            //  Particle bi is processed by ALL threads (they split the W^Dim stencil
            //  cells in a strided loop over idx).  Because only one particle is live
            //  per iteration, each thread writes to a disjoint hidx → no atomic.
            //  sync() after each particle guards the next particle's writes to the
            //  same local_r cells, and also guards the next batch's load phase.
            constexpr int stencil_total = (Dim == 1) ? W : (Dim == 2) ? W * W : W * W * W;

            for (int bi = 0; bi < batch_size; ++bi) {
                const RealType* kw = kerevals_sm + bi * static_cast<int>(Dim) * W;
                const int*      sh = shifts_sm   + bi * static_cast<int>(Dim);

                const RealType vr = vals_r_sm[bi];
                RealType       vi = RealType(0);
                if constexpr (gcplx && vcplx) vi = vals_i_sm[bi];

                // Fold all particle-dependent offsets into one scalar so the
                // hot inner loop only adds compile-time-bounded stencil offsets
                // (strides are loop-invariant → compiler strength-reduces to adds).
                size_t base_hidx;
                if constexpr (Dim == 1) {
                    base_hidx = static_cast<size_t>(sh[0]);
                } else if constexpr (Dim == 2) {
                    base_hidx = static_cast<size_t>(sh[0])
                              + static_cast<size_t>(hs[0]) * static_cast<size_t>(sh[1]);
                } else {
                    base_hidx = static_cast<size_t>(sh[0])
                              + static_cast<size_t>(hs[0])
                                    * (static_cast<size_t>(sh[1])
                                       + static_cast<size_t>(hs[1])
                                             * static_cast<size_t>(sh[2]));
                }

                for (int idx = tid; idx < stencil_total; idx += block_size) {
                    const int i0 =                idx % W;
                    const int i1 = (Dim >= 2) ? (idx / W) % W : 0;
                    const int i2 = (Dim == 3) ?  idx / (W * W) : 0;

                    RealType w;
                    size_t   hidx;
                    if constexpr (Dim == 1) {
                        w    = kw[i0];
                        hidx = base_hidx + static_cast<size_t>(i0);
                    } else if constexpr (Dim == 2) {
                        w    = kw[i0] * kw[W + i1];
                        hidx = base_hidx
                             + static_cast<size_t>(i0)
                             + static_cast<size_t>(hs[0]) * static_cast<size_t>(i1);
                    } else {
                        w    = kw[i0] * kw[W + i1] * kw[2 * W + i2];
                        hidx = base_hidx
                             + static_cast<size_t>(i0)
                             + static_cast<size_t>(hs[0])
                                   * (static_cast<size_t>(i1)
                                      + static_cast<size_t>(hs[1]) * static_cast<size_t>(i2));
                    }

                    local_r[hidx] += vr * w;
                    if constexpr (gcplx) local_i[hidx] += vi * w;
                }
                sync();
            }
        }

        // ── 3. Flush local subgrid → global grid ──────────────────────────────
        //
        //  The final sync() inside the scatter loop guarantees all local_r writes
        //  are visible.  Each thread owns a disjoint strided slice of htot; no
        //  additional barrier is required.
        for (size_t idx = static_cast<size_t>(tid); idx < htot;
             idx += static_cast<size_t>(block_size)) {

            const RealType sum_r = local_r[idx];
            const RealType sum_i = gcplx ? local_i[idx] : RealType(0);
            if (sum_r == RealType(0) && sum_i == RealType(0))
                continue;

            // Decode flat index → histogram coordinates
            size_t tmp = idx;
            Kokkos::Array<int, Dim> hc{};
            for (unsigned d = 0; d < Dim; ++d) {
                hc[d] = static_cast<int>(tmp % static_cast<size_t>(hs[d]));
                tmp   /= static_cast<size_t>(hs[d]);
            }

            // Map histogram → local grid; skip ghost overflow
            Kokkos::Array<int, Dim> gc{};
            bool valid = true;
            for (unsigned d = 0; d < Dim; ++d) {
                const int lc = tb[d] + hc[d] - half_left;
                if (lc < -args.nghost || lc >= args.n_grid_local[d] + args.nghost) {
                    valid = false;
                    break;
                }
                gc[d] = lc + args.nghost;
            }
            if (!valid) continue;

            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                if constexpr (gcplx) {
                    RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                    Kokkos::atomic_add(&ptr[0], sum_r);
                    Kokkos::atomic_add(&ptr[1], sum_i);
                } else {
                    Kokkos::atomic_add(&args.grid(gc[Is]...),
                                       static_cast<grid_value_t>(sum_r));
                }
            }(std::make_index_sequence<Dim>{});
        }
    }

    // ── Kokkos operator ───────────────────────────────────────────────────────
    //
    //  Used for non-GPU backends (CPU, OpenMP, …) and as a GPU fallback when
    //  use_native_gpu is false.  Allocates scratch from Kokkos team scratch and
    //  calls run_tile with team.team_rank() as tid — exactly equivalent to
    //  threadIdx.x in the native path.  All threads in the team enter this
    //  operator simultaneously (Kokkos TeamPolicy semantics); run_tile's internal
    //  stride loops distribute work without a wrapping TeamThreadRange.
    KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
        using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
        constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
        constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

        const size_t htot = hist_total_;
        const int    np   = args.batch_np;

        scratch_real_view local_r_v(team.team_scratch(0), htot);
        scratch_real_view local_i_v;
        if constexpr (gcplx)
            local_i_v = scratch_real_view(team.team_scratch(0), htot);

        scratch_real_view kerevals_v(
            team.team_scratch(0),
            static_cast<size_t>(np * static_cast<int>(Dim) * W));
        scratch_real_view vals_r_v(team.team_scratch(0), static_cast<size_t>(np));
        scratch_real_view vals_i_v;
        if constexpr (gcplx && vcplx)
            vals_i_v = scratch_real_view(team.team_scratch(0), static_cast<size_t>(np));
        scratch_int_view shifts_v(
            team.team_scratch(0),
            static_cast<size_t>(np * static_cast<int>(Dim)));

        run_tile(
            team.team_rank(), team.team_size(),
            static_cast<size_t>(team.league_rank()),
            local_r_v.data(),
            gcplx            ? local_i_v.data() : nullptr,
            kerevals_v.data(),
            vals_r_v.data(),
            (gcplx && vcplx) ? vals_i_v.data() : nullptr,
            shifts_v.data(),
            KokkosBarrier{team});
    }

    // run() is defined below, after the native kernel template.
    void run(size_t n_particles);
};


// ── run() ─────────────────────────────────────────────────────────────────────
template <int W, class Types, class Policy>
void GridParallelScatter<W, Types, Policy>::run(size_t n_particles) {
    using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
    constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
    constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

    size_t n_tiles = 1;
    for (unsigned d = 0; d < Dim; ++d)
        n_tiles *= static_cast<size_t>(args.num_tiles[d]);

    if (n_tiles == 0 || n_particles == 0)
        return;

    hist_total_ = 1;
    for (unsigned d = 0; d < Dim; ++d)
        hist_total_ *= static_cast<size_t>(args.tile_size[d] + W + 1);

    const int np = args.batch_np;

    if constexpr (use_native_gpu) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        // Mirrors the alloc_r / alloc_i layout in grid_parallel_scatter_native.
        const size_t htot = hist_total_;
        size_t smem = (gcplx ? 2 : 1) * htot * sizeof(RealType)
                    + static_cast<size_t>(np * static_cast<int>(Dim) * W) * sizeof(RealType)
                    + static_cast<size_t>(np) * sizeof(RealType)
                    + (gcplx && vcplx ? static_cast<size_t>(np) * sizeof(RealType) : 0);
        smem  = (smem + sizeof(int) - 1) & ~(sizeof(int) - 1); // align before int array
        smem += static_cast<size_t>(np * static_cast<int>(Dim)) * sizeof(int);

        grid_parallel_scatter_native<W, Types, Policy>
            <<<static_cast<unsigned>(n_tiles), args.team_size, smem>>>(*this);
#endif
    } else {
        const size_t scratch = kokkos_scratch_size<gcplx, vcplx>(args.tile_size, np);
        Kokkos::parallel_for(
            "GridParallelScatter",
            team_policy(static_cast<int>(n_tiles), args.team_size)
                .set_scratch_size(0, Kokkos::PerTeam(scratch)),
            *this);
    }
}

} // namespace ippl::Interpolation::detail

#endif // IPPL_GRID_PARALLEL_SCATTER_H