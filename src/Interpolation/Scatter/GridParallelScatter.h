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
//             scratch_real_view kerevals(team.team_scratch(0), batches * static_cast<int>(Dim) * W);
//             scratch_real_view vals_r(team.team_scratch(0), batches);
//             scratch_real_view vals_i;
//             if constexpr (vcplx && gcplx)
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
//                             args.kernel((gp - (RealType(idx0 + wi) + RealType(0.5))) * args.inv_hw);
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

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

namespace ippl::Interpolation::detail {

    template <int W, class Types, class Policy>
    struct GridParallelScatter {
        static_assert(Policy::use_sorting,
                      "GridParallelScatter assumes sorted/bin-partitioned particles");

        static constexpr bool     requires_binning = true;
        static constexpr unsigned Dim              = Types::Dim;
        static constexpr int      half_left        = (W + 1) / 2;
        static constexpr int      vector_length    = 32;

        // ── Batch size for staged version ─────────────────────────────────────
        // Tune upward until shared-memory occupancy starts to drop.
        // Scratch ≈ htot + batch_np*(Dim*W + 1) floats + Dim*batch_np ints.
        static constexpr int batch_np = 16;

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

        // ── Arguments (oversubscription_factor removed) ───────────────────────
        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            Kokkos::View<size_t*, memory_space> permute;
            Kokkos::View<size_t*, memory_space> bin_offsets;
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
        size_t hist_total_ = 1;

        // ── Geometry helpers ───────────────────────────────────────────────────
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
                t /= static_cast<size_t>(args.num_tiles[d]);
            }
            return tb;
        }

        // ── Flat → ic[] decode, specialised per dimension ─────────────────────
        //
        // For Dim == 3 the naïve loop calls div and mod separately for each
        // dimension, giving three integer-divide pairs.  By precomputing the
        // combined stride hs01 = hs[0]*hs[1] we reduce to two divmod pairs.
        // Each pair is written as  q = n/s; r = n - q*s  so that a compiler
        // (or the hardware) can fuse them into a single divide instruction and
        // recover the remainder with a cheap subtract-multiply instead of a
        // second divide.
        //
        // Inputs
        //   idx  — flat index in [0, htot)
        //   hs   — histogram extents per dimension
        //   hs0, hs01 — precomputed strides (avoid recomputing inside every cell)
        KOKKOS_INLINE_FUNCTION void decode_ic(size_t idx, const Vector<int, Dim>& hs,
                                              size_t hs0, size_t hs01,
                                              Kokkos::Array<int, Dim>& ic) const {
            if constexpr (Dim == 1) {
                (void)hs; (void)hs0; (void)hs01;
                ic[0] = static_cast<int>(idx);

            } else if constexpr (Dim == 2) {
                (void)hs; (void)hs01;
                const size_t q0 = idx / hs0;
                ic[0] = static_cast<int>(idx - q0 * hs0);   // idx % hs0, no second divide
                ic[1] = static_cast<int>(q0);

            } else if constexpr (Dim == 3) {
                (void)hs;
                // Two divmod ops instead of three:
                const size_t q1  = idx / hs01;
                const size_t rem = idx - q1 * hs01;           // idx % hs01
                const size_t q0  = rem / hs0;
                ic[0] = static_cast<int>(rem - q0 * hs0);    // rem % hs0
                ic[1] = static_cast<int>(q0);
                ic[2] = static_cast<int>(q1);

            } else {
                // Generic fallback for Dim > 3
                size_t tmp = idx;
                for (unsigned d = 0; d < Dim; ++d) {
                    const size_t hsd = static_cast<size_t>(hs[d]);
                    const size_t q   = tmp / hsd;
                    ic[d]            = static_cast<int>(tmp - q * hsd);
                    tmp              = q;
                }
            }
        }

        // ── Scratch layout (staged version) ────────────────────────────────────
        //
        //  [A]  htot                 local_r      real subgrid
        //  [B]  htot                 local_i      imag subgrid (complex only)
        //  [C]  Dim * batch_np * W   kerevals     layout: [d][bi][wi]
        //  [D]  batch_np             vals_r
        //  [E]  batch_np             vals_i       (complex particle values only)
        //  [F]  Dim * batch_np       shifts       layout: [d][bi]
        //
        // Key vs. previous [bi][d][W] / [bi][d] layout:
        //   kbase[d] = kerevals + d*batch_np*W   is a compile-time-offset pointer.
        //   sbase[d] = shifts   + d*batch_np     is a compile-time-offset pointer.
        //   Gather loop accesses kbase[d][bi_w + wi] (1 add + load, no d-multiply)
        //   and sbase[d][bi] (0 arithmetic, just load), eliminating two multiplies
        //   per dimension per particle from the previous implementation.
        template <bool IsComplex>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /* */, int /* */) {
            size_t htot = 1;
            for (unsigned d = 0; d < Dim; ++d)
                htot *= static_cast<size_t>(tile_size[d] + W + 1);

            return (IsComplex ? 2 : 1) * scratch_real_view::shmem_size(htot)
                 + scratch_real_view::shmem_size(static_cast<int>(Dim) * batch_np * W)
                 + scratch_real_view::shmem_size(batch_np)
                 + (IsComplex ? scratch_real_view::shmem_size(batch_np) : 0)
                 + scratch_int_view::shmem_size(static_cast<int>(Dim) * batch_np);
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Operator 1 — staged batch (multiple parallel_for, shared-memory subgrid)
        //
        //  Integer cost per (cell, particle) in the gather hot loop:
        //    bi_w = bi * W         1 multiply  (hoisted outside the d-loop)
        //    sbase[d][bi]          1 load       (base pointer pre-offset, no mul)
        //    ic[d] - sh            1 subtract
        //    wi_u >= W (unsigned)  1 compare    (fuses wi<0 and wi>=W)
        //    kbase[d][bi_w + wi]   1 add + load (no d-multiply)
        //    w *= kw               1 FP mul
        //    acc += val * w        1-2 FMAs
        //
        //  Previous layout needed  bi*Dim  and  bi*Dim*W  multiplies per particle
        //  plus  d*W  per dimension — all eliminated here.
        // ════════════════════════════════════════════════════════════════════════
        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
            constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

            const size_t tile_id   = static_cast<size_t>(team.league_rank());
            const size_t bin_start = args.bin_offsets(tile_id);
            const size_t bin_end   = args.bin_offsets(tile_id + 1);
            const auto   tile_base = decode_tile_base(tile_id);
            const auto   hs        = hist_size();
            const size_t htot      = hist_total_;

            // Precompute decode strides once per tile — captured by the lambdas below.
            const size_t hs0  = static_cast<size_t>(hs[0]);
            const size_t hs01 = (Dim >= 3) ? hs0 * static_cast<size_t>(hs[1]) : size_t(1);

            // ── Scratch ────────────────────────────────────────────────────────
            scratch_real_view local_r(team.team_scratch(0), htot);
            scratch_real_view local_i;
            if constexpr (gcplx)
                local_i = scratch_real_view(team.team_scratch(0), htot);

            scratch_real_view kerevals_buf(team.team_scratch(0),
                                           static_cast<int>(Dim) * batch_np * W);
            scratch_real_view vals_r(team.team_scratch(0), batch_np);
            scratch_real_view vals_i;
            if constexpr (vcplx && gcplx)
                vals_i = scratch_real_view(team.team_scratch(0), batch_np);
            scratch_int_view shifts_buf(team.team_scratch(0), static_cast<int>(Dim) * batch_np);

            // Per-dim base pointers into scratch (register-sized, Dim ≤ 3 entries).
            // kbase[d] absorbs the d*batch_np*W offset so the gather only needs
            // kbase[d][bi_w + wi] = 1 add + 1 load.
            // sbase[d] absorbs d*batch_np so the gather only needs sbase[d][bi] = load.
            Kokkos::Array<RealType*, Dim> kbase{};
            Kokkos::Array<int*, Dim>      sbase{};
            for (unsigned d = 0; d < Dim; ++d) {
                kbase[d] = kerevals_buf.data() + static_cast<int>(d) * batch_np * W;
                sbase[d] = shifts_buf.data()   + static_cast<int>(d) * batch_np;
            }

            // ── Zero local subgrid ─────────────────────────────────────────────
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team, htot), [&](size_t i) {
                local_r(i) = RealType(0);
                if constexpr (gcplx)
                    local_i(i) = RealType(0);
            });
            team.team_barrier();

            const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
                                                               args.n_grid};

            // ── Batch loop ─────────────────────────────────────────────────────
            for (size_t bb = bin_start; bb < bin_end; bb += batch_np) {
                const int bs = static_cast<int>(
                    Kokkos::min(bin_end - bb, static_cast<size_t>(batch_np)));

                // Load phase — flat index is (wi, d, bi) so adjacent lanes touch
                // adjacent wi entries (coalesced reads of kbase[d]).
                // wi == 0 guard makes the sbase write non-racy: all W lanes for
                // a given (bi, d) would write the same value, but only one does.
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, bs * static_cast<int>(Dim) * W),
                    [&](int flat) {
                        const int wi = flat % W;
                        const int d  = (flat / W) % static_cast<int>(Dim);
                        const int bi = flat / (W * static_cast<int>(Dim));

                        const size_t   p    = args.permute(bb + static_cast<size_t>(bi));
                        const RealType gp   = transform.toGridCoordinate(args.x(p)[d], d);
                        const int      idx0 = transform.getStencilBase(gp - RealType(0.5), W);

                        kbase[d][bi * W + wi] =
                            args.kernel((gp - (RealType(idx0 + wi) + RealType(0.5))) * args.inv_hw);

                        if (wi == 0)
                            sbase[d][bi] =
                                idx0 - args.local_offset[d] + half_left - tile_base[d];
                    });

                Kokkos::parallel_for(Kokkos::TeamVectorRange(team, bs), [&](int bi) {
                    const size_t p = args.permute(bb + static_cast<size_t>(bi));
                    if constexpr (vcplx) {
                        vals_r(bi) = args.values(p).real();
                        if constexpr (gcplx)
                            vals_i(bi) = args.values(p).imag();
                    } else {
                        vals_r(bi) = static_cast<RealType>(args.values(p));
                    }
                });

                team.team_barrier();

                // Gather phase — each lane owns one output cell for the entire batch.
                // decode_ic() is called once per cell outside the particle loop.
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team, htot), [&](size_t idx) {
                    Kokkos::Array<int, Dim> ic{};
                    decode_ic(idx, hs, hs0, hs01, ic);

                    RealType acc_r = RealType(0);
                    RealType acc_i = RealType(0);

                    for (int bi = 0; bi < bs; ++bi) {
                        // bi*W hoisted: one multiply per particle, shared across Dim loads.
                        const int bi_w = bi * W;

                        RealType w  = RealType(1);
                        bool     ok = true;
                        for (unsigned d = 0; d < Dim; ++d) {
                            // Unsigned cast fuses (wi < 0 || wi >= W) into one compare.
                            const auto wi_u = static_cast<unsigned>(ic[d] - sbase[d][bi]);
                            if (wi_u >= static_cast<unsigned>(W)) { ok = false; break; }
                            // No d-multiply: kbase[d] already points at [d*batch_np*W].
                            w *= kbase[d][bi_w + static_cast<int>(wi_u)];
                        }
                        if (!ok) continue;

                        acc_r += vals_r(bi) * w;
                        if constexpr (gcplx)
                            acc_i += vals_i(bi) * w;
                    }

                    local_r(idx) += acc_r;
                    if constexpr (gcplx)
                        local_i(idx) += acc_i;
                });

                team.team_barrier();
            }

            // ── Flush local subgrid → global grid ──────────────────────────────
            // One atomic per live cell regardless of particle count.
            // decode_ic() re-runs here; cheaper than caching ic[] for htot cells.
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team, htot), [&](size_t idx) {
                Kokkos::Array<int, Dim> ic{};
                decode_ic(idx, hs, hs0, hs01, ic);

                Kokkos::Array<int, Dim> gc{};
                for (unsigned d = 0; d < Dim; ++d) {
                    const int loc = tile_base[d] + ic[d] - half_left;
                    if (loc < -args.nghost || loc >= args.n_grid_local[d] + args.nghost)
                        return;
                    gc[d] = loc + args.nghost;
                }

                [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    if constexpr (gcplx) {
                        RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                        Kokkos::atomic_add(&ptr[0], local_r(idx));
                        Kokkos::atomic_add(&ptr[1], local_i(idx));
                    } else {
                        Kokkos::atomic_add(&args.grid(gc[Is]...),
                                           static_cast<grid_value_t>(local_r(idx)));
                    }
                }(std::make_index_sequence<Dim>{});
            });
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Operator 2 — single parallel_for, register accumulation
        //
        //  Algorithm
        //  ---------
        //  One TeamVectorRange over htot cells; each GPU thread independently:
        //    1. Decodes its cell index ic[] and checks global-grid validity.
        //       Out-of-bounds cells return immediately — zero particle work.
        //    2. Iterates over every particle in the bin from global memory,
        //       evaluating the separable kernel inline (no kerevals staging).
        //    3. Accumulates acc_r / acc_i in registers.
        //    4. Issues a single atomic_add directly to global grid.
        //
        //  Properties
        //  ----------
        //  • Zero shared memory, zero barriers, zero inter-lane communication.
        //  • No zeroing pass, no reduction pass, no flush pass.
        //  • Particle positions are read htot-fold from global memory (one read
        //    per cell), so this version is bandwidth-bound for large bins.
        //  • Kernel is evaluated per cell per particle (not shared); worthwhile
        //    when the kernel is cheap relative to integer + synchronisation cost
        //    or when shared-memory pressure limits occupancy of the staged version.
        //  • Ideal regime: sparse bins (few particles), or kernels with small W.
        // ════════════════════════════════════════════════════════════════════════
        struct SinglePassTag {};

        KOKKOS_INLINE_FUNCTION void operator()(SinglePassTag, const team_member& team) const {
            using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
            constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

            const size_t tile_id   = static_cast<size_t>(team.league_rank());
            const size_t bin_start = args.bin_offsets(tile_id);
            const size_t bin_end   = args.bin_offsets(tile_id + 1);
            const auto   tile_base = decode_tile_base(tile_id);
            const auto   hs        = hist_size();

            const size_t hs0  = static_cast<size_t>(hs[0]);
            const size_t hs01 = (Dim >= 3) ? hs0 * static_cast<size_t>(hs[1]) : size_t(1);

            const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
                                                               args.n_grid};

            // Single parallel_for: one cell per lane.
            // No shared memory used — all intermediate state lives in registers.
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team, hist_total_), [&](size_t idx) {
                // ── Decode idx → ic[], gc[] (once per cell) ─────────────────────
                // gc[] validity check happens before the particle loop: out-of-bounds
                // ghost cells pay zero particle cost.
                Kokkos::Array<int, Dim> ic{};
                Kokkos::Array<int, Dim> gc{};
                decode_ic(idx, hs, hs0, hs01, ic);

                for (unsigned d = 0; d < Dim; ++d) {
                    const int loc = tile_base[d] + ic[d] - half_left;
                    if (loc < -args.nghost || loc >= args.n_grid_local[d] + args.nghost)
                        return;  // skip ghost: return from lambda == continue in parallel_for
                    gc[d] = loc + args.nghost;
                }

                // ── Register accumulation — all particles, no staging ────────────
                RealType acc_r = RealType(0);
                RealType acc_i = RealType(0);

                for (size_t ip = bin_start; ip < bin_end; ++ip) {
                    const size_t p = args.permute(ip);

                    // Inline separable kernel.  Early exit per dimension keeps cost
                    // proportional to the number of in-stencil dimensions rather than
                    // always paying for all Dim evaluations.
                    RealType w  = RealType(1);
                    bool     ok = true;
                    for (unsigned d = 0; d < Dim && ok; ++d) {
                        const RealType gp   = transform.toGridCoordinate(args.x(p)[d], d);
                        const int      base = transform.getStencilBase(gp - RealType(0.5), W);
                        const int      sh   =
                            base - args.local_offset[d] + half_left - tile_base[d];
                        const auto wi_u = static_cast<unsigned>(ic[d] - sh);
                        if (wi_u >= static_cast<unsigned>(W)) { ok = false; break; }
                        w *= args.kernel(
                            (gp - (RealType(base + static_cast<int>(wi_u)) + RealType(0.5)))
                            * args.inv_hw);
                    }
                    if (!ok) continue;

                    if constexpr (vcplx) {
                        acc_r += args.values(p).real() * w;
                        if constexpr (gcplx)
                            acc_i += args.values(p).imag() * w;
                    } else {
                        acc_r += static_cast<RealType>(args.values(p)) * w;
                    }
                }

                // ── Direct atomic write — no shared intermediate ─────────────────
                // Each lane writes to a disjoint global cell: no intra-team conflict.
                [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    if constexpr (gcplx) {
                        RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                        Kokkos::atomic_add(&ptr[0], acc_r);
                        Kokkos::atomic_add(&ptr[1], acc_i);
                    } else {
                        Kokkos::atomic_add(&args.grid(gc[Is]...),
                                           static_cast<grid_value_t>(acc_r));
                    }
                }(std::make_index_sequence<Dim>{});
            });
            // No team_barrier() — every lane wrote to a disjoint cell.
        }

        // ── Shared setup for both run() functions ──────────────────────────────
        KOKKOS_INLINE_FUNCTION size_t compute_n_tiles() const {
            size_t n = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n *= static_cast<size_t>(args.num_tiles[d]);
            return n;
        }

        void setup_hist_total() {
            hist_total_ = 1;
            for (unsigned d = 0; d < Dim; ++d)
                hist_total_ *= static_cast<size_t>(args.tile_size[d] + W + 1);
        }

        // ── run(): staged version ──────────────────────────────────────────────
        void run(size_t n_particles) {
            using grid_value_t  = typename decltype(args.grid)::non_const_value_type;
            constexpr bool cplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

            const size_t n_tiles = compute_n_tiles();
            if (n_tiles == 0 || n_particles == 0) return;

            setup_hist_total();
            const size_t scratch = compute_scratch_size<cplx>(args.tile_size, 1, 1);

            Kokkos::parallel_for(
                "GridParallelScatterStaged",
                team_policy(n_tiles, args.team_size, vector_length)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch)),
                *this);
        }

        // ── run_single_pass(): zero-scratch version ────────────────────────────
        void run_single_pass(size_t n_particles) {
            const size_t n_tiles = compute_n_tiles();
            if (n_tiles == 0 || n_particles == 0) return;

            setup_hist_total();

            using tagged_policy = Kokkos::TeamPolicy<execution_space, SinglePassTag>;
            Kokkos::parallel_for(
                "GridParallelScatterSinglePass",
                tagged_policy(n_tiles, args.team_size, vector_length),
                *this);
        }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_GRID_PARALLEL_SCATTER_H