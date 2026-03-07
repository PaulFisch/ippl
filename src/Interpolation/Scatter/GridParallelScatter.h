// #ifndef IPPL_GRID_PARALLEL_SCATTER_H
// #define IPPL_GRID_PARALLEL_SCATTER_H
//
// #include <Kokkos_Core.hpp>
//
// #include "Interpolation/CoordinateTransform.h"
// #include "Interpolation/Scatter/ScatterArgumentsBase.h"
//
// // ============================================================================
// //  GridParallelScatter  –  vectorized, conflict-free shared-memory histogramming
// // ============================================================================
// //
// // Algorithm overview
// // ------------------
// // The team is composed of `team_size` Kokkos "threads", each of which maps to
// // one GPU warp of `vector_length` (= 32) CUDA threads.  We call these warps
// // "vectors" throughout.
// //
// //   • Every vector owns a private slice of the shared-memory histogram:
// //       hist_r[ vec_id * hist_stride + local_entry ]
// //     Because only one vector ever writes to its own slice during the particle
// //     loop, NO atomics are needed at this stage.
// //
// //   • Particles in the bin are distributed across vectors in a round-robin
// //     fashion (stride = num_vectors).  For each particle, the vector's 32
// //     lanes cooperate via ThreadVectorRange to:
// //       1. Compute Dim*W kernel weights in parallel.
// //       2. Scatter W^Dim stencil contributions into the private histogram.
// //
// //   • After a team barrier, the reduction phase merges all per-vector copies:
// //     using a nested (TeamThreadRange × ThreadVectorRange) loop, every CUDA
// //     thread in the team sums exactly one histogram entry across all vectors,
// //     then issues a single atomic add to global memory.
// //
// // Bank-conflict avoidance
// // -----------------------
// // GPU shared memory has 32 banks (one per 4-byte word, cycling with stride 1).
// // Two accesses conflict when they target the same bank in the same clock.
// //
// //  Scatter phase: consecutive ThreadVectorRange lanes scatter to
// //    hist[ vec_id*stride + (bh + lane) ].  Consecutive lanes → consecutive
// //    banks → zero bank conflicts.
// //
// //  Reduction phase (inner loop over v):
// //    Lane l reads  hist[ v*stride + base + l ]  for v = 0..nv-1.
// //    Within one warp access (same v), the 32 lanes read 32 consecutive
// //    addresses → consecutive banks → zero conflicts.
// //    Across successive v, the stride between copies is `hist_stride`.
// //    If hist_stride were a multiple of 32, all v values would map the same
// //    logical entry l to the SAME bank → 32-way serial access.
// //    Solution: pad hist_stride to the next odd multiple of 32 (or simply add
// //    1 whenever htot ≡ 0 (mod 32)).  This scatters successive copies across
// //    different banks, keeping accesses conflict-free.
// //
// //  kw / base_s scratch: each vector owns a private slice (index by vec_id),
// //    sized Dim*W and Dim respectively.  Within a single ThreadVectorRange
// //    over Dim*W, lane l writes kw[l] — consecutive → consecutive banks.
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
//         // GPU warp size.  Every Kokkos "thread" in the team consists of this
//         // many CUDA threads, which collaborate through ThreadVectorRange.
//         static constexpr int vector_length = 32;
//
//         using RealType        = typename Types::RealType;
//         using ValueType       = typename Types::ValueType;
//         using memory_space    = typename Types::memory_space;
//         using execution_space = typename Types::execution_space;
//
//         // 3-level policy: league_size × team_size × vector_length
//         //   team_size   = number of warps  (Kokkos "threads") per team
//         //   vector_length = CUDA threads   (lanes) per warp
//         using team_policy   = Kokkos::TeamPolicy<execution_space>;
//         using team_member   = typename team_policy::member_type;
//         using scratch_space = typename execution_space::scratch_memory_space;
//
//         using scratch_real_view =
//             Kokkos::View<RealType*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
//         using scratch_int_view =
//             Kokkos::View<int*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
//
//         // ── Bank-conflict padding ────────────────────────────────────────────
//         // Pad the per-vector histogram stride so it is never a multiple of 32
//         // words.  This prevents all nv vectors from landing in the same bank
//         // for the same logical entry during the reduction loop.
//         static constexpr size_t kBankCount = 32;
//
//         KOKKOS_INLINE_FUNCTION
//         static size_t padded_stride(size_t htot) noexcept {
//             return (htot % kBankCount == 0) ? htot + 1 : htot;
//         }
//
//         // ── Scratch layout (per team) ────────────────────────────────────────
//         //
//         //  Offset  Length                     Purpose
//         //  ──────  ─────────────────────────  ──────────────────────────────
//         //  [A]     nv * hist_stride           hist_r  (real part)
//         //  [B]     nv * hist_stride           hist_i  (imag, complex only)
//         //  [C]     nv * Dim * W               kw      (kernel weights per vec)
//         //  [D]     nv * Dim                   base_s  (stencil bases  per vec)
//         //
//         //  nv = team_size  (number of warps per tile-team)
//         //
//         //  When z_batches > 1, the z-dimension of the histogram is reduced from
//         //  (tile_z + W + 1) to (tile_z + z_batch_size + 1), where
//         //  z_batch_size = ceil(W / z_batches).
//         // ────────────────────────────────────────────────────────────────────
//         template <bool IsComplex>
//         static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int team_size,
//                                            int z_batches = 1) {
//             const int z_batch_size = (W + z_batches - 1) / z_batches;
//
//             size_t htot = 1;
//             for (unsigned d = 0; d < Dim; ++d) {
//                 // For Dim==3 and d==2 (z-dimension), use reduced size when z_batches > 1
//                 const int hist_dim = (Dim == 3 && d == 2 && z_batches > 1)
//                                          ? tile_size[d] + z_batch_size + 1
//                                          : tile_size[d] + W + 1;
//                 htot *= static_cast<size_t>(hist_dim);
//             }
//
//             const int    nv     = std::max(1, team_size);
//             const size_t stride = padded_stride(htot);
//
//             return (IsComplex ? 2 : 1) * scratch_real_view::shmem_size(nv * stride)
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
//         // Set in run(), read in operator()
//         size_t sub_teams_per_tile_ = 1;   // oversubscription along the tile axis
//         size_t hist_stride_        = 1;   // padded stride between per-vector copies
//
//         // Z-batching state: set in run() before each batch, read in operator()
//         int z_batch_size_ = W;   // number of z-stencil points per batch
//         int z_start_      = 0;   // first z-stencil index for this batch
//         int z_end_        = W;   // one past last z-stencil index for this batch
//
//         // ── Geometry helpers ─────────────────────────────────────────────────
//         // When z_batches > 1, the z-dimension of the histogram is reduced.
//         KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
//             Vector<int, Dim> hs;
//             for (unsigned d = 0; d < Dim; ++d) {
//                 // For d==2 (z-dimension) in 3D, use z_batch_size_ instead of W
//                 hs[d] = args.tile_size[d]
//                         + ((Dim == 3 && d == 2) ? z_batch_size_ : W) + 1;
//             }
//             return hs;
//         }
//
//         KOKKOS_INLINE_FUNCTION size_t hist_total() const {
//             size_t n = 1;
//             for (unsigned d = 0; d < Dim; ++d) {
//                 const int hist_dim = args.tile_size[d]
//                                      + ((Dim == 3 && d == 2) ? z_batch_size_ : W) + 1;
//                 n *= static_cast<size_t>(hist_dim);
//             }
//             return n;
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
//             // ── Warp identity ────────────────────────────────────────────────
//             // team_rank ∈ [0, team_size): the rank of this warp within the team.
//             // The 32 CUDA threads of this warp all share the same team_rank and
//             // cooperate through ThreadVectorRange.
//             const int vec_id = team.team_rank();   // warp index within the team
//             const int nv     = team.team_size();   // total warps per team
//
//             // ── Tile / sub-team bookkeeping ──────────────────────────────────
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
//             const size_t htot      = hist_total();
//             const size_t stride    = hist_stride_;   // padded stride
//
//             // ── Scratch allocation ───────────────────────────────────────────
//             // Histograms: nv private copies, each padded to `stride` elements.
//             // Layout:  hist_r[ vec_id * stride + local_idx ]
//             // The padding ensures that, for fixed local_idx, consecutive vec_id
//             // values map to different shared-memory banks (see file header).
//             scratch_real_view hist_r(team.team_scratch(0), nv * stride);
//             scratch_real_view hist_i;
//             if constexpr (gcplx)
//                 hist_i = scratch_real_view(team.team_scratch(0), nv * stride);
//
//             // Kernel weights: nv private slices, each Dim*W elements.
//             // Layout:  kw[ vec_id * Dim * W + d * W + i ]
//             // Written by ThreadVectorRange during the particle loop; consecutive
//             // lanes write consecutive entries → consecutive banks, no conflicts.
//             scratch_real_view kw(team.team_scratch(0), nv * static_cast<int>(Dim) * W);
//
//             // Stencil bases: nv private slices, each Dim elements (int).
//             // Layout:  base_s[ vec_id * Dim + d ]
//             scratch_int_view base_s(team.team_scratch(0), nv * static_cast<int>(Dim));
//
//             // Convenience raw pointers into this warp's private slices
//             RealType* my_kw   = kw.data()    + vec_id * (static_cast<int>(Dim) * W);
//             int*      my_base = base_s.data() + vec_id * static_cast<int>(Dim);
//
//             // ── Zero this warp's histogram slice ─────────────────────────────
//             // All 32 lanes cooperate: lane l zeroes entries l, l+32, l+64, …
//             // Different warps zero different slices simultaneously — no conflicts.
//             Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, htot), [&](size_t i) {
//                 hist_r(vec_id * stride + i) = RealType(0);
//                 if constexpr (gcplx)
//                     hist_i(vec_id * stride + i) = RealType(0);
//             });
//             team.team_barrier();  // all slices zeroed before any warp starts scattering
//
//             // ── Particle loop ────────────────────────────────────────────────
//             // Round-robin across warps: warp vec_id handles particles
//             //   pstart + vec_id,  pstart + vec_id + nv,  pstart + vec_id + 2*nv, …
//             //
//             // For each particle the 32 lanes cooperate via ThreadVectorRange:
//             //   Step 1 – compute Dim*W kernel weights and Dim stencil bases.
//             //   Step 2 – scatter W^Dim stencil contributions to the private hist.
//             //
//             // Only this warp ever writes to hist_r[vec_id * stride + *], so the
//             // += operations in Step 2 are RACE-FREE without any atomics.
//             const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
//                                                                args.n_grid};
//
//             for (size_t ip = pstart + static_cast<size_t>(vec_id); ip < pend;
//                  ip += static_cast<size_t>(nv)) {
//
//                 const size_t p = args.permute(ip);
//
//                 // Particle value — identical for all 32 lanes, read redundantly
//                 // (broadcast from L2/L1 cache; no divergence).
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
//                 // Lane `flat` computes weight for dimension d = flat/W, offset i = flat%W.
//                 // After the range completes (synchronous within the warp), my_kw[0..DimW-1]
//                 // and my_base[0..Dim-1] are fully initialised in shared memory.
//                 //
//                 // Bank mapping: lane l writes kw[vec_id*DimW + l] → bank (vec_id*DimW + l) % 32.
//                 // Consecutive lanes → consecutive banks → zero conflicts.
//                 Kokkos::parallel_for(
//                     Kokkos::ThreadVectorRange(team, static_cast<int>(Dim) * W),
//                     [&](int flat) {
//                         const int      d    = flat / W;
//                         const int      i    = flat % W;
//                         const RealType gp   = transform.toGridCoordinate(args.x(p)[d], d);
//                         const int      idx0 = transform.getStencilBase(gp - RealType(0.5), W);
//                         // Normalised offset from the stencil base (in [0, 1) for i=0)
//                         my_kw[d * W + i] = args.kernel((gp - (RealType(idx0 + i) + RealType(0.5))) * args.inv_hw);
//                         // Only the lane with i==0 writes the base for dimension d.
//                         // If Dim*W > 32 there may be multiple rounds; lane `flat%32` is
//                         // unique per entry within each round, so no write races.
//                         if (i == 0)
//                             my_base[d] = idx0 - args.local_offset[d];
//                     });
//                 // ThreadVectorRange is synchronous inside the warp — no extra barrier.
//
//                 // ── Step 2: scatter to private histogram ──────────────────────
//                 // Pointer into this warp's slice (vec_id already baked in).
//                 RealType* h_r = hist_r.data() + vec_id * stride;
//                 RealType* h_i = gcplx ? hist_i.data() + vec_id * stride : nullptr;
//
//                 if constexpr (Dim == 1) {
//                     const int bh0 = my_base[0] + half_left - tile_base[0];
//
//                     Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W), [&](int i0) {
//                         // Lane i0 handles stencil offset i0; hidx is unique per lane
//                         // → consecutive banks for consecutive i0 → no conflicts.
//                         const size_t hidx = static_cast<size_t>(bh0 + i0);
//                         const RealType w  = my_kw[i0];
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
//                         // LayoutLeft-friendly order: i0 is fastest (innermost dim).
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
//                     const int bh0 = my_base[0] + half_left - tile_base[0];
//                     const int bh1 = my_base[1] + half_left - tile_base[1];
//                     const int bh2 = my_base[2] + half_left - tile_base[2];
//
//                     // When z_batches > 1, only process z-stencil indices in [z_start_, z_end_).
//                     // The histogram z-index is adjusted by subtracting z_start_.
//                     const int z_range = z_end_ - z_start_;
//                     Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W * W * z_range),
//                                          [&](int flat) {
//                         const int      i0   = flat % W;
//                         const int      i1   = (flat / W) % W;
//                         const int      i2   = z_start_ + flat / (W * W);
//                         const RealType w    = my_kw[i0] * my_kw[W + i1] * my_kw[2 * W + i2];
//                         // Adjust histogram z-index by subtracting z_start_
//                         const size_t   hidx = static_cast<size_t>(bh0 + i0)
//                                             + static_cast<size_t>(hs[0])
//                                                   * (static_cast<size_t>(bh1 + i1)
//                                                      + static_cast<size_t>(hs[1])
//                                                            * static_cast<size_t>(bh2 + i2 - z_start_));
//                         h_r[hidx] += val_r * w;
//                         if constexpr (gcplx)
//                             h_i[hidx] += val_i * w;
//                     });
//                 }
//                 // ThreadVectorRange is synchronous → the next particle iteration
//                 // is safe without an extra barrier.
//             }
//
//             // Wait for ALL warps to complete their particle loops before reduction.
//             team.team_barrier();
//
//             // ── Reduction: merge nv histogram copies → global grid ────────────
//             //
//             // We need every histogram entry to be summed across all nv copies and
//             // then atomically added to the global grid EXACTLY ONCE (no redundancy).
//             //
//             // Strategy: distribute htot entries among all team_size * vector_length
//             // CUDA threads using a two-level loop:
//             //
//             //   TeamThreadRange over chunks (each chunk = one warp's work):
//             //     → distributes `ceil(htot / VL)` chunks among the `nv` warps
//             //
//             //   ThreadVectorRange(team, VL) inside each chunk:
//             //     → 32 lanes each reduce one histogram entry
//             //
//             // Each CUDA thread thus handles exactly one entry (for full tiles) and
//             // issues exactly one atomic add.  No entry is touched by two threads.
//             //
//             // Bank analysis for the inner loop over v:
//             //   Lane l reads  hist_r[ v * stride + base + l ]  for v = 0..nv-1.
//             //   Within one v, 32 consecutive addresses → 32 consecutive banks → OK.
//             //   Across v, the stride shifts the bank by  stride % 32 positions.
//             //   Because stride is padded to be non-divisible by 32, this shift is
//             //   never 0, so consecutive v never re-use the same bank → OK.
//             const size_t chunks =
//                 (htot + static_cast<size_t>(vector_length) - 1)
//                 / static_cast<size_t>(vector_length);
//
//             Kokkos::parallel_for(Kokkos::TeamThreadRange(team, chunks), [&](size_t chunk) {
//                 Kokkos::parallel_for(
//                     Kokkos::ThreadVectorRange(team, vector_length), [&](int lane) {
//                         const size_t idx = chunk * static_cast<size_t>(vector_length)
//                                          + static_cast<size_t>(lane);
//                         if (idx >= htot)
//                             return;
//
//                         // Sum this entry across all nv per-vector copies.
//                         // Sequential over v, but nv is small (e.g. 4) and the
//                         // accesses pattern is stride-friendly (see bank analysis above).
//                         RealType sum_r = RealType(0);
//                         RealType sum_i = RealType(0);
//                         for (int v = 0; v < nv; ++v) {
//                             sum_r += hist_r(v * stride + idx);
//                             if constexpr (gcplx)
//                                 sum_i += hist_i(v * stride + idx);
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
//                         // For z-dimension (d==2) in 3D with z-batching, add z_start_ back
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
//                         // One atomic per entry — much fewer than the original design
//                         // where every particle-stencil-point pair was atomic.
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
//         //
//         // Launch parameters
//         // -----------------
//         //   league_size = n_tiles × oversubscription_factor
//         //   team_size   = args.team_size  (= number of warps per tile-team)
//         //   vector_length = 32            (CUDA lanes per warp)
//         //
//         // Sub-team oversubscription splits a tile's bin into
//         // `oversubscription_factor` sub-ranges handled by different teams.
//         // This hides latency when bins are large.
//         //
//         // Z-batching (z_batches > 1): splits the W z-stencil points into
//         // z_batches batches to reduce shared memory pressure.  Each batch
//         // processes ceil(W/z_batches) z-stencil points.  The kernel is
//         // launched once per batch; all batches accumulate to the same grid
//         // via atomic adds.
//         //
//         // Scratch memory is allocated at level 0 (shared memory on GPU).
//         // ─────────────────────────────────────────────────────────────────────
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
//             // Z-batching setup
//             const int z_batches = (Dim == 3) ? std::max(1, args.z_batches) : 1;
//             z_batch_size_       = (W + z_batches - 1) / z_batches;
//
//             // Compute padded histogram stride (bank-conflict avoidance, see header)
//             // With z-batching, z-dimension uses z_batch_size_ instead of W
//             size_t htot = 1;
//             for (unsigned d = 0; d < Dim; ++d) {
//                 const int hist_dim = args.tile_size[d]
//                                      + ((Dim == 3 && d == 2) ? z_batch_size_ : W) + 1;
//                 htot *= static_cast<size_t>(hist_dim);
//             }
//             hist_stride_ = padded_stride(htot);
//
//             const size_t scratch =
//                 compute_scratch_size<cplx>(args.tile_size, args.team_size, z_batches);
//
//             // Launch kernel once per z-batch.  When z_batches=1, this is a single launch.
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

#ifndef IPPL_GRID_PARALLEL_SCATTER_H
#define IPPL_GRID_PARALLEL_SCATTER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

// ============================================================================
//  GridParallelScatter — vectorized, conflict-free shared-memory histogramming
// ============================================================================
//
// Changes from prior version
// --------------------------
//  1. BANK-CONFLICT FIX (the original analysis was wrong in two ways):
//
//     a) Scatter phase — the real hazard:
//        For W < 32 a single 32-lane warp iteration spans multiple histogram
//        rows. Consecutive lanes that cross a row boundary jump by hs0 words
//        instead of 1.  If hs0 ≡ 0 (mod 32) every row starts in the same
//        bank ⟹ up to (32/W)-way conflict.
//        Fix: pad hs0_stride_ = bank_pad(tile_size[0]+W+1) so it is never
//        divisible by 32.  All hidx computations use hs0_stride_ as the
//        x-dimension stride.  The padded slots are never written (they decode
//        outside hist_ext(0) and are discarded in the reduction) but they
//        shift successive rows into different banks.
//
//     b) Reduction phase — the ORIGINAL claim was wrong:
//        The v-loop is SEQUENTIAL (one v per clock), so consecutive v values
//        never produce simultaneous conflicting accesses.  Within one v, the
//        32 lanes read 32 consecutive addresses ⟹ 32 distinct banks, always.
//        ⟹ NO padding of hist_stride_ is needed for bank-conflict avoidance.
//        hist_stride_ == htot (using hs0_stride_ as the x-stride).
//
//  2. REDUCED INTEGER INSTRUCTIONS — weight-computation restructuring:
//     Original: for each of Dim*W work items, call toGridCoordinate and
//       getStencilBase, computing the grid coord W times per dimension.
//     New: two separate passes per particle:
//       Pass A (Dim work items): compute gp[d] and base_s[d] once per dim.
//       Pass B (Dim*W work items): compute kernel(gp[d], base_s[d], i).
//     Savings: (W-1)*Dim fewer toGridCoordinate + getStencilBase calls per
//     particle.  For W=8, Dim=3: 21 fewer floating-point-heavy calls.
//     Requires nv*Dim extra RealType scratch words (gp_s).
//
//  3. NATIVE GPU KERNEL (GridParallelScatterNative):
//     Compiled only when KOKKOS_ENABLE_CUDA or KOKKOS_ENABLE_HIP is defined.
//     Replaces all Kokkos team/vector wrappers with direct hardware idioms:
//       threadIdx.x = lane (0..31), threadIdx.y = vec_id (0..nv-1)
//       blockIdx.x  = league_rank
//       extern __shared__ char smem[] for scratch
//       __syncwarp / __syncthreads instead of team barriers
//       __ldg for read-only global loads
//       #pragma unroll on W-bounded loops
//       __launch_bounds__ to guide register allocation
//     Falls back to an alias of GridParallelScatter on CPU-only builds so
//     call-sites need no #ifdefs.
//
// Bank-conflict analysis (corrected)
// -----------------------------------
//  Histogram layout: hidx = x + hs0_stride_*y + hs0_stride_*hs1*z
//  (hs0_stride_ is bank-padded; hs1, hs2 are unpadded extents)
//
//  Scatter phase, one 32-lane warp iteration, flat = b..b+31:
//    i0 = flat % W,  i1 = flat / W  (Dim=2 example)
//    hidx = bh0+i0 + hs0_stride_*(bh1+i1)
//    Consecutive flat within same row (same i1): hidx differs by 1 → OK.
//    Row boundary (flat = kW-1 → kW): hidx jumps by hs0_stride_-(W-1).
//    With hs0_stride_ % 32 ≠ 0, this jump is never ≡ -(W-1) mod 32 in a
//    way that repeats a bank; consecutive rows land in shifted banks → OK.
//
//  kw scratch: lane flat writes kw[vec_id*DimW + flat] → consecutive banks.
//  gp_s scratch: lane d writes gp_s[vec_id*Dim + d], Dim ≤ 3 → no conflict.
// ============================================================================

// ─── GPU backend detection ───────────────────────────────────────────────────
#if defined(KOKKOS_ENABLE_CUDA)
#  define IPPL_GPU_BACKEND_CUDA 1
#  define IPPL_HAS_GPU           1
#elif defined(KOKKOS_ENABLE_HIP)
#  define IPPL_GPU_BACKEND_HIP  1
#  define IPPL_HAS_GPU           1
#endif

namespace ippl::Interpolation::detail {

// ─── Bank-conflict-free stride helper ────────────────────────────────────────
//
//  Returns n if n % 32 != 0, else n+1.
//  Guarantees that stride-n accesses from a 32-lane warp always visit distinct
//  shared-memory banks (bank = (word_address % 32) for 4-byte banks).
//
static constexpr size_t kBankCount = 32;

KOKKOS_INLINE_FUNCTION
static constexpr size_t bank_pad(size_t n) noexcept {
    return (n % kBankCount == 0u) ? n + 1u : n;
}

// ─── Helper: integer div/mod using bit-ops when W is a power of two ──────────
template <int W>
KOKKOS_FORCEINLINE_FUNCTION static constexpr int idiv(int x) noexcept {
    if constexpr ((W & (W - 1)) == 0) {
        // Compile-time shift for power-of-2 W
        constexpr int shift = __builtin_ctz(static_cast<unsigned>(W));
        return x >> shift;
    } else {
        return x / W;
    }
}
template <int W>
KOKKOS_FORCEINLINE_FUNCTION static constexpr int imod(int x) noexcept {
    if constexpr ((W & (W - 1)) == 0) {
        return x & (W - 1);
    } else {
        return x % W;
    }
}

// ============================================================================
//  GridParallelScatter  (Kokkos version — portable, all backends)
// ============================================================================

template <int W, class Types, class Policy>
struct GridParallelScatter {
    static_assert(Policy::use_sorting,
                  "GridParallelScatter assumes sorted/bin-partitioned particles");

    static constexpr bool     requires_binning = true;
    static constexpr unsigned Dim              = Types::Dim;
    static constexpr int      half_left        = (W + 1) / 2;
    static constexpr int      vector_length    = 32;

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

    // ── Scratch layout (per team) ────────────────────────────────────────────
    //
    //  [A]  nv × hist_stride   RealType  hist_r
    //  [B]  nv × hist_stride   RealType  hist_i    (complex grids only)
    //  [C]  nv × Dim × W       RealType  kw        (kernel weights)
    //  [D]  nv × Dim           RealType  gp_s      (grid coords, one per dim) ← NEW
    //  [E]  nv × Dim           int       base_s    (stencil base, one per dim)
    //
    //  hist_stride = hs0_stride_ × hs1 × [hs2]
    //  hs0_stride_ = bank_pad(tile_size[0] + W + 1)   ← ensures ≢ 0 (mod 32)
    // ────────────────────────────────────────────────────────────────────────

    // Compute hs0_stride (static, for scratch-size queries before run())
    static size_t compute_hs0(const Vector<int, Dim>& tile_size) noexcept {
        return bank_pad(static_cast<size_t>(tile_size[0] + W + 1));
    }

    template <bool IsComplex>
    static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int team_size,
                                       int z_batches = 1) {
        const int    z_batch_size = (W + z_batches - 1) / z_batches;
        const size_t hs0          = compute_hs0(tile_size);
        size_t       htot         = hs0;
        for (unsigned d = 1; d < Dim; ++d) {
            const int ext = (Dim == 3 && d == 2 && z_batches > 1)
                                ? tile_size[d] + z_batch_size + 1
                                : tile_size[d] + W + 1;
            htot *= static_cast<size_t>(ext);
        }
        const int nv = std::max(1, team_size);
        // [A]+[B] hist, [C] kw, [D] gp_s, [E] base_s
        return (IsComplex ? 2 : 1) * scratch_real_view::shmem_size(nv * htot)
             + scratch_real_view::shmem_size(nv * static_cast<int>(Dim) * W)
             + scratch_real_view::shmem_size(nv * static_cast<int>(Dim))   // gp_s
             + scratch_int_view::shmem_size(nv * static_cast<int>(Dim));
    }

    // ── Arguments ───────────────────────────────────────────────────────────
    struct Arguments : ScatterArgumentsBase<Arguments, Types> {
        Kokkos::View<size_t*, memory_space> permute;
        Kokkos::View<size_t*, memory_space> bin_offsets;
        Vector<int, Dim> num_tiles;
        Vector<int, Dim> tile_size;
        int team_size;
        int oversubscription_factor;
        int z_batches;

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
            a.z_batches               = config.z_batches;
            return a;
        }
    };

    Arguments args;

    // Set in run(), read in operator()
    size_t sub_teams_per_tile_ = 1;
    size_t hist_stride_        = 1;  // htot with bank-padded x-stride
    size_t hs0_stride_         = 1;  // bank-padded x-dim histogram stride

    // Z-batching state
    int z_batch_size_ = W;
    int z_start_      = 0;
    int z_end_        = W;

    // ── Geometry helpers ─────────────────────────────────────────────────────
    // Unpadded extent of histogram dimension d (for range checks & y/z strides)
    KOKKOS_INLINE_FUNCTION int hist_ext(unsigned d) const noexcept {
        if (Dim == 3 && d == 2) return args.tile_size[d] + z_batch_size_ + 1;
        return args.tile_size[d] + W + 1;
    }

    // Stride in the flat histogram array along dimension d.
    // d=0 uses the bank-padded hs0_stride_; d>=1 multiply further.
    KOKKOS_INLINE_FUNCTION size_t hist_stride_d1() const noexcept {
        return hs0_stride_;                              // stride per y-row
    }
    KOKKOS_INLINE_FUNCTION size_t hist_stride_d2() const noexcept {
        return hs0_stride_ * static_cast<size_t>(hist_ext(1));  // stride per z-slab
    }

    KOKKOS_INLINE_FUNCTION Vector<int, Dim> decode_tile_base(size_t tile_id) const noexcept {
        Vector<int, Dim> tb;
        for (size_t t = tile_id, d = Dim; d-- > 0;) {
            tb[d] = static_cast<int>(t % static_cast<size_t>(args.num_tiles[d]))
                    * args.tile_size[d];
            t /= static_cast<size_t>(args.num_tiles[d]);
        }
        return tb;
    }

    // ── Kernel operator ──────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
        using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
        constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
        constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

        const int    vec_id   = team.team_rank();
        const int    nv       = team.team_size();
        const size_t league_r = static_cast<size_t>(team.league_rank());
        const size_t tile_id  = league_r / sub_teams_per_tile_;
        const size_t sub_id   = league_r % sub_teams_per_tile_;

        const size_t bin_start = args.bin_offsets(tile_id);
        const size_t bin_end   = args.bin_offsets(tile_id + 1);
        const size_t bin_size  = bin_end - bin_start;

        const size_t ppt    = (bin_size + sub_teams_per_tile_ - 1) / sub_teams_per_tile_;
        const size_t pstart = bin_start + sub_id * ppt;
        if (pstart >= bin_end) return;
        const size_t pend = Kokkos::min(bin_end, pstart + ppt);

        const auto   tile_base = decode_tile_base(tile_id);
        const size_t htot      = hist_stride_;   // == hist_stride_ (== htot, no extra pad)
        const size_t stride    = hist_stride_;

        // Precompute histogram strides (avoids redundant multiplies in scatter)
        const size_t sx  = hs0_stride_;
        const size_t sxy = (Dim >= 3) ? hist_stride_d2() : size_t(0);

        // ── Scratch ──────────────────────────────────────────────────────────
        scratch_real_view hist_r(team.team_scratch(0), nv * stride);
        scratch_real_view hist_i;
        if constexpr (gcplx)
            hist_i = scratch_real_view(team.team_scratch(0), nv * stride);

        scratch_real_view kw_view(team.team_scratch(0), nv * static_cast<int>(Dim) * W);
        scratch_real_view gp_view(team.team_scratch(0), nv * static_cast<int>(Dim));
        scratch_int_view  base_view(team.team_scratch(0), nv * static_cast<int>(Dim));

        // Per-warp raw pointers into private slices
        RealType* my_kw   = kw_view.data()   + vec_id * (static_cast<int>(Dim) * W);
        RealType* my_gp   = gp_view.data()   + vec_id * static_cast<int>(Dim);
        int*      my_base = base_view.data()  + vec_id * static_cast<int>(Dim);

        // ── Zero this warp's histogram slice ─────────────────────────────────
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, htot), [&](size_t i) {
            hist_r(vec_id * stride + i) = RealType(0);
            if constexpr (gcplx)
                hist_i(vec_id * stride + i) = RealType(0);
        });
        team.team_barrier();

        const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};

        // ── Particle loop ─────────────────────────────────────────────────────
        for (size_t ip = pstart + static_cast<size_t>(vec_id); ip < pend;
             ip += static_cast<size_t>(nv)) {

            const size_t p = args.permute(ip);

            RealType val_r = RealType(0), val_i = RealType(0);
            if constexpr (vcplx) {
                val_r = args.values(p).real();
                if constexpr (gcplx) val_i = args.values(p).imag();
            } else {
                val_r = static_cast<RealType>(args.values(p));
            }

            // ── Step 1a: grid coords + stencil bases, ONE per dimension ───────
            // Only Dim lanes are active; saves (W-1)*Dim redundant coordinate
            // transforms compared to the original combined DimW loop.
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, static_cast<int>(Dim)), [&](int d) {
                    const RealType gp = transform.toGridCoordinate(args.x(p)[d], d);
                    my_gp[d]          = gp;
                    my_base[d]        = transform.getStencilBase(gp - RealType(0.5), W)
                                        - args.local_offset[d];
                });
            // ThreadVectorRange is warp-synchronous; my_gp/my_base are visible below.

            // ── Step 1b: kernel weights, Dim*W entries ────────────────────────
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, static_cast<int>(Dim) * W), [&](int flat) {
                    const int      d    = flat / W;
                    const int      i    = flat % W;
                    const int      idx0 = my_base[d] + args.local_offset[d];
                    my_kw[flat] = args.kernel(
                        (my_gp[d] - (RealType(idx0 + i) + RealType(0.5))) * args.inv_hw);
                });

            // ── Step 2: scatter to this warp's private histogram slice ────────
            RealType* h_r = hist_r.data() + vec_id * stride;
            RealType* h_i = gcplx ? hist_i.data() + vec_id * stride : nullptr;

            if constexpr (Dim == 1) {
                const int bh0 = my_base[0] + half_left - tile_base[0];
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W), [&](int i0) {
                    h_r[static_cast<size_t>(bh0 + i0)] += val_r * my_kw[i0];
                    if constexpr (gcplx)
                        h_i[static_cast<size_t>(bh0 + i0)] += val_i * my_kw[i0];
                });

            } else if constexpr (Dim == 2) {
                const int bh0 = my_base[0] + half_left - tile_base[0];
                const int bh1 = my_base[1] + half_left - tile_base[1];
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W * W), [&](int flat) {
                    const int    i0   = imod<W>(flat);
                    const int    i1   = idiv<W>(flat);
                    // Use bank-padded sx (== hs0_stride_) as the row stride
                    const size_t hidx = static_cast<size_t>(bh0 + i0)
                                      + sx * static_cast<size_t>(bh1 + i1);
                    const RealType w  = my_kw[i0] * my_kw[W + i1];
                    h_r[hidx] += val_r * w;
                    if constexpr (gcplx) h_i[hidx] += val_i * w;
                });

            } else if constexpr (Dim == 3) {
                const int bh0    = my_base[0] + half_left - tile_base[0];
                const int bh1    = my_base[1] + half_left - tile_base[1];
                const int bh2    = my_base[2] + half_left - tile_base[2];
                const int hs1    = hist_ext(1);
                const int zrange = z_end_ - z_start_;
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, W * W * zrange), [&](int flat) {
                        const int      i0   = imod<W>(flat);
                        const int      i1   = imod<W>(idiv<W>(flat));
                        const int      i2   = z_start_ + flat / (W * W);
                        const RealType w    = my_kw[i0] * my_kw[W + i1] * my_kw[2 * W + i2];
                        const size_t   hidx = static_cast<size_t>(bh0 + i0)
                                           + sx * (static_cast<size_t>(bh1 + i1)
                                                   + static_cast<size_t>(hs1)
                                                         * static_cast<size_t>(bh2 + i2 - z_start_));
                        h_r[hidx] += val_r * w;
                        if constexpr (gcplx) h_i[hidx] += val_i * w;
                    });
            }
        }  // particle loop

        team.team_barrier();

        // ── Reduction: merge nv per-vector copies → global grid ───────────────
        //
        // Each of (nv × 32) CUDA threads handles one histogram entry.
        // Distribution: outer TeamThreadRange over warps, inner ThreadVectorRange
        // over 32 lanes.  No entry is touched by two threads → one atomic per entry.
        //
        // Bank analysis (corrected):
        //   The v-loop is SEQUENTIAL — only one v is active per clock.
        //   Within one v, 32 lanes read 32 consecutive addresses → 32 distinct banks.
        //   No bank conflicts possible regardless of hist_stride_ value.
        //   Therefore hist_stride_ requires no additional padding.
        //
        const size_t chunks = (htot + static_cast<size_t>(vector_length) - 1)
                            / static_cast<size_t>(vector_length);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, chunks), [&](size_t chunk) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, vector_length), [&](int lane) {
                    const size_t idx = chunk * static_cast<size_t>(vector_length)
                                     + static_cast<size_t>(lane);
                    if (idx >= htot) return;

                    RealType sum_r = RealType(0), sum_i = RealType(0);
                    for (int v = 0; v < nv; ++v) {
                        sum_r += hist_r(v * stride + idx);
                        if constexpr (gcplx)
                            sum_i += hist_i(v * stride + idx);
                    }

                    // Decode flat idx using bank-padded hs0_stride_ for dimension 0.
                    // Entries in the padding region (x >= hist_ext(0)) are discarded.
                    size_t tmp = idx;
                    Kokkos::Array<int, Dim> hc{};
                    hc[0] = static_cast<int>(tmp % hs0_stride_);
                    tmp   /= hs0_stride_;
                    if (hc[0] >= hist_ext(0)) return;   // padding slot, skip
                    if constexpr (Dim >= 2) {
                        hc[1] = static_cast<int>(tmp % static_cast<size_t>(hist_ext(1)));
                        tmp   /= static_cast<size_t>(hist_ext(1));
                    }
                    if constexpr (Dim >= 3) {
                        hc[2] = static_cast<int>(tmp);
                    }

                    // Map histogram coords → local grid coords; skip ghosts overflow
                    Kokkos::Array<int, Dim> gc{};
                    for (unsigned d = 0; d < Dim; ++d) {
                        int hc_d = hc[d];
                        // For the z-dimension under z-batching, add z_start_ back
                        if constexpr (Dim == 3) { if (d == 2) hc_d += z_start_; }
                        const int local = tile_base[d] + hc_d - half_left;
                        if (local < -args.nghost
                            || local >= args.n_grid_local[d] + args.nghost)
                            return;
                        gc[d] = local + args.nghost;
                    }

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
                });
        });
    }  // operator()

    // ── run() ────────────────────────────────────────────────────────────────
    void run(size_t n_particles) {
        using grid_value_t  = typename decltype(args.grid)::non_const_value_type;
        constexpr bool cplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

        size_t n_tiles = 1;
        for (unsigned d = 0; d < Dim; ++d)
            n_tiles *= static_cast<size_t>(args.num_tiles[d]);
        if (n_tiles == 0 || n_particles == 0) return;

        sub_teams_per_tile_ = std::max(size_t(1), size_t(args.oversubscription_factor));

        const int z_batches = (Dim == 3) ? std::max(1, args.z_batches) : 1;
        z_batch_size_       = (W + z_batches - 1) / z_batches;

        // Correct bank-conflict-free x-dimension stride (see header)
        hs0_stride_ = bank_pad(static_cast<size_t>(args.tile_size[0] + W + 1));

        // Total histogram size per vector copy; NO extra reduction-phase padding needed
        size_t htot = hs0_stride_;
        for (unsigned d = 1; d < Dim; ++d) {
            const int ext = (Dim == 3 && d == 2)
                                ? args.tile_size[d] + z_batch_size_ + 1
                                : args.tile_size[d] + W + 1;
            htot *= static_cast<size_t>(ext);
        }
        hist_stride_ = htot;

        const size_t scratch =
            compute_scratch_size<cplx>(args.tile_size, args.team_size, z_batches);

        for (int batch = 0; batch < z_batches; ++batch) {
            z_start_ = batch * z_batch_size_;
            z_end_   = std::min((batch + 1) * z_batch_size_, W);

            Kokkos::parallel_for(
                "GridParallelScatter",
                team_policy(n_tiles * sub_teams_per_tile_, args.team_size, vector_length)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch)),
                *this);
        }
    }
};  // GridParallelScatter


// ============================================================================
//  GridParallelScatterNative  — raw CUDA/HIP kernel, no Kokkos wrapper overhead
// ============================================================================
//
//  Thread layout mirrors the Kokkos TeamPolicy(league, team, 32):
//    blockDim.x = 32          (= vector_length, one warp's CUDA threads)
//    blockDim.y = team_size   (= nv, warps per block)
//    gridDim.x  = n_tiles × oversubscription_factor
//
//  Advantages over the Kokkos version:
//    • threadIdx.x / threadIdx.y / blockIdx.x are hardware registers — zero
//      overhead; no team_rank() function calls or warp-mask operations.
//    • __shared__ arrays with statically-computable sizes allow the compiler
//      to place them in registers when small enough.
//    • __syncwarp() is a single instruction vs. team_barrier() overhead.
//    • __ldg() routes read-only global loads through the texture cache.
//    • #pragma unroll on W-bounded loops eliminates branch instructions.
//    • __launch_bounds__ guides register allocation for higher occupancy.
//
//  Safety guarantees:
//    • Guarded by #ifdef IPPL_HAS_GPU — not compiled on CPU-only builds.
//    • On CPU builds, GridParallelScatterNative is an alias of the portable
//      Kokkos version so call-sites need zero #ifdefs.
//    • The kernel body uses only standard CUDA/HIP intrinsics; no CUDA-
//      specific extensions are used outside the guarded region.
// ============================================================================

#ifdef IPPL_HAS_GPU

// Forward declaration so the __global__ function can reference the struct
template <int W, class Types, class Policy>
struct GridParallelScatterNative;

// ─── __global__ kernel ───────────────────────────────────────────────────────
//
//  The entire argument struct is passed by value (copied into device memory by
//  the CUDA runtime).  Kokkos::View objects are trivially copyable handles.
//
//  __launch_bounds__(MAX_THREADS, MIN_BLOCKS):
//    MAX_THREADS = 256  (= 8 warps × 32 lanes — typical config; tune per arch)
//    MIN_BLOCKS  = 1    (conservative; raise if register count allows)
//
template <int W, class Types, class Policy>
__global__
__launch_bounds__(256, 1)
void gridScatterKernelNative(GridParallelScatterNative<W, Types, Policy> self)
{
    using RealType        = typename Types::RealType;
    using ValueType       = typename Types::ValueType;
    using grid_value_t    = typename decltype(self.args.grid)::non_const_value_type;
    constexpr unsigned Dim   = Types::Dim;
    constexpr int half_left  = (W + 1) / 2;
    constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
    constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

    // ── Thread identity — zero-cost hardware registers ────────────────────────
    const int    lane     = static_cast<int>(threadIdx.x);   // 0..31
    const int    vec_id   = static_cast<int>(threadIdx.y);   // warp index 0..nv-1
    const int    nv       = static_cast<int>(blockDim.y);    // warps per block
    const size_t league_r = static_cast<size_t>(blockIdx.x);

    const size_t tile_id = league_r / self.sub_teams_per_tile_;
    const size_t sub_id  = league_r % self.sub_teams_per_tile_;

    // ── Bin bounds (read-only → __ldg) ───────────────────────────────────────
    const size_t bin_start = __ldg(&self.args.bin_offsets.data()[tile_id]);
    const size_t bin_end   = __ldg(&self.args.bin_offsets.data()[tile_id + 1]);
    const size_t bin_size  = bin_end - bin_start;

    const size_t ppt    = (bin_size + self.sub_teams_per_tile_ - 1) / self.sub_teams_per_tile_;
    const size_t pstart = bin_start + sub_id * ppt;
    if (pstart >= bin_end) return;
    const size_t pend = Kokkos::min(bin_end, pstart + ppt);

    // ── Tile base coordinates (decode once in registers) ─────────────────────
    int tile_base[Dim];
    {
        size_t t = tile_id;
        for (int d = static_cast<int>(Dim) - 1; d >= 0; --d) {
            tile_base[d] = static_cast<int>(t % static_cast<size_t>(self.args.num_tiles[d]))
                           * self.args.tile_size[d];
            t /= static_cast<size_t>(self.args.num_tiles[d]);
        }
    }

    // Precompute histogram strides into registers (avoids recomputing each particle)
    const size_t sx   = self.hs0_stride_;
    const size_t hs1  = (Dim >= 2) ? static_cast<size_t>(self.hist_ext(1)) : size_t(1);
    const size_t sxy  = (Dim >= 3) ? sx * hs1 : size_t(0);
    const size_t htot = self.hist_stride_;

    // ── Dynamic shared memory layout ─────────────────────────────────────────
    //
    //  We carve the raw byte buffer manually so that each array starts on a
    //  RealType-aligned boundary.  Order:
    //    [A] hist_r : nv * htot  RealType
    //    [B] hist_i : nv * htot  RealType   (complex only; zero-sized otherwise)
    //    [C] kw     : nv * Dim * W  RealType
    //    [D] gp_s   : nv * Dim      RealType
    //    [E] base_s : nv * Dim      int
    //
    extern __shared__ char smem_raw[];
    char* smem_ptr = smem_raw;

    auto carve_real = [&](size_t count) -> RealType* {
        // Align to RealType boundary
        const size_t align = alignof(RealType);
        uintptr_t p = reinterpret_cast<uintptr_t>(smem_ptr);
        p = (p + align - 1u) & ~(align - 1u);
        smem_ptr = reinterpret_cast<char*>(p) + count * sizeof(RealType);
        return reinterpret_cast<RealType*>(p);
    };
    auto carve_int = [&](size_t count) -> int* {
        const size_t align = alignof(int);
        uintptr_t p = reinterpret_cast<uintptr_t>(smem_ptr);
        p = (p + align - 1u) & ~(align - 1u);
        smem_ptr = reinterpret_cast<char*>(p) + count * sizeof(int);
        return reinterpret_cast<int*>(p);
    };

    RealType* hist_r = carve_real(static_cast<size_t>(nv) * htot);
    RealType* hist_i = gcplx ? carve_real(static_cast<size_t>(nv) * htot) : nullptr;
    RealType* kw_all = carve_real(static_cast<size_t>(nv) * Dim * W);
    RealType* gp_all = carve_real(static_cast<size_t>(nv) * Dim);
    int*      bs_all = carve_int (static_cast<size_t>(nv) * Dim);

    // Per-warp slice pointers
    RealType* my_hist_r = hist_r + static_cast<size_t>(vec_id) * htot;
    RealType* my_hist_i = gcplx  ? hist_i + static_cast<size_t>(vec_id) * htot : nullptr;
    RealType* my_kw     = kw_all + static_cast<size_t>(vec_id) * Dim * W;
    RealType* my_gp     = gp_all + static_cast<size_t>(vec_id) * Dim;
    int*      my_base   = bs_all + static_cast<size_t>(vec_id) * Dim;

    // ── Zero this warp's histogram slice ─────────────────────────────────────
    // Each lane zeros a stride-32 subset; all nv warps run simultaneously.
    for (size_t i = static_cast<size_t>(lane); i < htot; i += 32u) {
        my_hist_r[i] = RealType(0);
        if constexpr (gcplx) my_hist_i[i] = RealType(0);
    }
    __syncthreads();   // all warps done zeroing before any scatter

    const CoordinateTransform<RealType, Dim> transform{
        self.args.origin, self.args.invdx, self.args.n_grid};

    // ── Particle loop ─────────────────────────────────────────────────────────
    // Round-robin: warp vec_id handles particles pstart+vec_id, ..., step nv.
    for (size_t ip = pstart + static_cast<size_t>(vec_id); ip < pend;
         ip += static_cast<size_t>(nv)) {

        const size_t p = __ldg(&self.args.permute.data()[ip]);

        // Particle value — same for all 32 lanes (broadcast from L1)
        RealType val_r = RealType(0), val_i = RealType(0);
        if constexpr (vcplx) {
            val_r = self.args.values(p).real();
            if constexpr (gcplx) val_i = self.args.values(p).imag();
        } else {
            val_r = static_cast<RealType>(self.args.values(p));
        }

        // ── Step 1a: grid coords + stencil base, ONE per dimension ────────────
        // Only Dim lanes (0..Dim-1) are active; no wasted work per dimension.
        // Saves (W-1)*Dim coordinate-transform calls vs. original DimW loop.
        if (lane < static_cast<int>(Dim)) {
            const RealType gp = transform.toGridCoordinate(self.args.x(p)[lane], lane);
            my_gp[lane]   = gp;
            my_base[lane] = transform.getStencilBase(gp - RealType(0.5), W)
                            - self.args.local_offset[lane];
        }
        // Ensure my_gp and my_base are visible to all 32 lanes before step 1b.
        __syncwarp(0xFFFFFFFFu);

        // ── Step 1b: kernel weights (Dim*W entries) ───────────────────────────
        // All lanes 0..Dim*W-1 active; remainder idle.
        if (lane < static_cast<int>(Dim) * W) {
            const int      d    = lane / W;
            const int      i    = lane % W;
            const int      idx0 = my_base[d] + self.args.local_offset[d];
            my_kw[lane] = self.args.kernel(
                (my_gp[d] - (RealType(idx0 + i) + RealType(0.5))) * self.args.inv_hw);
        }
        // Ensure my_kw is visible to all 32 lanes before step 2.
        __syncwarp(0xFFFFFFFFu);

        // ── Step 2: scatter to private histogram ──────────────────────────────
        // Each lane handles flat = lane, lane+32, lane+64, …
        // flat → hidx is injective (shown in header), so no intra-warp races.
        // No __syncwarp needed between scatter iterations.
        if constexpr (Dim == 1) {
            const int bh0 = my_base[0] + half_left - tile_base[0];
            #pragma unroll
            for (int i0 = lane; i0 < W; i0 += 32) {
                const RealType w = my_kw[i0];
                my_hist_r[static_cast<size_t>(bh0 + i0)] += val_r * w;
                if constexpr (gcplx)
                    my_hist_i[static_cast<size_t>(bh0 + i0)] += val_i * w;
            }

        } else if constexpr (Dim == 2) {
            const int bh0 = my_base[0] + half_left - tile_base[0];
            const int bh1 = my_base[1] + half_left - tile_base[1];
            #pragma unroll
            for (int flat = lane; flat < W * W; flat += 32) {
                const int      i0   = imod<W>(flat);
                const int      i1   = idiv<W>(flat);
                const size_t   hidx = static_cast<size_t>(bh0 + i0)
                                    + sx * static_cast<size_t>(bh1 + i1);
                const RealType w    = my_kw[i0] * my_kw[W + i1];
                my_hist_r[hidx] += val_r * w;
                if constexpr (gcplx) my_hist_i[hidx] += val_i * w;
            }

        } else if constexpr (Dim == 3) {
            const int bh0    = my_base[0] + half_left - tile_base[0];
            const int bh1    = my_base[1] + half_left - tile_base[1];
            const int bh2    = my_base[2] + half_left - tile_base[2];
            const int zrange = self.z_end_ - self.z_start_;
            const int total  = W * W * zrange;
            #pragma unroll
            for (int flat = lane; flat < total; flat += 32) {
                const int      i0   = imod<W>(flat);
                const int      i1   = imod<W>(idiv<W>(flat));
                const int      i2   = self.z_start_ + flat / (W * W);
                const RealType w    = my_kw[i0] * my_kw[W + i1] * my_kw[2 * W + i2];
                const size_t   hidx = static_cast<size_t>(bh0 + i0)
                                    + sx * (static_cast<size_t>(bh1 + i1)
                                            + hs1 * static_cast<size_t>(bh2 + i2 - self.z_start_));
                my_hist_r[hidx] += val_r * w;
                if constexpr (gcplx) my_hist_i[hidx] += val_i * w;
            }
        }
        // No __syncwarp needed: next particle iteration reads my_gp/my_base fresh.
    }  // particle loop

    __syncthreads();   // all warps done scattering before reduction

    // ── Reduction: merge nv histogram copies → global grid ────────────────────
    //
    // We distribute htot entries among all (nv × 32) threads in the block.
    // Warp `w` handles entries: w*32, w*32+1, ..., (w+32-1); then w*32+nv*32, etc.
    // = warp w handles chunks c = w, w+nv, w+2*nv, ...  (each chunk = 32 entries)
    //
    // Bank analysis: within one v, 32 lanes read 32 consecutive addresses →
    //   32 distinct banks.  The sequential v-loop adds no concurrency, so there
    //   are no cross-v bank conflicts regardless of stride.  (See header.)
    const size_t chunks = (htot + 31u) / 32u;

    for (size_t chunk = static_cast<size_t>(vec_id); chunk < chunks;
         chunk += static_cast<size_t>(nv)) {
        const size_t idx = chunk * 32u + static_cast<size_t>(lane);
        if (idx >= htot) continue;

        // Decode: dimension 0 uses hs0_stride_ (bank-padded); skip padding entries.
        {
            size_t tmp = idx;
            const int hc0 = static_cast<int>(tmp % self.hs0_stride_);
            if (hc0 >= self.hist_ext(0)) continue;   // padding slot
            tmp /= self.hs0_stride_;
            int hc[Dim];
            hc[0] = hc0;
            if constexpr (Dim >= 2) {
                hc[1] = static_cast<int>(tmp % static_cast<size_t>(self.hist_ext(1)));
                tmp   /= static_cast<size_t>(self.hist_ext(1));
            }
            if constexpr (Dim >= 3) {
                hc[2] = static_cast<int>(tmp);
            }

            // Sum across all nv vector copies
            RealType sum_r = RealType(0), sum_i = RealType(0);
            for (int v = 0; v < nv; ++v) {
                sum_r += hist_r[static_cast<size_t>(v) * htot + idx];
                if constexpr (gcplx)
                    sum_i += hist_i[static_cast<size_t>(v) * htot + idx];
            }

            // Map histogram coords → local grid coords; skip out-of-bounds
            int gc[Dim];
            bool oob = false;
            for (unsigned d = 0; d < Dim; ++d) {
                int hc_d = hc[d];
                if constexpr (Dim == 3) { if (d == 2) hc_d += self.z_start_; }
                const int local = tile_base[d] + hc_d - half_left;
                if (local < -self.args.nghost
                    || local >= self.args.n_grid_local[d] + self.args.nghost) {
                    oob = true; break;
                }
                gc[d] = local + self.args.nghost;
            }
            if (oob) continue;

            // Single atomic add per entry
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                if constexpr (gcplx) {
                    RealType* ptr = reinterpret_cast<RealType*>(&self.args.grid(gc[Is]...));
                    Kokkos::atomic_add(&ptr[0], sum_r);
                    Kokkos::atomic_add(&ptr[1], sum_i);
                } else {
                    Kokkos::atomic_add(&self.args.grid(gc[Is]...),
                                       static_cast<grid_value_t>(sum_r));
                }
            }(std::make_index_sequence<Dim>{});
        }
    }  // reduction
}  // gridScatterKernelNative

// ─── GridParallelScatterNative struct ────────────────────────────────────────

template <int W, class Types, class Policy>
struct GridParallelScatterNative {
    static_assert(Policy::use_sorting,
                  "GridParallelScatterNative assumes sorted/bin-partitioned particles");

    static constexpr bool     requires_binning = true;
    static constexpr unsigned Dim              = Types::Dim;
    static constexpr int      vector_length    = 32;

    using RealType        = typename Types::RealType;
    using ValueType       = typename Types::ValueType;
    using memory_space    = typename Types::memory_space;
    using execution_space = typename Types::execution_space;

    // Re-use the same Arguments type from the Kokkos version
    using Arguments = typename GridParallelScatter<W, Types, Policy>::Arguments;
    Arguments args;

    // State set in run(), read inside the kernel
    size_t sub_teams_per_tile_ = 1;
    size_t hist_stride_        = 1;
    size_t hs0_stride_         = 1;
    int    z_batch_size_ = W;
    int    z_start_      = 0;
    int    z_end_        = W;

    // Must be accessible from the __global__ kernel (host+device)
    KOKKOS_INLINE_FUNCTION int hist_ext(unsigned d) const noexcept {
        if (Dim == 3 && d == 2) return args.tile_size[d] + z_batch_size_ + 1;
        return args.tile_size[d] + W + 1;
    }

    // Scratch size query (identical logic to the Kokkos version)
    template <bool IsComplex>
    static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int team_size,
                                       int z_batches = 1) {
        return GridParallelScatter<W, Types, Policy>
                   ::template compute_scratch_size<IsComplex>(tile_size, team_size, z_batches);
    }

    void run(size_t n_particles) {
        using grid_value_t  = typename decltype(args.grid)::non_const_value_type;
        constexpr bool cplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

        size_t n_tiles = 1;
        for (unsigned d = 0; d < Dim; ++d)
            n_tiles *= static_cast<size_t>(args.num_tiles[d]);
        if (n_tiles == 0 || n_particles == 0) return;

        sub_teams_per_tile_ = std::max(size_t(1), size_t(args.oversubscription_factor));

        const int z_batches = (Dim == 3) ? std::max(1, args.z_batches) : 1;
        z_batch_size_       = (W + z_batches - 1) / z_batches;

        hs0_stride_ = bank_pad(static_cast<size_t>(args.tile_size[0] + W + 1));

        size_t htot = hs0_stride_;
        for (unsigned d = 1; d < Dim; ++d) {
            const int ext = (Dim == 3 && d == 2)
                                ? args.tile_size[d] + z_batch_size_ + 1
                                : args.tile_size[d] + W + 1;
            htot *= static_cast<size_t>(ext);
        }
        hist_stride_ = htot;

        // Scratch size (bytes for dynamic __shared__ allocation)
        const size_t hist_bytes = (cplx ? 2u : 1u) * htot * sizeof(RealType);
        const size_t kw_bytes   = static_cast<size_t>(args.team_size) * Dim * W * sizeof(RealType);
        const size_t gp_bytes   = static_cast<size_t>(args.team_size) * Dim * sizeof(RealType);
        const size_t bs_bytes   = static_cast<size_t>(args.team_size) * Dim * sizeof(int);
        // Alignment padding: at most (alignof(RealType)-1) bytes per array boundary
        constexpr size_t align_pad = alignof(RealType) - 1u;
        const size_t smem_bytes = static_cast<size_t>(args.team_size)
                                  * (hist_bytes + 3u * align_pad)
                                + kw_bytes + gp_bytes
                                + bs_bytes + 2u * align_pad;

        const dim3 block(static_cast<unsigned>(vector_length),
                         static_cast<unsigned>(args.team_size), 1u);

        for (int batch = 0; batch < z_batches; ++batch) {
            z_start_ = batch * z_batch_size_;
            z_end_   = std::min((batch + 1) * z_batch_size_, W);

            const dim3 grid(static_cast<unsigned>(n_tiles * sub_teams_per_tile_), 1u, 1u);

#if defined(IPPL_GPU_BACKEND_CUDA)
            // Obtain the Kokkos CUDA stream so we stay in the same execution context
            auto space = Kokkos::DefaultExecutionSpace();
            cudaStream_t stream = space.cuda_stream();
            gridScatterKernelNative<W, Types, Policy>
                <<<grid, block, smem_bytes, stream>>>(*this);
#elif defined(IPPL_GPU_BACKEND_HIP)
            auto& space = Kokkos::DefaultExecutionSpace();
            hipStream_t stream = space.hip_stream();
            hipLaunchKernelGGL((gridScatterKernelNative<W, Types, Policy>),
                               grid, block, smem_bytes, stream, *this);
#endif
        }
        // Execution is asynchronous; caller is responsible for fencing if needed.
    }
};  // GridParallelScatterNative (GPU)

#else  // !IPPL_HAS_GPU  ── CPU-only build ─────────────────────────────────────

// Provide the same name so call-sites compile without #ifdefs.
// On CPU the Kokkos version is used directly.
template <int W, class Types, class Policy>
using GridParallelScatterNative = GridParallelScatter<W, Types, Policy>;

#endif  // IPPL_HAS_GPU

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_GRID_PARALLEL_SCATTER_H