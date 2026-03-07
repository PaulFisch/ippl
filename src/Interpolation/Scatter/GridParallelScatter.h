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
//  GridParallelScatter  –  vectorized shared-memory histogramming
// ============================================================================
//
// Algorithm overview
// ------------------
// The team is composed of `team_size` Kokkos "threads" (warps), each of which
// maps to one GPU warp of `vector_length` (= 32) CUDA threads.
//
//   • Every vector owns a private slice of the shared-memory histogram:
//       hist_r[ vec_id * htot + local_entry ]
//     Only one vector writes to its own slice during the particle loop,
//     so no atomics are needed at this stage.
//
//   • Particles in the bin are distributed across vectors in round-robin
//     fashion. For each particle, the vector's 32 lanes cooperate via
//     ThreadVectorRange to:
//       1. Compute Dim*W kernel weights in parallel.
//       2. Scatter W^Dim stencil contributions into the private histogram.
//
//   • After a team barrier, the reduction phase merges all per-vector copies:
//     a nested (TeamThreadRange × ThreadVectorRange) loop sums each histogram
//     entry across all vectors and issues a single atomic add to global memory.
//
// ============================================================================

namespace ippl::Interpolation::detail {

    template <int W, class Types, class Policy>
    struct GridParallelScatter {
        static_assert(Policy::use_sorting,
                      "GridParallelScatter assumes sorted/bin-partitioned particles");

        static constexpr bool     requires_binning = true;
        static constexpr unsigned Dim              = Types::Dim;
        static constexpr int      half_left        = (W + 1) / 2;

        static constexpr int vector_length = 32;

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

        // ── Scratch layout (per team) ────────────────────────────────────────
        //
        //  [A]  nv * htot   hist_r  (real part)
        //  [B]  nv * htot   hist_i  (imag, complex only)
        //  [C]  nv * Dim*W  kw      (kernel weights per vector)
        //  [D]  nv * Dim    base_s  (stencil bases per vector)
        //
        //  When z_batches > 1, the z-dimension of the histogram is reduced from
        //  (tile_z + W + 1) to (tile_z + z_batch_size + 1).
        // ────────────────────────────────────────────────────────────────────
        template <bool IsComplex>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int team_size,
                                           int z_batches = 1) {
            const int z_batch_size = (W + z_batches - 1) / z_batches;

            size_t htot = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                const int hist_dim = (Dim == 3 && d == 2 && z_batches > 1)
                                         ? tile_size[d] + z_batch_size + 1
                                         : tile_size[d] + W + 1;
                htot *= static_cast<size_t>(hist_dim);
            }

            const int nv = std::max(1, team_size);

            return (IsComplex ? 2 : 1) * scratch_real_view::shmem_size(nv * htot)
                   + scratch_real_view::shmem_size(nv * static_cast<int>(Dim) * W)
                   + scratch_int_view::shmem_size(nv * static_cast<int>(Dim));
        }

        // ── Arguments ───────────────────────────────────────────────────────
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

        size_t sub_teams_per_tile_ = 1;
        size_t hist_total_         = 1;

        // Z-batching state
        int z_batch_size_ = W;
        int z_start_      = 0;
        int z_end_        = W;

        // ── Geometry helpers ─────────────────────────────────────────────────
        KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
            Vector<int, Dim> hs;
            for (unsigned d = 0; d < Dim; ++d)
                hs[d] = args.tile_size[d] + ((Dim == 3 && d == 2) ? z_batch_size_ : W) + 1;
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

        // ── Kernel operator ──────────────────────────────────────────────────
        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
            constexpr bool gcplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool vcplx = std::is_same_v<ValueType, Kokkos::complex<RealType>>;

            const int vec_id = team.team_rank();
            const int nv     = team.team_size();

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

            const auto   tile_base = decode_tile_base(tile_id);
            const auto   hs        = hist_size();
            const size_t htot      = hist_total_;

            // ── Scratch allocation ───────────────────────────────────────────
            scratch_real_view hist_r(team.team_scratch(0), nv * htot);
            scratch_real_view hist_i;
            if constexpr (gcplx)
                hist_i = scratch_real_view(team.team_scratch(0), nv * htot);

            scratch_real_view kw(team.team_scratch(0), nv * static_cast<int>(Dim) * W);
            scratch_int_view  base_s(team.team_scratch(0), nv * static_cast<int>(Dim));

            RealType* my_kw   = kw.data()    + vec_id * (static_cast<int>(Dim) * W);
            int*      my_base = base_s.data() + vec_id * static_cast<int>(Dim);

            // ── Zero this warp's histogram slice ─────────────────────────────
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, htot), [&](size_t i) {
                hist_r(vec_id * htot + i) = RealType(0);
                if constexpr (gcplx)
                    hist_i(vec_id * htot + i) = RealType(0);
            });
            team.team_barrier();

            // ── Particle loop ────────────────────────────────────────────────
            const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
                                                               args.n_grid};

            for (size_t ip = pstart + static_cast<size_t>(vec_id); ip < pend;
                 ip += static_cast<size_t>(nv)) {

                const size_t p = args.permute(ip);

                RealType val_r = RealType(0);
                RealType val_i = RealType(0);
                if constexpr (vcplx) {
                    val_r = args.values(p).real();
                    if constexpr (gcplx)
                        val_i = args.values(p).imag();
                } else {
                    val_r = static_cast<RealType>(args.values(p));
                }

                // ── Step 1: kernel weights ────────────────────────────────────
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, static_cast<int>(Dim) * W),
                    [&](int flat) {
                        const int      d    = flat / W;
                        const int      i    = flat % W;
                        const RealType gp   = transform.toGridCoordinate(args.x(p)[d], d);
                        const int      idx0 = transform.getStencilBase(gp - RealType(0.5), W);
                        my_kw[d * W + i] =
                            args.kernel((gp - (RealType(idx0 + i) + RealType(0.5))) * args.inv_hw);
                        if (i == 0)
                            my_base[d] = idx0 - args.local_offset[d];
                    });

                // ── Step 2: scatter to private histogram ──────────────────────
                RealType* h_r = hist_r.data() + vec_id * htot;
                RealType* h_i = gcplx ? hist_i.data() + vec_id * htot : nullptr;

                if constexpr (Dim == 1) {
                    const int bh0 = my_base[0] + half_left - tile_base[0];

                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W), [&](int i0) {
                        const size_t   hidx = static_cast<size_t>(bh0 + i0);
                        const RealType w    = my_kw[i0];
                        h_r[hidx] += val_r * w;
                        if constexpr (gcplx)
                            h_i[hidx] += val_i * w;
                    });

                } else if constexpr (Dim == 2) {
                    const int bh0 = my_base[0] + half_left - tile_base[0];
                    const int bh1 = my_base[1] + half_left - tile_base[1];

                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W * W),
                                         [&](int flat) {
                        const int      i0   = flat % W;
                        const int      i1   = flat / W;
                        const RealType w    = my_kw[i0] * my_kw[W + i1];
                        const size_t   hidx = static_cast<size_t>(bh0 + i0)
                                            + static_cast<size_t>(hs[0])
                                                  * static_cast<size_t>(bh1 + i1);
                        h_r[hidx] += val_r * w;
                        if constexpr (gcplx)
                            h_i[hidx] += val_i * w;
                    });

                } else if constexpr (Dim == 3) {
                    const int bh0      = my_base[0] + half_left - tile_base[0];
                    const int bh1      = my_base[1] + half_left - tile_base[1];
                    const int bh2      = my_base[2] + half_left - tile_base[2];
                    const int z_range  = z_end_ - z_start_;

                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W * W * z_range),
                                         [&](int flat) {
                        const int      i0   = flat % W;
                        const int      i1   = (flat / W) % W;
                        const int      i2   = z_start_ + flat / (W * W);
                        const RealType w    = my_kw[i0] * my_kw[W + i1] * my_kw[2 * W + i2];
                        const size_t   hidx =
                            static_cast<size_t>(bh0 + i0)
                            + static_cast<size_t>(hs[0])
                                  * (static_cast<size_t>(bh1 + i1)
                                     + static_cast<size_t>(hs[1])
                                           * static_cast<size_t>(bh2 + i2 - z_start_));
                        h_r[hidx] += val_r * w;
                        if constexpr (gcplx)
                            h_i[hidx] += val_i * w;
                    });
                }
            }

            team.team_barrier();

            // ── Reduction: merge nv histogram copies → global grid ────────────
            const size_t chunks = (htot + static_cast<size_t>(vector_length) - 1)
                                  / static_cast<size_t>(vector_length);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, chunks), [&](size_t chunk) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, vector_length), [&](int lane) {
                        const size_t idx = chunk * static_cast<size_t>(vector_length)
                                         + static_cast<size_t>(lane);
                        if (idx >= htot)
                            return;

                        RealType sum_r = RealType(0);
                        RealType sum_i = RealType(0);
                        for (int v = 0; v < nv; ++v) {
                            sum_r += hist_r(v * htot + idx);
                            if constexpr (gcplx)
                                sum_i += hist_i(v * htot + idx);
                        }

                        // Decode flat histogram index → per-dim histogram coordinates
                        size_t tmp = idx;
                        Kokkos::Array<int, Dim> hc{};
                        for (unsigned d = 0; d < Dim; ++d) {
                            hc[d] = static_cast<int>(tmp % static_cast<size_t>(hs[d]));
                            tmp /= static_cast<size_t>(hs[d]);
                        }

                        // Map histogram coords → local grid coords; skip ghost overflow
                        Kokkos::Array<int, Dim> gc{};
                        for (unsigned d = 0; d < Dim; ++d) {
                            int hc_adjusted = hc[d];
                            if constexpr (Dim == 3) {
                                if (d == 2)
                                    hc_adjusted += z_start_;
                            }
                            const int local = tile_base[d] + hc_adjusted - half_left;
                            if (local < -args.nghost
                                || local >= args.n_grid_local[d] + args.nghost)
                                return;
                            gc[d] = local + args.nghost;
                        }

                        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                            if constexpr (gcplx) {
                                RealType* ptr =
                                    reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                                Kokkos::atomic_add(&ptr[0], sum_r);
                                Kokkos::atomic_add(&ptr[1], sum_i);
                            } else {
                                Kokkos::atomic_add(&args.grid(gc[Is]...),
                                                   static_cast<grid_value_t>(sum_r));
                            }
                        }(std::make_index_sequence<Dim>{});
                    });
            });
        }

        // ── run() ────────────────────────────────────────────────────────────
        void run(size_t n_particles) {
            using grid_value_t  = typename decltype(args.grid)::non_const_value_type;
            constexpr bool cplx = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

            size_t n_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n_tiles *= static_cast<size_t>(args.num_tiles[d]);

            if (n_tiles == 0 || n_particles == 0)
                return;

            sub_teams_per_tile_ = std::max(size_t(1), size_t(args.oversubscription_factor));

            const int z_batches = (Dim == 3) ? std::max(1, args.z_batches) : 1;
            z_batch_size_       = (W + z_batches - 1) / z_batches;

            // Compute histogram total size
            hist_total_ = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                const int hist_dim = args.tile_size[d]
                                     + ((Dim == 3 && d == 2) ? z_batch_size_ : W) + 1;
                hist_total_ *= static_cast<size_t>(hist_dim);
            }

            const size_t scratch =
                compute_scratch_size<cplx>(args.tile_size, args.team_size, z_batches);

            for (int batch = 0; batch < z_batches; ++batch) {
                z_start_ = batch * z_batch_size_;
                z_end_   = std::min((batch + 1) * z_batch_size_, W);

                Kokkos::parallel_for(
                    "GridParallelScatterVectorized",
                    team_policy(n_tiles * sub_teams_per_tile_, args.team_size, vector_length)
                        .set_scratch_size(0, Kokkos::PerTeam(scratch)),
                    *this);
            }
        }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_GRID_PARALLEL_SCATTER_H