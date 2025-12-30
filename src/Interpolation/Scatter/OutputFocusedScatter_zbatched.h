#ifndef IPPL_OUTPUTFOCUSED_SCATTER_ZLOOP_H
#define IPPL_OUTPUTFOCUSED_SCATTER_ZLOOP_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>

#include "../InterpolationTypes.h"

#include "../CoordinateTransform.h"
#include "../InterpolationUtil.h"
#include "../KernelEvaluator.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {
            /**
             * @brief Scatter functor with z-batch looping within a single team
             *
             * This version reduces shared memory requirements by processing the z-dimension
             * of the kernel stencil in batches. Unlike the multi-team z-strided version,
             * this uses a single team per tile and loops over z-batches internally,
             * recomputing z-kernel values and flushing the histogram for each batch.
             *
             * Memory reduction: histogram z-size is (tile_size_z + z_batch_size) instead of
             *                   (tile_size_z + W)
             *
             * Template parameters:
             * @tparam W The kernel width (compile-time constant for optimization)
             * @tparam ZBatchSize The z-batch size (compile-time constant, should divide W evenly for best performance)
             * @tparam RealType The floating point type (float or double)
             * @tparam ExecSpace The Kokkos execution space
             * @tparam KernelType The kernel function type
             * @tparam ValueType The type of values being scattered
             * @tparam GridViewType The type of the grid view
             * @tparam BinOffsetsType The type of bin offsets view
             * @tparam PermuteType The type of permutation view
             * @tparam PositionViewType The type of the position view
             */
            template <int W, int ZBatchSize, typename RealType, typename ExecSpace,
                      typename KernelType, typename ValueType, typename GridViewType,
                      typename BinOffsetsType, typename PermuteType,
                      typename PositionViewType =
                          Kokkos::View<RealType* [3], typename ExecSpace::memory_space>>
            struct OutputFocusedScatterFunctor3DZLoop {
                using real_type        = RealType;
                using value_type       = ValueType;
                using memory_space     = typename ExecSpace::memory_space;
                using size_type        = int;
                using team_policy      = Kokkos::TeamPolicy<ExecSpace>;
                using team_member      = typename team_policy::member_type;
                using scratch_space    = typename ExecSpace::scratch_memory_space;
                using shared_real_view = Kokkos::View<real_type*, scratch_space,
                                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

                // Input data
                std::decay_t<BinOffsetsType> bin_offsets;
                std::decay_t<PermuteType> permute;
                std::decay_t<PositionViewType> x;                // Particle positions in PHYSICAL coordinates
                Kokkos::View<value_type*, memory_space> values;  // Values to scatter
                std::decay_t<GridViewType> grid;                 // Output grid

                // Parameters
                Kokkos::Array<int, 3> n_grid;        // GLOBAL grid dimensions
                Kokkos::Array<int, 3> n_grid_local;  // LOCAL grid dimensions
                Kokkos::Array<int, 3> local_offset;  // First global index of local domain
                Kokkos::Array<int, 3> num_tiles;
                int tile_size_x, tile_size_y, tile_size_z;
                int nghost;        // ghost cell offset for field
                real_type inv_hw;  // 1 / half_width for kernel scaling
                std::decay_t<KernelType> kernel;

                // Mesh information for coordinate transformation
                Kokkos::Array<real_type, 3> origin_;   // Physical origin from mesh
                Kokkos::Array<real_type, 3> invdx_;    // Inverse mesh spacing (1/dx)

                // Compile-time constants
                static constexpr int w              = W;
                static constexpr int z_batch_size   = ZBatchSize;
                static constexpr int num_z_batches  = (W + ZBatchSize - 1) / ZBatchSize;
                static constexpr int hw             = W / 2;
                static constexpr bool odd           = (W & 1);
                static constexpr int half_left      = (W - 1) / 2;

                // Helper to access position component
                template <typename PosView>
                KOKKOS_INLINE_FUNCTION static auto get_component(const PosView& pos, size_type i,
                                                                 int d) -> decltype(pos(i, d)) {
                    return pos(i, d);
                }

                template <typename PosView>
                KOKKOS_INLINE_FUNCTION static auto get_component(const PosView& pos, size_type i,
                                                                 int d) -> decltype(pos(i)[d]) {
                    return pos(i)[d];
                }

                // Histogram size calculations - z dimension is reduced!
                KOKKOS_INLINE_FUNCTION constexpr int hist_size_x() const { return tile_size_x + W; }
                KOKKOS_INLINE_FUNCTION constexpr int hist_size_y() const { return tile_size_y + W; }
                KOKKOS_INLINE_FUNCTION constexpr int hist_size_z() const { return tile_size_z + z_batch_size; }

                KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
                    const int team_id = team.league_rank();

                    const size_type tile_linear = team_id;

                    const int tile_x = tile_linear % num_tiles[0];
                    const int tile_y = (tile_linear / num_tiles[0]) % num_tiles[1];
                    const int tile_z = tile_linear / (num_tiles[0] * num_tiles[1]);

                    // Tile bounds in local grid coordinates
                    const int tile_x0 = tile_x * tile_size_x;
                    const int tile_y0 = tile_y * tile_size_y;
                    const int tile_z0 = tile_z * tile_size_z;

                    const int hx = hist_size_x();
                    const int hy = hist_size_y();
                    const int hz = hist_size_z();

                    // Type traits for grid
                    using grid_element_type = std::remove_reference_t<decltype(grid(0, 0, 0))>;
                    constexpr bool value_is_complex =
                        std::is_same_v<value_type, Kokkos::complex<real_type>>;
                    constexpr bool grid_is_complex =
                        std::is_same_v<grid_element_type, Kokkos::complex<real_type>>;

                    const size_t hist_total = hx * hy * hz;

                    // Allocate shared memory for histogram (reduced z-size!)
                    shared_real_view hist_r(team.team_scratch(0), hist_total);
                    shared_real_view hist_c;
                    if constexpr (grid_is_complex) {
                        hist_c = shared_real_view(team.team_scratch(0), hist_total);
                    }

                    // Allocate shared memory for kernel values
                    // We store: W values for x, W values for y, z_batch_size values for current z-batch
                    shared_real_view kernel_vals(team.team_scratch(0), 2 * W + z_batch_size);

                    // Get particles in this tile
                    const size_type pstart = bin_offsets(tile_linear);
                    const size_type pend   = bin_offsets(tile_linear + 1);

                    // Loop over z-batches
                    for (int z_batch = 0; z_batch < num_z_batches; ++z_batch) {
                        const int z_offset = z_batch * z_batch_size;
                        const int z_count = (z_batch == num_z_batches - 1)
                                                ? (W - z_batch * z_batch_size)
                                                : z_batch_size;

                        // Initialize histogram to zero for this z-batch
                        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hist_total),
                                             [&](int i) {
                                                 hist_r(i) = 0;
                                                 if constexpr (grid_is_complex) {
                                                     hist_c(i) = 0;
                                                 }
                                             });
                        team.team_barrier();

                        // Process all particles for this z-batch
                        for (int i = pstart; i < pend; ++i) {
                            const size_type j    = permute(i);
                            const value_type val = values(j);

                            // Create coordinate transform with mesh information
                            ippl::Vector<real_type, 3> origin_vec, invdx_vec;
                            ippl::Vector<int, 3> n_grid_vec;
                            for (unsigned d = 0; d < 3; ++d) {
                                origin_vec[d] = origin_[d];
                                invdx_vec[d] = invdx_[d];
                                n_grid_vec[d] = n_grid[d];
                            }
                            CoordinateTransform<real_type, 3> transform(origin_vec, invdx_vec, n_grid_vec);

                            real_type grid_pos[3];
                            int stencil_base[3];

                            for (int d = 0; d < 3; ++d) {
                                // Transform from physical coordinates to grid coordinates
                                grid_pos[d] = transform.toGridCoordinate(get_component(x, j, d), d);
                                stencil_base[d] = transform.getStencilBase(grid_pos[d], W);
                            }

                            // Precompute kernel values using KernelEvaluator
                            KernelEvaluator<W, 3, std::decay_t<KernelType>, real_type> keval;
                            keval.evaluate(grid_pos, stencil_base, inv_hw, kernel);

                            // Copy kernel values to shared memory for team-parallel access
                            // Store: W values for x, W values for y, z_count values for current z-batch
                            Kokkos::parallel_for(
                                Kokkos::TeamThreadRange(team, 2 * W + z_count),
                                [&](int flat_w) {
                                    if (flat_w < W) {
                                        kernel_vals(flat_w) = keval.getValue(0, flat_w);
                                    } else if (flat_w < 2 * W) {
                                        kernel_vals(flat_w) = keval.getValue(1, flat_w - W);
                                    } else {
                                        // For z-batch, we need to offset into the z values
                                        int z_idx = flat_w - 2 * W + z_offset;
                                        kernel_vals(flat_w) = keval.getValue(2, z_idx);
                                    }
                                });
                            team.team_barrier();

                            // Convert stencil_base from global to local coordinates
                            const int local_stencil_x = stencil_base[0] - local_offset[0];
                            const int local_stencil_y = stencil_base[1] - local_offset[1];
                            const int local_stencil_z = stencil_base[2] - local_offset[2];

                            // Scatter to local histogram
                            const int point_tile_x = local_stencil_x - tile_x0;
                            const int point_tile_y = local_stencil_y - tile_y0;
                            const int point_tile_z = local_stencil_z - tile_z0;

                            Kokkos::parallel_for(
                                Kokkos::TeamThreadMDRange(team, W, W, z_count),
                                [&](int wx, int wy, int wz) {
                                    const real_type kernel_val =
                                        kernel_vals(wx) * kernel_vals(W + wy)
                                        * kernel_vals(2 * W + wz);

                                    // Note: wz is local to this batch (0 to z_count-1)
                                    // The histogram z-index is point_tile_z + wz
                                    const int hist_idx =
                                        (((point_tile_z + wz) * hy + (point_tile_y + wy)) * hx
                                         + (point_tile_x + wx));

                                    if constexpr (value_is_complex && grid_is_complex) {
                                        hist_r(hist_idx) += val.real() * kernel_val;
                                        hist_c(hist_idx) += val.imag() * kernel_val;
                                    } else if constexpr (!value_is_complex && grid_is_complex) {
                                        hist_r(hist_idx) += val * kernel_val;
                                    } else {
                                        hist_r(hist_idx) += val * kernel_val;
                                    }
                                });
                            team.team_barrier();
                        }

                        // Flush histogram to global grid for this z-batch
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, hist_total), [&](int hist_idx) {
                                // Skip if histogram value is zero (common case)
                                bool has_contribution = (hist_r(hist_idx) != 0);
                                if constexpr (grid_is_complex) {
                                    has_contribution = has_contribution || (hist_c(hist_idx) != 0);
                                }
                                if (!has_contribution) return;

                                int hist_x = hist_idx % hx;
                                int hist_y = (hist_idx / hx) % hy;
                                int hist_z = hist_idx / (hx * hy);

                                // Compute LOCAL grid indices
                                // Note: z_offset is added here for the actual grid position
                                int local_x = tile_x0 + hist_x - half_left;
                                int local_y = tile_y0 + hist_y - half_left;
                                int local_z = tile_z0 + hist_z - half_left + z_offset;

                                // Bounds check (including ghosts)
                                if (local_x < -nghost
                                    || local_x >= static_cast<int>(n_grid_local[0]) + nghost
                                    || local_y < -nghost
                                    || local_y >= static_cast<int>(n_grid_local[1]) + nghost
                                    || local_z < -nghost
                                    || local_z >= static_cast<int>(n_grid_local[2]) + nghost) {
                                    return;
                                }

                                // Atomic add to global grid
                                if constexpr (grid_is_complex) {
                                    double* addr_as_double = reinterpret_cast<double*>(
                                        &grid(local_x + nghost, local_y + nghost, local_z + nghost));
                                    Kokkos::atomic_add(&addr_as_double[0], hist_r(hist_idx));
                                    Kokkos::atomic_add(&addr_as_double[1], hist_c(hist_idx));
                                } else {
                                    Kokkos::atomic_add(
                                        &grid(local_x + nghost, local_y + nghost, local_z + nghost),
                                        hist_r(hist_idx));
                                }
                            });
                        team.team_barrier();  // Ensure flush is complete before next batch
                    }  // end z-batch loop
                }
            };

            /**
             * @brief Dispatcher for z-loop scatter with different kernel widths and batch sizes
             */
            template <int W, int ZBatchSize, int MaxW>
            struct OutputFocusedScatterZLoopDispatcher {
                template <typename RealType, typename ExecSpace, typename KernelType,
                          typename ValueType, typename GridViewType, typename PositionViewType,
                          typename PermuteViewType, typename BinOffsetsViewType>
                static void dispatch_3d(
                    int w, int z_batch_size, BinOffsetsViewType bin_offsets,
                    PermuteViewType permute, PositionViewType x,
                    Kokkos::View<ValueType*, typename ExecSpace::memory_space> values,
                    GridViewType grid, Kokkos::Array<int, 3> n_grid,
                    Kokkos::Array<int, 3> n_grid_local, Kokkos::Array<int, 3> local_offset,
                    Kokkos::Array<int, 3> num_tiles, int tile_size_x, int tile_size_y,
                    int tile_size_z, int nghost, RealType inv_hw, const KernelType& kernel,
                    int team_size,
                    Kokkos::Array<RealType, 3> origin,
                    Kokkos::Array<RealType, 3> invdx) {

                    if constexpr (W <= MaxW) {
                        if (w == W && z_batch_size == ZBatchSize) {
                            // Matched both W and ZBatchSize
                            OutputFocusedScatterFunctor3DZLoop<
                                W, ZBatchSize, RealType, ExecSpace, KernelType, ValueType,
                                GridViewType, decltype(bin_offsets), decltype(permute),
                                PositionViewType>
                                functor{bin_offsets,  permute,      x,
                                        values,       grid,         n_grid,
                                        n_grid_local, local_offset, num_tiles,
                                        tile_size_x,  tile_size_y,  tile_size_z,
                                        nghost,       inv_hw,       kernel,
                                        origin,       invdx};

                            // Calculate scratch memory size (reduced due to z-batching!)
                            const size_t hist_size = functor.hist_size_x() * functor.hist_size_y()
                                                     * functor.hist_size_z();
                            using grid_element_type =
                                std::remove_reference_t<decltype(grid(0, 0, 0))>;
                            constexpr bool is_complex =
                                std::is_same_v<grid_element_type, Kokkos::complex<RealType>>;
                            const size_t scratch_size =
                                is_complex ? (2 * hist_size + (2 * W + ZBatchSize))
                                           : (hist_size + (2 * W + ZBatchSize));
                            const size_t scratch_bytes = scratch_size * sizeof(RealType);

                            // One team per tile (no z-splitting across teams)
                            const int n_tiles_total = num_tiles[0] * num_tiles[1] * num_tiles[2];

                            using team_policy = Kokkos::TeamPolicy<ExecSpace>;
                            team_policy policy(n_tiles_total, team_size);
                            policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

                            Kokkos::fence();
                            Kokkos::parallel_for("output_focused_scatter_zloop", policy, functor);
                            Kokkos::fence();
                        } else if (w == W) {
                            // W matched, try next ZBatchSize
                            // For simplicity, we support ZBatchSize = 1, 2, 4, 8
                            if constexpr (ZBatchSize < W) {
                                OutputFocusedScatterZLoopDispatcher<
                                    W, (ZBatchSize * 2 > W ? W : ZBatchSize * 2),
                                    MaxW>::template dispatch_3d<RealType, ExecSpace, KernelType,
                                                                ValueType, GridViewType,
                                                                PositionViewType, PermuteViewType,
                                                                BinOffsetsViewType>(
                                    w, z_batch_size, bin_offsets, permute, x, values, grid, n_grid,
                                    n_grid_local, local_offset, num_tiles, tile_size_x, tile_size_y,
                                    tile_size_z, nghost, inv_hw, kernel, team_size, origin, invdx);
                            } else {
                                throw std::runtime_error(
                                    "Z batch size not supported for this kernel width");
                            }
                        } else {
                            // Try next W
                            OutputFocusedScatterZLoopDispatcher<W + 1, 1, MaxW>::template dispatch_3d<
                                RealType, ExecSpace, KernelType, ValueType, GridViewType,
                                PositionViewType, PermuteViewType, BinOffsetsViewType>(
                                w, z_batch_size, bin_offsets, permute, x, values, grid, n_grid,
                                n_grid_local, local_offset, num_tiles, tile_size_x, tile_size_y,
                                tile_size_z, nghost, inv_hw, kernel, team_size, origin, invdx);
                        }
                    } else {
                        throw std::runtime_error("Kernel width exceeds maximum supported width");
                    }
                }
            };

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_OUTPUTFOCUSED_SCATTER_ZLOOP_H