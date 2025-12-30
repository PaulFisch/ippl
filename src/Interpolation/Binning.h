#ifndef IPPL_INTERPOLATION_BINNING_H
#define IPPL_INTERPOLATION_BINNING_H

#include <Kokkos_Core.hpp>

#include "CoordinateTransform.h"
#include "InterpolationUtil.h"
#include "Particle/ParticleLayout.h"
#include "Particle/SortBuffer.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

            template <int Dim, typename MemorySpace>
            struct BinningResult {
                Kokkos::View<size_type*, MemorySpace> permute;
                Kokkos::View<size_type*, MemorySpace> bin_offsets;
                Vector<int, Dim> num_tiles;
            };

            /**
             * @brief Generic BinOp for tiled scatter/gather
             *
             * Bins particles into tiles based on their grid coordinates.
             * This is a generic implementation that works with any coordinate system and dimension.
             *
             * @tparam Dim Spatial dimension
             * @tparam RealType Floating point type
             * @tparam ExecSpace Kokkos execution space
             */
            template <unsigned Dim, typename RealType, typename ExecSpace>
            struct BinOp {
                using size_type = typename ExecSpace::memory_space::size_type;

                Vector<int, Dim> n_grid_global;  // GLOBAL grid dimensions (for coord transform)
                Vector<int, Dim> n_grid_local;   // LOCAL grid dimensions
                Vector<int, Dim> local_offset;   // First global index of local domain
                Vector<int, Dim> tile_size;      // Tile size per dimension
                Vector<int, Dim> num_tiles;      // Number of tiles per dimension (in local domain)
                int w;                           // kernel width

                // Mesh-based coordinate transformation
                CoordinateTransform<RealType, Dim> transform;

                KOKKOS_INLINE_FUNCTION int max_bins() const {
                    int total = 1;
                    for (unsigned d = 0; d < Dim; ++d) {
                        total *= (num_tiles[d] + 1);
                    }
                    return total + 1;
                }

                // Helper to access position component - works with both View<T*[3]> and
                // View<Vector<T,3>*>
                template <class ViewType>
                KOKKOS_INLINE_FUNCTION static auto get_component(const ViewType& keys, int i, int d)
                    -> decltype(keys(i, d)) {
                    return keys(i, d);  // For View<T*[3]>
                }

                template <class ViewType>
                KOKKOS_INLINE_FUNCTION static auto get_component(const ViewType& keys, int i, int d)
                    -> decltype(keys(i)[d]) {
                    return keys(i)[d];  // For View<Vector<T,3>*>
                }

                template <class ViewType>
                KOKKOS_INLINE_FUNCTION int bin(const ViewType& keys, int i) const {
                    // Compute tile index for each dimension
                    int tile[Dim];
                    for (unsigned d = 0; d < Dim; ++d) {
                        tile[d] = compute_bin(get_component(keys, i, d), d);
                    }

                    // Check if particle is outside local domain
                    for (unsigned d = 0; d < Dim; ++d) {
                        if (tile[d] < 0 || tile[d] >= static_cast<int>(num_tiles[d])) {
                            assert(false);
                            return max_bins() - 1;  // Put out-of-domain particles in last bin
                        }
                    }

                    int bin_idx = 0;
                    int stride  = 1;
                    for (int d = Dim - 1; d >= 0; --d) {
                        bin_idx += tile[d] * stride;
                        stride *= num_tiles[d];
                    }
                    return bin_idx;
                }

                KOKKOS_INLINE_FUNCTION int compute_bin(RealType val, int dim) const {
                    // Use mesh-based coordinate transformation
                    RealType grid_pos = transform.toGridCoordinate(val, dim);
                    // Sorting happens such
                    int global_idx = transform.getStencilCenter(grid_pos, w);
                    // Convert to local grid index
                    int local_idx = global_idx - local_offset[dim];
                    // Compute local tile index
                    return local_idx / tile_size[dim];
                }
            };

            /**
             * @brief Sort particles by tile and compute bin offsets - allocation-free
             * implementation
             *
             * This implementation performs binning directly without using Kokkos::BinSort,
             * avoiding all internal allocations. All required storage must be pre-allocated
             * and passed in.
             *
             * @tparam Dim Spatial dimension
             * @tparam RealType The floating point type
             * @tparam PositionViewType The position view type (View<T*[Dim]> or
             * View<Vector<T,Dim>*>)
             * @tparam ExecSpace Kokkos execution space
             * @tparam PermuteViewType View type for permutation indices
             * @tparam OffsetViewType View type for bin offsets
             * @tparam CountViewType View type for bin counts (atomic)
             *
             * @param x Positions in PHYSICAL coordinates (arbitrary domain)
             * @param n_grid_global GLOBAL grid dimensions
             * @param n_grid_local LOCAL grid dimensions
             * @param local_offset First global index of local domain
             * @param tile_size Tile size per dimension
             * @param w Kernel width
             * @param origin Mesh origin (physical coordinates)
             * @param invdx Inverse mesh spacing (1/dx)
             * @param permute [out] Pre-allocated view for permutation vector (size >= n_particles)
             * @param bin_offsets [out] Pre-allocated view for bin offsets (size >= n_tiles + 1)
             * @param bin_count [temp] Pre-allocated view for bin counts (size >= n_tiles), will be
             * zeroed
             * @param n_particles Number of particles to process
             */
            template <unsigned Dim, typename RealType, typename PositionViewType,
                      typename ExecSpace, typename PermuteViewType, typename OffsetViewType,
                      typename CountViewType>
            void bin_sort(PositionViewType x, Vector<int, Dim> n_grid_global,
                          Vector<int, Dim> n_grid_local, Vector<int, Dim> local_offset,
                          Vector<int, Dim> tile_size, int w, Vector<RealType, Dim> origin,
                          Vector<RealType, Dim> invdx, PermuteViewType& permute,
                          OffsetViewType& bin_offsets, CountViewType& bin_count,
                          const size_t n_particles) {
                using size_type = typename ExecSpace::memory_space::size_type;

                static IpplTimings::TimerRef binSortTimer = IpplTimings::getTimer("binSort");
                IpplTimings::startTimer(binSortTimer);

                // Calculate number of tiles based on LOCAL grid
                Vector<int, Dim> num_tiles;
                size_type n_bins = 1;
                for (unsigned d = 0; d < Dim; ++d) {
                    num_tiles[d] = (n_grid_local[d] + tile_size[d] - 1) / tile_size[d] + 1;
                    n_bins *= num_tiles[d];
                }

                // Create coordinate transform
                CoordinateTransform<RealType, Dim> transform(origin, invdx, n_grid_global);

                // Create BinOp
                BinOp<Dim, RealType, ExecSpace> bin_op{
                    n_grid_global, n_grid_local, local_offset, tile_size, num_tiles, w, transform};

                // Zero out the bin count array
                static IpplTimings::TimerRef zeroTimer = IpplTimings::getTimer("binZero");
                IpplTimings::startTimer(zeroTimer);
                Kokkos::deep_copy(bin_count, 0);
                IpplTimings::stopTimer(zeroTimer);

                // Step 1: Count particles per bin
                static IpplTimings::TimerRef countTimer = IpplTimings::getTimer("binCount");
                IpplTimings::startTimer(countTimer);

                // Create atomic view of bin_count for thread-safe incrementing
                using AtomicCountViewType =
                    Kokkos::View<size_t*, typename CountViewType::memory_space,
                                 Kokkos::MemoryTraits<Kokkos::Atomic>>;
                AtomicCountViewType bin_count_atomic = bin_count;

                Kokkos::parallel_for(
                    "BinSort::CountBins", Kokkos::RangePolicy<ExecSpace>(0, n_particles),
                    KOKKOS_LAMBDA(const size_type i) {
                        const int bin_idx = bin_op.bin(x, i);
                        bin_count_atomic(bin_idx)++;
                    });
                Kokkos::fence();
                IpplTimings::stopTimer(countTimer);

                // Step 2: Compute bin offsets via exclusive prefix scan
                static IpplTimings::TimerRef scanTimer = IpplTimings::getTimer("binScan");
                IpplTimings::startTimer(scanTimer);

                Kokkos::parallel_scan(
                    "BinSort::ComputeOffsets", Kokkos::RangePolicy<ExecSpace>(0, n_bins),
                    KOKKOS_LAMBDA(const size_type i, size_type& running_sum, const bool final) {
                        if (final) {
                            bin_offsets(i) = running_sum;
                        }
                        running_sum += bin_count(i);
                    });

                // Set the last offset (total count)
                Kokkos::parallel_for(
                    "BinSort::SetLastOffset", Kokkos::RangePolicy<ExecSpace>(0, 1),
                    KOKKOS_LAMBDA(const size_type) {
                        size_type total = 0;
                        for (size_type i = 0; i < n_bins; ++i) {
                            total += bin_count(i);
                        }
                        bin_offsets(n_bins) = total;
                    });
                Kokkos::fence();
                IpplTimings::stopTimer(scanTimer);

                // Step 3: Reset bin counts and place particles into permutation array
                static IpplTimings::TimerRef binningTimer = IpplTimings::getTimer("binPlacement");
                IpplTimings::startTimer(binningTimer);

                Kokkos::deep_copy(bin_count, 0);

                Kokkos::parallel_for(
                    "BinSort::CreatePermutation", Kokkos::RangePolicy<ExecSpace>(0, n_particles),
                    KOKKOS_LAMBDA(const size_type i) {
                        const int bin_idx             = bin_op.bin(x, i);
                        const int local_offset_in_bin = bin_count_atomic(bin_idx)++;
                        permute(bin_offsets(bin_idx) + local_offset_in_bin) = i;
                    });
                Kokkos::fence();
                IpplTimings::stopTimer(binningTimer);

                IpplTimings::stopTimer(binSortTimer);
            }

            /**
             * @brief Overload maintaining backward compatibility - allocates bin_count internally
             *
             * Note: This version still allocates a temporary bin_count array.
             * For fully allocation-free operation, use the version that takes bin_count as
             * parameter.
             */
            template <unsigned Dim, typename RealType, typename PositionViewType,
                      typename ExecSpace, typename PermuteViewType, typename OffsetViewType>
            void bin_sort(PositionViewType x, Vector<int, Dim> n_grid_global,
                          Vector<int, Dim> n_grid_local, Vector<int, Dim> local_offset,
                          Vector<int, Dim> tile_size, int w, Vector<RealType, Dim> origin,
                          Vector<RealType, Dim> invdx, PermuteViewType& permute,
                          OffsetViewType& bin_offsets, const size_t n_particles) {
                using memory_space = typename ExecSpace::memory_space;

                auto& buf_handler = ippl::detail::getDefaultSortBufferManager<memory_space>();
                auto bin_count    = buf_handler.mortonKeys();

                bin_sort<Dim, RealType, PositionViewType, ExecSpace>(
                    x, n_grid_global, n_grid_local, local_offset, tile_size, w, origin, invdx,
                    permute, bin_offsets, bin_count, n_particles);
            }

            /**
             * Note: This binning, algorithmically, only works for unifrom meshes. One would
             *       need to perform quite different stuff to allow for general meshes.
             * @tparam ParticleT
             * @tparam ParticleProperties
             * @tparam Dim
             * @param particles
             * @param fieldLayout
             * @param bin_size
             * @param w
             */
            template <typename ParticleT, typename FieldT, class... ParticleProperties,
                      unsigned Dim>
            std::tuple<Kokkos::View<size_type*,
                                    typename ParticleAttrib<Vector<ParticleT, Dim>,
                                                            ParticleProperties...>::memory_space>,
                       Kokkos::View<size_type*,
                                    typename ParticleAttrib<Vector<ParticleT, Dim>,
                                                            ParticleProperties...>::memory_space>,
                       Vector<int, Dim>>
            bin_particles(
                const ParticleAttrib<Vector<ParticleT, Dim>, ParticleProperties...>& particles,
                FieldLayout<Dim> fieldLayout, UniformCartesian<FieldT, Dim> mesh,
                Vector<int, Dim> bin_size, int w) {
                using AttribType   = std::decay_t<decltype(particles)>;
                using ExecSpace    = AttribType::execution_space;
                using memory_space = AttribType::memory_space;

                const NDIndex<Dim>& lDom = fieldLayout.getLocalNDIndex();
                const NDIndex<Dim>& gDom = fieldLayout.getDomain();
                Vector<int, Dim> ngrid_global;
                Vector<int, Dim> ngrid_local;
                Vector<int, Dim> local_offset;
                for (unsigned d = 0; d < Dim; ++d) {
                    ngrid_global[d] = gDom[d].length();
                    ngrid_local[d]  = lDom[d].length();
                    local_offset[d] = lDom[d].first();
                }
                auto particle_view       = particles.getView();
                const auto invdx         = 1.0 / mesh.getMeshSpacing();
                const size_t n_particles = particles.getParticleCount();

                Vector<int, Dim> num_tiles;
                size_t total_tiles = 1;
                for (unsigned d = 0; d < Dim; ++d) {
                    num_tiles[d] = (ngrid_local[d] + bin_size[d] - 1) / bin_size[d] + 1;
                    total_tiles *= num_tiles[d];
                }

                auto& buf_handler = ippl::detail::getDefaultSortBufferManager<memory_space>();
                buf_handler.ensureCapacity(std::max(n_particles + 1, total_tiles + 1));

                auto permute     = buf_handler.indices();
                auto bin_offsets = buf_handler.indicesSorted();
                auto bin_count   = buf_handler.mortonKeys();

                bin_sort<Dim, ParticleT, std::decay_t<decltype(particle_view)>, ExecSpace>(
                    particle_view, ngrid_global, ngrid_local, local_offset, bin_size, w,
                    mesh.getOrigin(), invdx, permute, bin_offsets, bin_count, n_particles);

                return {permute, bin_offsets, num_tiles};
            }
        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_INTERPOLATION_BINNING_H