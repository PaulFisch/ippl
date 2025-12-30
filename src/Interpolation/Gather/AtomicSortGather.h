#ifndef IPPL_ATOMIC_SORT_GATHER_3D_H
#define IPPL_ATOMIC_SORT_GATHER_3D_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>

#include "../InterpolationTypes.h"

#include "../CoordinateTransform.h"
#include "../InterpolationUtil.h"
#include "../KernelEvaluator.h"
#include "../StencilHelper.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

            /**
             * @brief  gather functor with precomputed kernel values
             *
             * Template parameters:
             * @tparam W Compile-time kernel width
             * @tparam RealType The floating point type
             * @tparam ExecSpace The Kokkos execution space
             * @tparam KernelType The kernel function type
             * @tparam ValueType The type of values being gathered
             * @tparam GridViewType The grid view type
             * @tparam PositionViewType The type of the position view
             * @tparam PermuteViewType The type of the permutation view
             */
            template <int W,
                      unsigned Dim,
                      typename RealType,
                      typename ExecSpace,
                      typename KernelType,
                      typename ValueType,
                      typename GridViewType,
                      typename PositionViewType =
                          Kokkos::View<ippl::Vector<RealType, Dim>*,
                                       typename ExecSpace::memory_space>,
                      typename PermuteViewType =
                          Kokkos::View<std::size_t*, typename ExecSpace::memory_space>>
            struct AtomicSortGatherFunctor {
                using real_type    = RealType;
                using value_type   = ValueType;
                using memory_space = typename ExecSpace::memory_space;
                using size_type    = typename memory_space::size_type;

                // Input data
                const std::decay_t<PositionViewType> x;       // particle positions in PHYSICAL coordinates
                const std::decay_t<GridViewType>     grid;    // input grid
                const std::decay_t<PermuteViewType>  permute; // permutation: sorted_index -> particle_index

                // Output data
                const Kokkos::View<value_type*, memory_space> values;  // per-particle gathered values

                // Parameters
                const Vector<int, Dim> n_grid;        // GLOBAL grid dimensions
                const Vector<int, Dim> n_grid_local;  // LOCAL grid dimensions
                const Vector<int, Dim> local_offset;  // first global index of local domain
                const int nghost;                      // ghost cell offset
                const real_type inv_hw;                // 1 / half_width for kernel scaling
                const bool add_to_attribute;           // if true, add to existing values; otherwise overwrite
                const std::decay_t<KernelType> kernel;

                // Mesh information for coordinate transformation
                const Vector<real_type, Dim> origin;   // Physical origin from mesh
                const Vector<real_type, Dim> invdx;    // Inverse mesh spacing (1/dx)

                KOKKOS_INLINE_FUNCTION void operator()(const size_type j_sorted) const {
                    // Map sorted index -> actual particle index
                    const size_type p = permute(j_sorted);

                    // Set up coordinate transformation with mesh information
                    CoordinateTransform<real_type, Dim> transform(origin, invdx, n_grid);

                    // Transform from physical coordinates to grid coordinates
                    real_type pos[Dim];
                    int idx0[Dim];
                    for (int d = 0; d < Dim; ++d) {
                        pos[d] = transform.toGridCoordinate(x(p)[d], d);
                        idx0[d] = transform.getStencilBase(pos[d], W);
                    }

                    // Precompute kernel values using KernelEvaluator
                    KernelEvaluator<W, Dim, std::decay_t<KernelType>, real_type> keval;
                    keval.evaluate(pos, idx0, inv_hw, kernel);

                    // Convert base indices to local coordinates
                    int base_idx_local[Dim];
                    for (unsigned d = 0; d < Dim; ++d) {
                        base_idx_local[d] = idx0[d] - local_offset[d] + nghost;
                    }

#ifndef NDEBUG
                    // Check bounds in debug mode
                    for (unsigned d = 0; d < Dim; ++d) {
                        assert(base_idx_local[d] >= 0 &&
                               base_idx_local[d] + W <= n_grid_local[d] + 2 * nghost);
                    }
#endif

                    // Determine grid element type from grid view
                    using grid_element_type = std::remove_reference_t<
                        decltype(accessGrid<Dim>(grid, base_idx_local))>;

                    grid_element_type result(0);

                    // Gather with precomputed kernel values using iterateStencil
                    iterateStencil<W, Dim>([&](const int* stencil_offset) {
                        // Compute kernel weight
                        real_type kernel_val = 1.0;
                        for (unsigned d = 0; d < Dim; ++d) {
                            kernel_val *= keval.getValue(d, stencil_offset[d]);
                        }

                        // Access grid with dimension-agnostic helper
                        int idx[Dim];
                        for (unsigned d = 0; d < Dim; ++d) {
                            idx[d] = base_idx_local[d] + stencil_offset[d];
                        }
                        auto grid_val = accessGrid<Dim>(grid, idx);

                        result += grid_val * kernel_val;
                    });

                    // Write result to output, extracting real part if needed
                    constexpr bool val_is_complex =
                        std::is_same_v<value_type, Kokkos::complex<real_type>>;
                    constexpr bool grid_is_complex =
                        std::is_same_v<grid_element_type, Kokkos::complex<real_type>>;

                    if constexpr (grid_is_complex && !val_is_complex) {
                        // Grid is complex but output is real - extract real part
                        if (add_to_attribute) {
                            values(p) += result.real();
                        } else {
                            values(p) = result.real();
                        }
                    } else {
                        // Types match (both real or both complex)
                        if (add_to_attribute) {
                            values(p) += result;
                        } else {
                            values(p) = result;
                        }
                    }
                }
            };

            /**
             * @brief Dispatcher for gather with runtime kernel width
             *
             * @tparam Dim Spatial dimension (1, 2, 3, or higher)
             */
            template <unsigned Dim,
                      typename RealType,
                      typename ExecSpace,
                      typename KernelType,
                      typename ValueType,
                      typename GridViewType,
                      typename PositionViewType,
                      typename PermuteViewType,
                      typename OutputViewType>
            void dispatch_gather(
                const PositionViewType& x,
                const GridViewType& grid,
                const PermuteViewType& permute,
                const OutputViewType& values,
                const Vector<int, Dim>& n_grid,
                const Vector<int, Dim>& n_grid_local,
                const Vector<int, Dim>& local_offset,
                int w,
                int nghost,
                RealType inv_hw,
                bool add_to_attribute,
                const KernelType& kernel,
                size_t n_particles,
                const Vector<RealType, Dim>& origin,
                const Vector<RealType, Dim>& invdx) {

                auto create_and_run = [&]<int W>() {
                    AtomicSortGatherFunctor<W, Dim, RealType, ExecSpace, KernelType, ValueType,
                                            GridViewType, PositionViewType, PermuteViewType>
                        functor{x, grid, permute, values, n_grid, n_grid_local, local_offset,
                                nghost, inv_hw, add_to_attribute, kernel, origin, invdx};

                    // (paul) Remove the fences here with caution. HIP gives invalid memory
                    //         access errors with the current rocm (old) 6.0.2
                    Kokkos::fence();
                    Kokkos::parallel_for(
                        "AtomicSortGather",
                        Kokkos::RangePolicy<ExecSpace>(0, n_particles),
                        functor);
                    Kokkos::fence();
                };

                switch (w) {
                    case 1: create_and_run.template operator()<1>(); break;
                    case 2: create_and_run.template operator()<2>(); break;
                    case 3: create_and_run.template operator()<3>(); break;
                    case 4: create_and_run.template operator()<4>(); break;
                    case 5: create_and_run.template operator()<5>(); break;
                    case 6: create_and_run.template operator()<6>(); break;
                    case 7: create_and_run.template operator()<7>(); break;
                    case 8: create_and_run.template operator()<8>(); break;
                    case 9: create_and_run.template operator()<9>(); break;
                    case 10: create_and_run.template operator()<10>(); break;
                    case 11: create_and_run.template operator()<11>(); break;
                    case 12: create_and_run.template operator()<12>(); break;
                    case 13: create_and_run.template operator()<13>(); break;
                    case 14: create_and_run.template operator()<14>(); break;

                    default:
                        Kokkos::abort("AtomicSortGatherFunctor: unsupported kernel width");
                }
            }

            // Backward compatibility alias for 3D
            template <typename RealType,
                      typename ExecSpace,
                      typename KernelType,
                      typename ValueType,
                      typename GridViewType,
                      typename PositionViewType,
                      typename PermuteViewType,
                      typename OutputViewType>
            void dispatch_gather_3d(
                const PositionViewType& x,
                const GridViewType& grid,
                const PermuteViewType& permute,
                const OutputViewType& values,
                const Vector<int, 3>& n_grid,
                const Vector<int, 3>& n_grid_local,
                const Vector<int, 3>& local_offset,
                int w,
                int nghost,
                RealType inv_hw,
                bool add_to_attribute,
                const KernelType& kernel,
                size_t n_particles,
                const Vector<RealType, 3>& origin,
                const Vector<RealType, 3>& invdx) {
                dispatch_gather<3, RealType, ExecSpace, KernelType, ValueType,
                               GridViewType, PositionViewType, PermuteViewType, OutputViewType>(
                    x, grid, permute, values, n_grid, n_grid_local, local_offset,
                    w, nghost, inv_hw, add_to_attribute, kernel, n_particles,
                    origin, invdx);
            }

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_ATOMIC_SORT_GATHER_3D_H