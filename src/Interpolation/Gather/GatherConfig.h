#ifndef IPPL_GATHER_CONFIG_H
#define IPPL_GATHER_CONFIG_H

#include <array>

namespace ippl {
    namespace Interpolation {

        /**
         * @brief Scattering/gathering method for particle-grid interpolation
         *
         * Different methods offer various performance characteristics:
         * - Atomic: Simple atomic operations, no sorting
         * - AtomicSort: Atomic operations with particle sorting for better cache locality
         * - Tiled: Cache-friendly tiling with team policies and shared memory histograms
         */
        enum class GatherMethod {
            Atomic,
            AtomicSort,
            Tiled,
            Native
        };

        /**
         * @brief Configuration for scatter/gather operations
         */
        template <unsigned Dim>
        struct GatherConfig {
            GatherMethod method = GatherMethod::Tiled;

            // Tile size per dimension
            std::array<int, Dim> tile_size;

            // Team size for team-based methods
            int team_size = 16;

            bool add_to_attribute = false;

            /**
             * @brief Default constructor - initializes tile sizes based on Dim
             */
            GatherConfig() {
                if constexpr (Dim == 1) {
                    tile_size.fill(512);
                } else if constexpr (Dim == 2) {
                    tile_size.fill(32);
                } else {
                    tile_size.fill(8);
                }
            }

            /**
             * @brief Constructor with uniform tile size for all dimensions
             */
            explicit GatherConfig(int uniform_tile_size)
                : GatherConfig() {
                tile_size.fill(uniform_tile_size);
            }

            /**
             * @brief Constructor with per-dimension tile sizes
             */
            explicit GatherConfig(std::array<int, Dim> tile_sizes)
                : GatherConfig() {
                tile_size = tile_sizes;
            }

            /**
             * @brief Get tile size as Vector for use in kernels
             */
            Vector<int, Dim> get_tile_size() const {
                Vector<int, Dim> result;
                for (unsigned d = 0; d < Dim; ++d) {
                    result[d] = tile_size[d];
                }
                return result;
            }

            /**
             * @brief Set uniform tile size for all dimensions
             */
            GatherConfig& set_tile_size(int uniform_size) {
                tile_size.fill(uniform_size);
                return *this;
            }

            /**
             * @brief Set tile size per dimension
             */
            GatherConfig& set_tile_size(std::array<int, Dim> sizes) {
                tile_size = sizes;
                return *this;
            }

            /**
             * @brief Set tile size for a specific dimension
             */
            GatherConfig& set_tile_size(unsigned dim, int size) {
                tile_size[dim] = size;
                return *this;
            }

            bool do_binning() const {
                return method == GatherMethod::AtomicSort;
            }

            /**
             * @brief Get default configuration for an execution space
             */
            template <typename ExecSpace>
            static GatherConfig get_default();
        };

        // Helper to define defaults for execution space + dimension combinations
        namespace detail {
            template <unsigned Dim, typename ExecSpace, typename = void>
            struct GatherConfigDefault {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method = GatherMethod::Atomic;
                    return config;
                }
            };

#ifdef KOKKOS_ENABLE_SERIAL
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::Serial> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method = GatherMethod::Atomic;
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_CUDA
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::Cuda> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method    = GatherMethod::AtomicSort;
                    config.team_size = 32;

                    if constexpr (Dim == 1) {
                        config.tile_size = {512};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else if constexpr (Dim == 3) {
                        config.tile_size = {4, 4, 4};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_OPENMP
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::OpenMP> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method    = GatherMethod::Atomic;
                    config.team_size = 4;

                    if constexpr (Dim == 1) {
                        config.tile_size = {256};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else if constexpr (Dim == 3) {
                        config.tile_size = {9, 9, 9};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_HIP
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::HIP> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method    = GatherMethod::Tiled;
                    config.team_size = 64;

                    if constexpr (Dim == 1) {
                        config.tile_size = {512};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else if constexpr (Dim == 3) {
                        config.tile_size = {6, 6, 6};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_THREADS
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::Threads> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method = GatherMethod::AtomicSort;
                    return config;
                }
            };
#endif
        }  // namespace detail

        // Implementation of get_default using the helper
        template <unsigned Dim>
        template <typename ExecSpace>
        GatherConfig<Dim> GatherConfig<Dim>::get_default() {
            return detail::GatherConfigDefault<Dim, ExecSpace>::get();
        }

    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_GATHER_CONFIG_H
