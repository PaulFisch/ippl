#ifndef IPPL_NUFFT_UTILITIES_H
#define IPPL_NUFFT_UTILITIES_H

/**
 * @file NUFFTUtilities.h
 * @brief NUFFT utility helpers (deconvolution factors and small array math)
 *        used by the native Kokkos NUFFT engine.
 */

#include <chrono>
#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Array.hpp>

#include "ESKernel.h"

namespace ippl {
namespace nufft {

    // ====================================================================
    // Type Definitions
    // ====================================================================

    template<typename T, size_t n>
    using array = Kokkos::Array<T, n>;

    // Multi-dimensional view type helper
    template<class T, int N>
    struct add_ptr_n : add_ptr_n<std::add_pointer_t<T>, N - 1> {};

    template<class T>
    struct add_ptr_n<T, 0> {
        using type = T;
    };

    template<class T, int N>
    using add_ptr_n_t = typename add_ptr_n<T, N>::type;

    template<class T, int Dim, class... Properties>
    using view_nd_t = Kokkos::View<add_ptr_n_t<T, Dim>, Properties...>;

    template<class T, int Dim, class MemorySpace, std::size_t... I>
    static view_nd_t<T, Dim, MemorySpace>
    make_view_impl(const std::string_view label, const array<typename MemorySpace::size_type, Dim> &n,
                   std::index_sequence<I...>) {
        return view_nd_t<T, Dim, MemorySpace>(std::string(label),
                                              static_cast<typename MemorySpace::size_type>(n[I])...);
    }

    template<class T, int Dim, class MemorySpace>
    view_nd_t<T, Dim, MemorySpace>
    make_view(std::string_view label, const array<typename MemorySpace::size_type, Dim> &n) {
        static_assert(Dim > 0, "Dim must be >= 1");
        return make_view_impl<T, Dim, MemorySpace>(label, n, std::make_index_sequence<Dim>{});
    }

    // ====================================================================
    // Quadrature
    // ====================================================================

    /**
     * @brief Computes Gauss-Legendre quadrature nodes and weights.
     */
    template<typename ExecSpace, typename RealType = double>
    void gauss_legendre(int n,
                        Kokkos::View<RealType *, typename ExecSpace::memory_space> &nodes,
                        Kokkos::View<RealType *, typename ExecSpace::memory_space> &weights) {
        constexpr RealType eps = std::numeric_limits<RealType>::epsilon();

        // Create host views to compute values
        auto h_nodes = Kokkos::create_mirror_view(Kokkos::HostSpace(), nodes);
        auto h_weights = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);

        for (int i = 0; i < n; ++i) {
            RealType x = std::cos(Kokkos::numbers::pi_v<RealType> * (i + 0.75) / (n + 0.5));
            RealType pp, delta;

            do {
                RealType p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; ++j) {
                    const RealType p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * x * p2 - j * p3) / (j + 1.0);
                }
                pp = n * (x * p1 - p2) / (x * x - 1.0);
                delta = p1 / pp;
                x -= delta;
            } while (std::abs(delta) > eps);

            h_nodes(i) = x;
            h_weights(i) = 2.0 / ((1.0 - x * x) * pp * pp);
        }

        // Exploit symmetry
        for (int i = 0; i < n / 2; ++i) {
            const int j = n - 1 - i;
            h_nodes(j) = -h_nodes(i);
            h_weights(j) = h_weights(i);
        }

        // Copy to device
        Kokkos::deep_copy(nodes, h_nodes);
        Kokkos::deep_copy(weights, h_weights);
    }


    // ====================================================================
    // Correction (Type 1 and Type 2)
    // ====================================================================

    /**
     * @brief Unified functor for applying correction (Type 1 or Type 2).
     */
    template<int Dim, class ExecSpace, typename RealType = double>
    struct CorrectionFunctor {
        using exec_space = ExecSpace;
        using memory_space = typename exec_space::memory_space;
        using real_type = RealType;
        using complex_type = Kokkos::complex<real_type>;
        using size_type = typename memory_space::size_type;
        using complex_view = Kokkos::View<complex_type *, memory_space>;
        using grid_view_type = view_nd_t<complex_type, Dim, memory_space>;

        grid_view_type input;
        array<complex_view, Dim> factors;
        grid_view_type output;
        array<size_type, Dim> n_modes;
        array<size_type, Dim> n_input;
        array<size_type, Dim> n_output;

        template<typename... Indices>
        KOKKOS_INLINE_FUNCTION void operator()(Indices... indices) const {
            array<int64_t, Dim> k{static_cast<int64_t>(indices)...};

            // Map NUFFT mode indices to FFT grid indices
            array<int64_t, Dim> in_idx, out_idx;
            complex_type factor(1.0, 0.0);

            for (int d = 0; d < Dim; ++d) {
                // Frequency wrapping for FFT layout
                in_idx[d] = (k[d] < n_modes[d] / 2) ? k[d] : n_input[d] - (n_modes[d] - k[d]);
                out_idx[d] = (k[d] < n_modes[d] / 2) ? k[d] : n_output[d] - (n_modes[d] - k[d]);
                factor *= factors[d](k[d]);
            }

            // Apply correction: output = input * factor
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                output(out_idx[Is]...) = input(in_idx[Is]...) * factor;
            }(std::make_index_sequence<Dim>{});
        }
    };

} // namespace nufft
} // namespace ippl

#endif // IPPL_NUFFT_UTILITIES_H
