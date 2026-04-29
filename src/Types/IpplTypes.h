//
// IpplTypes
//   Typedefs for basic types used throughout IPPL
//

#ifndef IPPL_TYPES_H
#define IPPL_TYPES_H

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda/std/limits>
#endif

#include <cstddef>  //For std::size_t

namespace ippl {
    namespace detail {
        typedef std::size_t size_type;

        template <typename T>
        T KOKKOS_INLINE_FUNCTION constexpr infinity() {
#ifdef KOKKOS_ENABLE_CUDA
            return cuda::std::numeric_limits<T>::infinity();
#else
            return std::numeric_limits<T>::infinity();
#endif
        }
    }  // namespace detail

}  // namespace ippl

#endif
