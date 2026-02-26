//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
#include <cstring>

#include "Archive.h"

namespace ippl {
    namespace detail {

        template <typename T, typename HashView, typename BufferPtr>
        struct SerializeHashFunctor {
            const T* view_data;
            HashView hash;
            BufferPtr buf;
            size_t elem_size;
            size_t wpos;

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                std::memcpy(buf + i * elem_size + wpos, view_data + hash(i), elem_size);
            }
        };

        template <typename T, unsigned Dim, typename HashView, typename BufferPtr>
        struct SerializeHashVectorFunctor {
            const Vector<T, Dim>* view_data;
            HashView hash;
            BufferPtr buf;
            size_t elem_size;
            size_t wpos;

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i, const size_t d) const {
                const Vector<T, Dim>* vec = view_data + hash(i);
                const T* elem             = reinterpret_cast<const T*>(vec) + d;
                std::memcpy(buf + (Dim * i + d) * elem_size + wpos, elem, elem_size);
            }
        };

        template <class... Properties>
        Archive<Properties...>::Archive(size_type size)
            : writepos_m(0)
            , readpos_m(0)
            , buffer_m("buffer", size) {}

        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::serialize(const Kokkos::View<T*, ViewArgs...>& view,
                                               size_type nsends) {
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            size_t size    = sizeof(T);
            auto& buffer   = buffer_m;
            auto& writepos = writepos_m;
            Kokkos::parallel_for(
                "Archive::serialize()", policy_type(0, nsends), KOKKOS_LAMBDA(const size_type i) {
                    std::memcpy(buffer.data() + i * size + writepos, view.data() + i, size);
                });
            Kokkos::fence();
            writepos_m += size * nsends;
        }

        template <class... Properties>
        template <typename T, class... ViewArgs, typename HashView>
        void Archive<Properties...>::serialize(const Kokkos::View<T*, ViewArgs...>& view,
                                               const HashView& hash, size_type nsends) {
            using exec_space  = HashView::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;
            using BufferPtr   = buffer_type::pointer_type;

            SerializeHashFunctor<T, HashView, BufferPtr> f{view.data(), hash, buffer_m.data(),
                                                           sizeof(T), writepos_m};

            Kokkos::parallel_for("Archive::serialize(hash)", policy_type(0, nsends), f);
            Kokkos::fence();
            writepos_m += sizeof(T) * nsends;
        }

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::serialize(
            const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view, size_type nsends) {
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;

            size_t size    = sizeof(T);
            auto& buffer   = buffer_m;
            auto& writepos = writepos_m;
            // Default index type for range policies is int64,
            // so we have to explicitly specify size_type (uint64)
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            Kokkos::parallel_for(
                "Archive::serialize()",
                // The constructor for Kokkos range policies always
                // expects int64 regardless of index type provided
                // by template parameters, so the typecast is necessary
                // to avoid compiler warnings
                mdrange_t({0, 0}, {(long int)nsends, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    std::memcpy(buffer.data() + (Dim * i + d) * size + writepos,
                                &(*(view.data() + i))[d], size);
                });
            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs, typename HashView>
        void Archive<Properties...>::serialize(
            const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view, const HashView& hash,
            size_type nsends) {
            using exec_space = typename HashView::execution_space;  // was missing typename
            size_t size      = sizeof(T);
            using BufferPtr  = typename buffer_type::pointer_type;
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;

            SerializeHashVectorFunctor<T, Dim, HashView, BufferPtr> f{
                view.data(), hash, buffer_m.data(), size, writepos_m};

            Kokkos::parallel_for("Archive::serialize(hash, vector)",
                                 mdrange_t({0, 0}, {(long int)nsends, Dim}), f);
            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<T*, ViewArgs...>& view,
                                                 size_type nrecvs) {
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            size_t size = sizeof(T);
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            auto buffer  = buffer_m;
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize()", policy_type(0, nrecvs), KOKKOS_LAMBDA(const size_type i) {
                    std::memcpy(view.data() + i, buffer.data() + i * size + readpos, size);
                });
            // Wait for deserialization kernel to complete
            // (as with serialization kernels)
            Kokkos::fence();
            readpos_m += size * nrecvs;
        }

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                                 size_type nrecvs) {
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;

            size_t size = sizeof(T);
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            auto buffer  = buffer_m;
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize()", mdrange_t({0, 0}, {(long int)nrecvs, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    std::memcpy(&(*(view.data() + i))[d],
                                buffer.data() + (Dim * i + d) * size + readpos, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }

        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<T*, ViewArgs...>& view,
                                                 size_type offset, size_type nrecvs) {
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;
            size_t size       = sizeof(T);
            if (offset + nrecvs > view.extent(0)) {
                Kokkos::resize(view, offset + nrecvs);
            }
            auto buffer  = buffer_m;
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize(offset)", policy_type(0, nrecvs),
                KOKKOS_LAMBDA(const size_type i) {
                    std::memcpy(view.data() + offset + i, buffer.data() + i * size + readpos, size);
                });
            Kokkos::fence();
            readpos_m += size * nrecvs;
        }

        // Offset deserialize — Vector
        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                                 size_type offset, size_type nrecvs) {
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            size_t size      = sizeof(T);
            if (offset + nrecvs > view.extent(0)) {
                Kokkos::resize(view, offset + nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            auto buffer  = buffer_m;
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize(offset, vector)", mdrange_t({0, 0}, {(long int)nrecvs, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    std::memcpy(&(*(view.data() + offset + i))[d],
                                buffer.data() + (Dim * i + d) * size + readpos, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }
    }  // namespace detail
}  // namespace ippl
