//
// Class HaloCells
//   The guard / ghost cells of BareField.
//

#include <memory>
#include <vector>

#include "Utility/IpplException.h"
#include "Utility/ParallelDispatch.h"

#include "Communicate/Communicator.h"

// Compile-time-toggled deadlock instrumentation.
//
//   cmake -DIPPL_HALO_DEBUG=ON ...
//
// Every halo exchange logs to stderr (flushed line-by-line):
//
//   [HALO/r=R][stage][k=v][...]
//
// Recommended workflow:
//   1. mpirun ... 2> halo.log
//   2. when it deadlocks, kill -SIGABRT $PID  (or wait for timeout)
//   3. sort -s -k1,1 halo.log    # group by rank
//      grep "isend\|recv" halo.log | sort | uniq -c
//   4. matched (target,tag,nsends) on rank A must equal (source,tag,nrecvs)
//      on rank A's peer. Any discrepancy is the deadlock.
//
// All logging is host-side only (no GPU prints).
#ifdef IPPL_HALO_DEBUG
#include <cstdio>
#include <atomic>
#include <sstream>
#include <mpi.h>
namespace ippl { namespace detail {
    inline std::atomic<unsigned long>& haloDebugSeq() {
        static std::atomic<unsigned long> s{0};
        return s;
    }
}}  // namespace ippl::detail
// Single fprintf so a line emitted by one rank does not interleave with
// a line emitted by another (POSIX guarantees fprintf with one buffer is
// atomic up to PIPE_BUF; one fprintf per line keeps the trace readable).
#define IPPL_HALO_LOG(stream)                                                              \
    do {                                                                                   \
        std::ostringstream _ipplhalolog_oss;                                               \
        _ipplhalolog_oss << "[HALO/r=" << ippl::Comm->rank()                               \
                         << "][seq=" << ippl::detail::haloDebugSeq().fetch_add(1) << "]"   \
                         << stream << '\n';                                                \
        const auto _ipplhalolog_s = _ipplhalolog_oss.str();                                \
        std::fprintf(stderr, "%s", _ipplhalolog_s.c_str());                                \
        std::fflush(stderr);                                                               \
    } while (0)
// Every IPPL_HALO_SYNC logs a "pre" line on every rank, calls MPI_Barrier,
// then logs "post". When one rank desyncs, the LAST line in the log is the
// SYNC label closest to the divergence:
//
//   - if all ranks have a "sync:pre id=FOO" but only some have
//     "sync:post id=FOO", the divergence happened *before* FOO and ranks
//     are stuck inside the barrier itself.
//   - if one rank has "sync:post id=FOO" but the others don't, that rank
//     reached past FOO while the others are stuck on a prior MPI op.
//
// Use ascending phase IDs so the trace is grep-able. Cheap (~µs per call);
// safe to sprinkle generously inside CG loops while we hunt the bug.
// `label` is streamed straight into the log via chained `<<`, so callers can
// pass either a single string ("halo:enter") or a composed expression
// ("BConds:bc_pre[" << idx << "]"). Don't wrap label in parens; that would
// break the `<<` chain.
#define IPPL_HALO_SYNC(label)                                                              \
    do {                                                                                   \
        IPPL_HALO_LOG("sync:pre id=" << label);                                            \
        MPI_Barrier(ippl::Comm->getCommunicator());                                        \
        IPPL_HALO_LOG("sync:post id=" << label);                                           \
    } while (0)
#else
#define IPPL_HALO_LOG(stream) do {} while (0)
#define IPPL_HALO_SYNC(label) do {} while (0)
#endif

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, class... ViewArgs>
        HaloCells<T, Dim, ViewArgs...>::HaloCells() {}

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::accumulateHalo(view_type& view, Layout_t* layout, int nghost) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL, nghost);
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::accumulateHalo_noghost(view_type& view, Layout_t* layout, int nghost) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL_NOGHOST, nghost);
        }
        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::fillHalo(view_type& view, Layout_t* layout, int nghost) {
            exchangeBoundaries<assign>(view, layout, INTERNAL_TO_HALO, nghost);
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <class Op>
        void HaloCells<T, Dim, ViewArgs...>::exchangeBoundaries(view_type& view, Layout_t* layout,
                                                                SendOrder order, int nghost) {
            using neighbor_list = typename Layout_t::neighbor_list;
            using range_list    = typename Layout_t::neighbor_range_list;

            auto& comm = layout->comm;

            const neighbor_list& neighbors = layout->getNeighbors();
            const range_list &sendRanges   = layout->getNeighborsSendRange(),
                             &recvRanges   = layout->getNeighborsRecvRange();

            auto ldom = layout->getLocalNDIndex();
            for (const auto& axis : ldom) {
                if ((axis.length() == 1) && (Dim != 1)) {
                    throw IpplException(
                        "HaloCells::exchangeBoundaries",
                        "Cannot do neighbour exchange when domain decomposition contains planes.");
                }
            }

            // needed for the NOGHOST approach - we want to remove the ghost
            // cells on the boundaries of the global domain from the halo
            // exchange when we set HALO_TO_INTERNAL_NOGHOST
            const auto domain = layout->getDomain();
            const auto& ldomains = layout->getHostLocalDomains();

            size_t totalRequests = 0;
            for (const auto& componentNeighbors : neighbors) {
                totalRequests += componentNeighbors.size();
            }

            int me = Comm->rank();

            IPPL_HALO_LOG("exchangeBoundaries:enter order=" << static_cast<int>(order)
                          << " totalRequests=" << totalRequests << " nghost=" << nghost);
            IPPL_HALO_SYNC("halo:enter");
#ifdef IPPL_HALO_DEBUG
            for (size_t k = 0; k < neighbors.size(); ++k) {
                std::ostringstream oss;
                oss << "neighbors[" << k << "]={";
                for (size_t i = 0; i < neighbors[k].size(); ++i) {
                    if (i) oss << ",";
                    oss << neighbors[k][i];
                }
                oss << "}";
                IPPL_HALO_LOG(oss.str());
            }
#endif

            using memory_space = typename view_type::memory_space;
            using buffer_type  = mpi::Communicator::buffer_type<memory_space>;
            std::vector<MPI_Request> requests(totalRequests);
            // sending loop
            constexpr size_t cubeCount = detail::countHypercubes(Dim) - 1;
            size_t requestIndex        = 0;
            for (size_t index = 0; index < cubeCount; index++) {
                int tag                        = mpi::tag::HALO + index;
                const auto& componentNeighbors = neighbors[index];
                for (size_t i = 0; i < componentNeighbors.size(); i++) {
                    int targetRank = componentNeighbors[i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        /*We store only the sending and receiving ranges
                         * of INTERNAL_TO_HALO and use the fact that the
                         * sending range of HALO_TO_INTERNAL is the receiving
                         * range of INTERNAL_TO_HALO and vice versa
                         */
                        range = sendRanges[index][i];
                    } else if (order == HALO_TO_INTERNAL_NOGHOST) {
                        range = recvRanges[index][i];

                        for (size_t j = 0; j < Dim; ++j) {
                            bool isLower = ((range.lo[j] + ldomains[me][j].first()
                                            - nghost) == domain[j].min());
                            bool isUpper = ((range.hi[j] - 1 + 
                                            ldomains[me][j].first() - nghost)
                                            == domain[j].max());
                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = recvRanges[index][i];
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf = comm.template getBuffer<memory_space, T>(nsends);

                    IPPL_HALO_LOG("isend index=" << index << " i=" << i
                                  << " target=" << targetRank << " tag=" << tag
                                  << " nsends=" << nsends);
                    comm.isend(targetRank, tag, haloData_m, *buf, requests[requestIndex++], nsends);
                    buf->resetWritePos();
                }
            }

            IPPL_HALO_LOG("exchangeBoundaries:sends_done count=" << requestIndex);
            IPPL_HALO_SYNC("halo:sends_done");

            // receiving loop
            for (size_t index = 0; index < cubeCount; index++) {
                int tag                        = mpi::tag::HALO + Layout_t::getMatchingIndex(index);
                const auto& componentNeighbors = neighbors[index];
                for (size_t i = 0; i < componentNeighbors.size(); i++) {
                    int sourceRank = componentNeighbors[i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        range = recvRanges[index][i];
                    } else if (order == HALO_TO_INTERNAL_NOGHOST) {
                        range = sendRanges[index][i];

                        for (size_t j = 0; j < Dim; ++j) {
                            bool isLower = ((range.lo[j] + ldomains[me][j].first()
                                            - nghost) == domain[j].min());
                            bool isUpper = ((range.hi[j] - 1 + 
                                            ldomains[me][j].first() - nghost)
                                            == domain[j].max());
                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = sendRanges[index][i];
                    }

                    size_type nrecvs = range.size();

                    buffer_type buf = comm.template getBuffer<memory_space, T>(nrecvs);

                    IPPL_HALO_LOG("recv:pre index=" << index << " i=" << i
                                  << " source=" << sourceRank << " tag=" << tag
                                  << " nrecvs=" << nrecvs);
                    comm.recv(sourceRank, tag, haloData_m, *buf, nrecvs * sizeof(T), nrecvs);
                    IPPL_HALO_LOG("recv:post index=" << index << " i=" << i
                                  << " source=" << sourceRank << " tag=" << tag);
                    buf->resetReadPos();

                    unpack<Op>(range, view, haloData_m);
                    IPPL_HALO_LOG("unpack:done index=" << index << " i=" << i
                                  << " source=" << sourceRank);
                }
            }

            IPPL_HALO_LOG("exchangeBoundaries:recvs_done");
            IPPL_HALO_SYNC("halo:recvs_done");

            if (totalRequests > 0) {
                IPPL_HALO_LOG("waitall:pre n=" << totalRequests);
                MPI_Waitall(totalRequests, requests.data(), MPI_STATUSES_IGNORE);
                IPPL_HALO_LOG("waitall:post");
            }

            comm.freeAllBuffers();
            IPPL_HALO_LOG("exchangeBoundaries:exit");
            IPPL_HALO_SYNC("halo:exit");
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::pack(const bound_type& range, const view_type& view,
                                                  databuffer_type& fd, size_type& nsends) {
            auto subview = makeSubview(view, range);

            auto& buffer = fd.buffer;

            size_t size = subview.size();
            nsends      = size;
            if (buffer.size() < size) {
                double overalloc = Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, static_cast<size_t>(size * overalloc));
            }

            using index_array_type =
                typename RangePolicy<Dim, typename view_type::execution_space>::index_array_type;
            ippl::parallel_for(
                "HaloCells::pack()", getRangePolicy(subview),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    int l = 0;

                    for (unsigned d1 = 0; d1 < Dim; d1++) {
                        int next = args[d1];
                        for (unsigned d2 = 0; d2 < d1; d2++) {
                            next *= subview.extent(d2);
                        }
                        l += next;
                    }

                    buffer(l) = apply(subview, args);
                });
            Kokkos::fence();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <typename Op>
        void HaloCells<T, Dim, ViewArgs...>::unpack(const bound_type& range, const view_type& view,
                                                    databuffer_type& fd) {
            auto subview = makeSubview(view, range);
            auto buffer  = fd.buffer;

            // 29. November 2020
            // https://stackoverflow.com/questions/3735398/operator-as-template-parameter
            Op op;

            using index_array_type =
                typename RangePolicy<Dim, typename view_type::execution_space>::index_array_type;
            ippl::parallel_for(
                "HaloCells::unpack()", getRangePolicy(subview),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    int l = 0;

                    for (unsigned d1 = 0; d1 < Dim; d1++) {
                        int next = args[d1];
                        for (unsigned d2 = 0; d2 < d1; d2++) {
                            next *= subview.extent(d2);
                        }
                        l += next;
                    }

                    op(apply(subview, args), buffer(l));
                });
            Kokkos::fence();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        auto HaloCells<T, Dim, ViewArgs...>::makeSubview(const view_type& view,
                                                         const bound_type& intersect) {
            auto makeSub = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
                return Kokkos::subview(view,
                                       Kokkos::make_pair(intersect.lo[Idx], intersect.hi[Idx])...);
            };
            return makeSub(std::make_index_sequence<Dim>{});
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <typename Op>
        void HaloCells<T, Dim, ViewArgs...>::applyPeriodicSerialDim(view_type& view,
                                                                    const Layout_t* layout,
                                                                    const int nghost) {
            int myRank           = layout->comm.rank();
            const auto& lDomains = layout->getHostLocalDomains();
            const auto& domain   = layout->getDomain();

            using exec_space = typename view_type::execution_space;
            using index_type = typename RangePolicy<Dim, exec_space>::index_type;

            Kokkos::Array<index_type, Dim> ext, begin, end;

            for (size_t i = 0; i < Dim; ++i) {
                ext[i]   = view.extent(i);
                begin[i] = 0;
            }

            Op op;

            for (unsigned d = 0; d < Dim; ++d) {
                end    = ext;
                end[d] = nghost;

                if (lDomains[myRank][d].length() == domain[d].length()) {
                    int N = view.extent(d) - 1;

                    using index_array_type =
                        typename RangePolicy<Dim,
                                             typename view_type::execution_space>::index_array_type;
                    ippl::parallel_for(
                        "applyPeriodicSerialDim", createRangePolicy<Dim, exec_space>(begin, end),
                        KOKKOS_LAMBDA(index_array_type & coords) {
                            // The ghosts are filled starting from the inside
                            // of the domain proceeding outwards for both lower
                            // and upper faces. The extra brackets and explicit
                            // mention

                            // nghost + i
                            coords[d] += nghost;
                            auto&& left = apply(view, coords);

                            // N - nghost - i
                            coords[d]    = N - coords[d];
                            auto&& right = apply(view, coords);

                            // nghost - 1 - i
                            coords[d] += 2 * nghost - 1 - N;
                            op(apply(view, coords), right);

                            // N - (nghost - 1 - i) = N - (nghost - 1) + i
                            coords[d] = N - coords[d];
                            op(apply(view, coords), left);
                        });

                    Kokkos::fence();
                }
            }
        }
    }  // namespace detail
}  // namespace ippl
