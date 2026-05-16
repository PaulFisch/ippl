//
// Class HaloCells
//   The guard / ghost cells of BareField.
//

#include <memory>
#include <vector>
#include <chrono>

#include "Utility/IpplException.h"
#include "Utility/ParallelDispatch.h"

#include "Communicate/Communicator.h"

// Compile-time-toggled deadlock instrumentation. Two independent switches
// so we can turn the trace on without changing inter-rank timing:
//
//   cmake -DIPPL_HALO_LOG=ON  ...  enables IPPL_HALO_LOG()
//   cmake -DIPPL_HALO_SYNC=ON ...  enables the MPI_Barrier in IPPL_HALO_SYNC()
//
// Typical hang-hunt: HALO_LOG=ON, HALO_SYNC=OFF (log without altering timing).
//
// Speedups vs. the original single-flag instrumentation:
//   - stderr is reconfigured to fully-buffered (1 MiB) once at the first
//     log call, so we no longer pay a write() syscall per line.
//   - the per-call ostringstream is thread_local and reused; it grows once
//     and stays at that size.
//   - we drop the per-line fflush; instead we flush every 64 lines and on
//     fatal signals, so a hang loses at most ~64 trace lines per rank
//     (smaller window than the original 256 because the last few dozen
//     lines are exactly what tells us which MPI op blocked).
//   - the atomic seq counter is a single CAS per call.
//
// Flushing on termination: we install SIGTERM/SIGINT/SIGHUP handlers that
// drain stderr before the process dies. slurm sends SIGTERM with a grace
// period before SIGKILL, so the handler reliably catches a timed-out job.
// SIGUSR1 also forces a flush, so `kill -USR1 <pid>` from another shell
// captures the live state of a stuck rank without killing it.
// Host-side only (no GPU prints).
#if defined(IPPL_HALO_LOG_ENABLE) || defined(IPPL_HALO_SYNC_ENABLE)
#include <cstdio>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <mpi.h>
namespace ippl { namespace detail {
    inline std::atomic<unsigned long>& haloDebugSeq() {
        static std::atomic<unsigned long> s{0};
        return s;
    }
    // Async-signal-safe (or close enough): fflush(NULL) flushes all open
    // streams. The C standard does not guarantee fflush is signal-safe, but
    // glibc's implementation is in practice safe for our case (no nested
    // signal, single writer). The worst case is a malformed last line, not
    // a deadlocked flush. Re-raise so the default action still terminates
    // the process for fatal signals.
    inline void haloDebugFatalSignal(int sig) {
        std::fflush(NULL);
        std::signal(sig, SIG_DFL);
        std::raise(sig);
    }
    inline void haloDebugUsrSignal(int /*sig*/) {
        std::fflush(NULL);  // SIGUSR1: flush and keep running.
    }
    // Reconfigure stderr once. Without this stderr is unbuffered by C
    // default and every log line costs a write() syscall, which is what
    // made HALO_DEBUG slow enough to mask races. Also install signal
    // handlers so the buffer drains on graceful termination.
    inline void haloDebugInit() {
        static std::once_flag flag;
        std::call_once(flag, []{
            static char buf[1u << 20];  // 1 MiB
            std::setvbuf(stderr, buf, _IOFBF, sizeof(buf));
            std::atexit([]{ std::fflush(NULL); });
            std::signal(SIGTERM, haloDebugFatalSignal);
            std::signal(SIGINT,  haloDebugFatalSignal);
            std::signal(SIGHUP,  haloDebugFatalSignal);
            std::signal(SIGUSR1, haloDebugUsrSignal);
        });
    }
    inline void haloDebugMaybeFlush() {
        static std::atomic<unsigned long> count{0};
        // 0x3F -> flush every 64 lines (~6 KB per rank). Small enough that
        // a kill -9 loses at most ~64 lines, large enough that the fwrite
        // path stays cheap.
        if ((count.fetch_add(1, std::memory_order_relaxed) & 0x3Fu) == 0x3Fu) {
            std::fflush(stderr);
        }
    }
}}  // namespace ippl::detail
#endif

#ifdef IPPL_HALO_LOG_ENABLE
// fprintf is thread-safe on POSIX; the formatting happens in a thread_local
// ostringstream so we don't allocate on every call.
#define IPPL_HALO_LOG(stream)                                                              \
    do {                                                                                   \
        ippl::detail::haloDebugInit();                                                     \
        thread_local std::ostringstream _ipplhalolog_oss;                                  \
        _ipplhalolog_oss.str(std::string{});                                               \
        _ipplhalolog_oss.clear();                                                          \
        _ipplhalolog_oss << "[HALO/r=" << ippl::Comm->rank()                               \
                         << "][seq=" << ippl::detail::haloDebugSeq().fetch_add(1) << "]"   \
                         << stream << '\n';                                                \
        const std::string& _ipplhalolog_s = _ipplhalolog_oss.str();                        \
        std::fwrite(_ipplhalolog_s.data(), 1, _ipplhalolog_s.size(), stderr);              \
        ippl::detail::haloDebugMaybeFlush();                                               \
    } while (0)
#else
#define IPPL_HALO_LOG(stream) do {} while (0)
#endif

#ifdef IPPL_HALO_SYNC_ENABLE
// If LOG is also on, the macro is HALO_LOG -> barrier -> HALO_LOG so the
// trace brackets the barrier. If LOG is off, this is just a bare barrier.
#define IPPL_HALO_SYNC(label)                                                              \
    do {                                                                                   \
        IPPL_HALO_LOG("sync:pre id=" << label);                                            \
        MPI_Barrier(ippl::Comm->getCommunicator());                                        \
        IPPL_HALO_LOG("sync:post id=" << label);                                           \
    } while (0)
#else
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

            using memory_space = typename view_type::memory_space;
            using buffer_type  = mpi::Communicator::buffer_type<memory_space>;
            constexpr size_t cubeCount = detail::countHypercubes(Dim) - 1;

            // Post all Irecvs first, then all Isends, then a single Waitall
            // over both sides. The previous "all Isends, then all blocking
            // Recvs" pattern is standard-conformant, but on GPU-aware Cray
            // MPICH + UCX we observed individual MPI_Isend calls blocking
            // for >145 s when the implementation's send-side rendezvous /
            // registration-cache resources fill before any matching receive
            // is posted. Posting Irecvs first gives the implementation
            // matching receives to clear those resources against, and is the
            // recommended pattern in both the Cray and UCX docs.
            std::vector<MPI_Request> recvRequests;
            std::vector<buffer_type> recvBuffers;
            std::vector<bound_type>  recvRangeList;
            std::vector<size_type>   recvSizes;
            recvRequests.reserve(totalRequests);
            recvBuffers.reserve(totalRequests);
            recvRangeList.reserve(totalRequests);
            recvSizes.reserve(totalRequests);

            // Irecv loop: allocate a unique buffer per neighbor and post the
            // Irecv. Deserialization into haloData_m is deferred to after
            // Waitall because haloData_m is a single scratch buffer that
            // gets overwritten by each deserialize/unpack pair.
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

                    MPI_Request req;
                    comm.irecv(sourceRank, tag, *buf, req, nrecvs * sizeof(T));

                    recvRequests.push_back(req);
                    recvBuffers.push_back(buf);
                    recvRangeList.push_back(range);
                    recvSizes.push_back(nrecvs);
                }
            }

            // Isend loop: same pack/serialize/Isend pattern as before. The
            // pack writes into haloData_m; isend serializes haloData_m into
            // a per-call buffer before launching MPI_Isend, so reusing
            // haloData_m across send iterations is safe.
            std::vector<MPI_Request> sendRequests(totalRequests);
            size_t sendIndex = 0;
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

                    comm.isend(targetRank, tag, haloData_m, *buf, sendRequests[sendIndex++], nsends);
                    buf->resetWritePos();
                }
            }

            if (totalRequests > 0) {
                MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUSES_IGNORE);
                MPI_Waitall(recvRequests.size(), recvRequests.data(), MPI_STATUSES_IGNORE);
            }

            // Deserialize + unpack each recv buffer in order. haloData_m is
            // overwritten by each pair, so the loop must be sequential.
            for (size_t k = 0; k < recvBuffers.size(); ++k) {
                haloData_m.deserialize(*recvBuffers[k], recvSizes[k]);
                recvBuffers[k]->resetReadPos();
                unpack<Op>(recvRangeList[k], view, haloData_m);
            }

            comm.freeAllBuffers();
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
