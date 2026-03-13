//
// Class ParticleSpatialLayout
//   Particle layout based on spatial decomposition.
//
//   This is a specialized version of ParticleLayout, which places particles
//   on processors based on their spatial location relative to a fixed grid.
//   In particular, this can maintain particles on processors based on a
//   specified FieldLayout or RegionLayout, so that particles are always on
//   the same node as the node containing the Field region to which they are
//   local.  This may also be used if there is no associated Field at all,
//   in which case a grid is selected based on an even distribution of
//   particles among processors.
//
//   After each 'time step' in a calculation, which is defined as a period
//   in which the particle positions may change enough to affect the global
//   layout, the user must call the 'update' routine, which will move
//   particles between processors, etc.  After the Nth call to update, a
//   load balancing routine will be called instead.  The user may set the
//   frequency of load balancing (N), or may supply a function to
//   determine if load balancing should be done or not.
//
#include <memory>
#include <numeric>
#include <vector>

#include "Utility/IpplTimings.h"
#include "Utility/ParallelDispatch.h"

#include "Communicate/Window.h"

namespace ippl {

    /*!
     * We need this struct since Kokkos parallel_scan only accepts
     * one variable of type ReturnType where to perform the reduction operation.
     * For more details, see
     * https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/parallel_scan.html.
     */
    struct increment_type {
        size_t count[2];

        KOKKOS_FUNCTION void init() {
            count[0] = 0;
            count[1] = 0;
        }

        KOKKOS_INLINE_FUNCTION increment_type& operator+=(bool* values) {
            count[0] += values[0];
            count[1] += values[1];
            return *this;
        }

        KOKKOS_INLINE_FUNCTION increment_type& operator+=(increment_type values) {
            count[0] += values.count[0];
            count[1] += values.count[1];
            return *this;
        }
    };

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    ParticleSpatialLayout<T, Dim, Mesh, Properties...>::ParticleSpatialLayout(FieldLayout<Dim>& fl,
                                                                              Mesh& mesh, bool fem,
                                                                              CountExchange mode)
        : rlayout_m(std::make_shared<RegionLayout_t>(fl, mesh, fem))
        , flayout_m(fl)
        , countExchangeMode_(mode) {
        const int nRanks = Comm->size();
        nRanks_          = nRanks;
        initScratch(nRanks);

        if (mode == CountExchange::RMA && nRanks > 1) {
            nRecvs_m.resize(Comm->size());
            window_m.create(*Comm, nRecvs_m.begin(), nRecvs_m.end());
        }
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::updateLayout(FieldLayout<Dim>& fl,
                                                                          Mesh& mesh) {
        rlayout_m->changeDomain(fl, mesh);
        neighbors_dirty_ = true;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::countExchangeRMA() {
        std::fill(nRecvs_m.begin(), nRecvs_m.end(), 0);
        window_m.fence(0);

        for (int rank : destinationRanks_host_) {
            if (rank == Comm->rank())
                continue;
            const int* src = &rankSendCount_h_(rank);
            window_m.put<int>(src, rank, Comm->rank());
        }

        window_m.fence(0);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::countExchangeP2P() {
        const int myRank = Comm->rank();
        const int tag    = Comm->next_tag(mpi::tag::P_SPATIAL_LAYOUT, mpi::tag::P_LAYOUT_CYCLE);

        // Zero the receive-count buffer on the device
        Kokkos::deep_copy(position_execution_space{}, recvCounts_d_, 0);
        Kokkos::fence();

        // Device pointer to send-count array
        int* d_sendCounts = rankSendCount_d_.data();
        int* d_recvCounts = recvCounts_d_.data();

        std::vector<MPI_Request> reqs;
        reqs.reserve(2 * std::max(0, nRanks_ - 1));

        // Post receives from all ranks (except self)
        for (int r = 0; r < nRanks_; ++r) {
            if (r == myRank)
                continue;
            MPI_Irecv(d_recvCounts + r, 1, MPI_INT, r, tag, Comm->getCommunicator(),
                      &reqs.emplace_back());
        }

        for (int r = 0; r < nRanks_; ++r) {
            if (r == myRank)
                continue;
            MPI_Isend(d_sendCounts + r, 1, MPI_INT, r, tag, Comm->getCommunicator(),
                      &reqs.emplace_back());
        }

        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::countExchangeAlltoall() {
        MPI_Alltoall(rankSendCount_d_.data(), 1, MPI_INT, recvCounts_d_.data(), 1, MPI_INT,
                     Comm->getCommunicator());
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::update(ParticleContainer& pc) {
        /* Apply Boundary Conditions */
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        this->applyBC(pc.R, rlayout_m->getDomain());
        IpplTimings::stopTimer(ParticleBCTimer);

        /* Update Timer for the rest of the function */
        static IpplTimings::TimerRef ParticleUpdateTimer = IpplTimings::getTimer("updateParticle");
        IpplTimings::startTimer(ParticleUpdateTimer);

        if (nRanks_ < 2) {
            IpplTimings::stopTimer(ParticleUpdateTimer);
            return;
        }

        ensureNeighborsCached();

        /* particle MPI exchange:
         *   1. figure out which particles need to go where -> locateParticles(...)
         *   2. fill send buffer and send particles
         *   3. delete invalidated particles
         *   4. receive particles
         */

        // 1.  figure out which particles need to go where -> locateParticles(...) ============= //

        static IpplTimings::TimerRef locateTimer = IpplTimings::getTimer("locateParticles");
        IpplTimings::startTimer(locateTimer);

        const size_type nInvalid = locateParticlesPacked(pc);

//         // Fast-path: skip all MPI if every particle is already on the correct rank.
//         // A single allreduce suffices
//         {
//             size_type globalInvalid = 0;
//             Comm->allreduce(nInvalid, globalInvalid, 1, std::plus<size_type>());
//             if (globalInvalid == 0) {
//                 IpplTimings::stopTimer(locateTimer);
//                 IpplTimings::stopTimer(ParticleUpdateTimer);
// #ifndef NDEBUG
//                 debugCheckAllParticlesLocal(pc);
// #endif
//                 return;
//             }
//         }

        // Copy metadata to host
        size_type nDest = 0;
        Kokkos::deep_copy(position_execution_space{}, nDest, nDest_d_);

        // destRanks prefix
        if (nDest > 0) {
            Kokkos::deep_copy(
                position_execution_space{},
                Kokkos::subview(destRanks_h_, std::make_pair(size_t(0), size_t(nDest))),
                Kokkos::subview(destRanks_d_, std::make_pair(size_t(0), size_t(nDest))));
        }

        // counts + offsets
        Kokkos::deep_copy(position_execution_space{}, rankSendCount_h_, rankSendCount_d_);
        Kokkos::deep_copy(position_execution_space{}, sendOffsets_h_, sendOffsets_d_);

        // Build host destination list without allocation
        destinationRanks_host_.clear();
        for (size_t i = 0; i < (size_t)nDest; ++i)
            destinationRanks_host_.push_back(destRanks_h_(i));

        IpplTimings::stopTimer(locateTimer);

        // 2. fill send buffer and send particles =============================================== //

        // 2.1 Count Exchange

        static IpplTimings::TimerRef preprocTimer = IpplTimings::getTimer("sendPreprocess");
        IpplTimings::startTimer(preprocTimer);

        if (countExchangeMode_ == CountExchange::RMA) {
            countExchangeRMA();
        } else if (countExchangeMode_ == CountExchange::P2P_GPU) {
            countExchangeP2P();
        } else {
            countExchangeAlltoall();
        }

        IpplTimings::stopTimer(preprocTimer);

        // 2.2 Particle Sends

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        IpplTimings::startTimer(sendTimer);

        std::vector<MPI_Request> requests(0);
        requests.reserve(destinationRanks_host_.size());

        int tag = Comm->next_tag(mpi::tag::P_SPATIAL_LAYOUT, mpi::tag::P_LAYOUT_CYCLE);

        std::vector<std::pair<int, size_t>> sends;  // (rank, idx into destinationRanks_host_)
        for (size_t i = 0; i < destinationRanks_host_.size(); ++i) {
            int rank = destinationRanks_host_[i];
            if (rank == Comm->rank())
                continue;
            if (rankSendCount_h_(rank) == 0)
                continue;
            sends.push_back({rank, i});
        }

        // Pre-allocate requests
        std::vector<MPI_Request> newRequests(sends.size(), MPI_REQUEST_NULL);

        for (int i = 0; i < static_cast<int>(sends.size()); ++i) {
            auto [rank, idx]      = sends[i];
            const size_type begin = static_cast<size_type>(sendOffsets_h_(rank));
            const size_type count = static_cast<size_type>(rankSendCount_h_(rank));
            auto ids_sub =
                Kokkos::subview(sendIds_d_, std::make_pair((size_t)begin, (size_t)(begin + count)));
            newRequests[i] = pc.sendToRank(rank, tag, ids_sub);
        }

        requests.insert(requests.end(), newRequests.begin(), newRequests.end());
        IpplTimings::stopTimer(sendTimer);

        // 2.3 Post receives

        static IpplTimings::TimerRef recvTimer = IpplTimings::getTimer("particleRecv");
        IpplTimings::startTimer(recvTimer);

        std::vector<std::pair<int, size_type>> recvList;

        if (countExchangeMode_ == CountExchange::RMA) {
            for (int rank = 0; rank < nRanks_; ++rank) {
                if (nRecvs_m[rank] > 0) {
                    recvList.push_back({rank, nRecvs_m[rank]});
                }
            }
        } else {
            auto recvCounts_h =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recvCounts_d_);
            for (int rank = 0; rank < nRanks_; ++rank) {
                if (recvCounts_h(rank) > 0) {
                    recvList.push_back({rank, recvCounts_h(rank)});
                }
            }
        }

        std::vector<MPI_Request> recvRequests(recvList.size(), MPI_REQUEST_NULL);
        std::vector<std::function<void(size_type)>> finalizers(recvList.size());

        for (size_t i = 0; i < recvList.size(); ++i) {
            auto [rank, count] = recvList[i];
            auto [req, fin]    = pc.postRecvFromRank(rank, tag, count);
            recvRequests[i]    = req;
            finalizers[i]      = std::move(fin);
        }

        IpplTimings::stopTimer(recvTimer);

        // 3. Internal destruction of invalid particles ======================================= //

        static IpplTimings::TimerRef destroyTimer = IpplTimings::getTimer("particleDestroy");
        IpplTimings::startTimer(destroyTimer);

        const auto myRank = Comm->rank();

        auto neighbors_view =
            Kokkos::subview(neighbors_d_, std::make_pair(size_t(0), size_t(neighbors_used_)));

        auto positions           = pc.R.getView();
        region_view_type Regions = rlayout_m->getdLocalRegions();
        const auto is            = std::make_index_sequence<Dim>{};

        auto destRankOf = KOKKOS_LAMBDA(const size_t i) {
            if (positionInRegion(is, positions(i), Regions(myRank)))
                return myRank;

            for (size_t j = 0; j < neighbors_view.extent(0); ++j) {
                const int r = neighbors_view(j);
                if (positionInRegion(is, positions(i), Regions(r)))
                    return r;
            }

            for (int r = 0; r < static_cast<int>(Regions.extent(0)); ++r) {
                if (positionInRegion(is, positions(i), Regions(r)))
                    return r;
            }

            // Inclusive fallback: catches particles sitting exactly on a region
            // lower boundary that the strict > check above missed.
            for (int r = 0; r < static_cast<int>(Regions.extent(0)); ++r) {
                if (positionInRegionInclusive(is, positions(i), Regions(r)))
                    return r;
            }
            return myRank;  // truly outside all regions — applyBC should have prevented this
        };

        pc.template internalDestroy<position_memory_space, position_execution_space>(
            KOKKOS_LAMBDA(size_t i) { return destRankOf(i) != myRank; }, nInvalid);
        Kokkos::fence();

        IpplTimings::stopTimer(destroyTimer);

        requests.insert(requests.end(), recvRequests.begin(), recvRequests.end());
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }
        Comm->freeAllBuffers();

        // 5. Deserialize
        for (auto& finalize : finalizers)
            finalize(pc.getLocalNum());

        IpplTimings::stopTimer(ParticleUpdateTimer);
    }

    // ------------------------------------------------------------------ //
    // debugCheckAllParticlesLocal                                          //
    // ------------------------------------------------------------------ //
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::debugCheckAllParticlesLocal(
        const ParticleContainer& pc) const {
        const int myRank = Comm->rank();

        // Bring positions and regions to host for a straightforward loop.
        auto R_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pc.R.getView());
        auto regions_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rlayout_m->getdLocalRegions());

        const auto is_   = std::make_index_sequence<Dim>{};
        size_t misplaced = 0;

        for (size_t i = 0; i < pc.getLocalNum(); ++i) {
            if (!positionInRegion(is_, R_host(i), regions_host(myRank))) {
                ++misplaced;
                // Print first few mis-placed particles to aid diagnosis.
                if (misplaced <= 10) {
                    std::cerr << "[PSL debug] rank " << myRank << "  particle " << i << "  pos=(";
                    for (unsigned d = 0; d < Dim; ++d)
                        std::cerr << R_host(i)[d] << (d + 1 < Dim ? ", " : "");
                    std::cerr << ")  local_region=[";
                    for (unsigned d = 0; d < Dim; ++d)
                        std::cerr << "(" << regions_host(myRank)[d].min() << ","
                                  << regions_host(myRank)[d].max() << ")"
                                  << (d + 1 < Dim ? "x" : "");
                    std::cerr << "]\n";
                }
            }
        }

        if (misplaced > 0) {
            std::cerr << "[PSL debug] rank " << myRank << ": " << misplaced << " of "
                      << pc.getLocalNum() << " particles outside local region after update()\n";
        }

        assert(misplaced == 0 && "ParticleSpatialLayout::update(): particles outside local region");
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialLayout<T, Dim, Mesh, Properties...>::positionInRegionInclusive(
        const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region) {
        return ((pos[Idx] >= region[Idx].min()) && ...) && ((pos[Idx] <= region[Idx].max()) && ...);
    };

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialLayout<T, Dim, Mesh, Properties...>::positionInRegion(
        const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region) {
        return ((pos[Idx] > region[Idx].min()) && ...) && ((pos[Idx] <= region[Idx].max()) && ...);
    };

    /* Helper function that evaluates the total number of neighbors for the current rank in Dim
     * dimensions.
     */
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    detail::size_type ParticleSpatialLayout<T, Dim, Mesh, Properties...>::getNeighborSize(
        const neighbor_list& neighbors) const {
        size_type totalSize = 0;

        for (const auto& componentNeighbors : neighbors) {
            totalSize += componentNeighbors.size();
        }

        return totalSize;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <typename ParticleContainer>
    size_t ParticleSpatialLayout<T, Dim, Mesh, Properties...>::locateParticlesPacked(
        const ParticleContainer& pc) {
        const int nRanks       = Comm->size();
        const size_type myRank = Comm->rank();

        ensureNeighborsCached();

        auto positions           = pc.R.getView();
        region_view_type Regions = rlayout_m->getdLocalRegions();
        const auto is            = std::make_index_sequence<Dim>{};

        using exec_space  = position_execution_space;
        using policy_type = Kokkos::RangePolicy<size_t, exec_space>;

        // Reset small device buffers
        Kokkos::deep_copy(rankSendCount_d_, size_type(0));
        Kokkos::deep_copy(cursor_d_, size_type(0));
        Kokkos::deep_copy(nDest_d_, size_type(0));

        const size_type neighbors_used = neighbors_used_;
        auto& neighbours_d             = neighbors_d_;

        // Destination rank computation (no per-particle storage)
        auto destRankOf = KOKKOS_LAMBDA(const size_t i)->size_type {
            if (positionInRegion(is, positions(i), Regions(myRank)))
                return myRank;

            for (int j = 0; j < static_cast<int>(neighbors_used); ++j) {
                const int r = neighbours_d(j);
                if (positionInRegion(is, positions(i), Regions(r)))
                    return r;
            }

            // slow-path: global scan
            for (int r = 0; r < static_cast<int>(Regions.extent(0)); ++r) {
                if (positionInRegion(is, positions(i), Regions(r)))
                    return r;
            }

            // Inclusive fallback: catches particles sitting exactly on a region
            // lower boundary that the strict > check above missed (e.g. (0,0,0))
            for (int r = 0; r < static_cast<int>(Regions.extent(0)); ++r) {
                if (positionInRegionInclusive(is, positions(i), Regions(r)))
                    return r;
            }
            return myRank;  // truly outside all regions — applyBC should have prevented this
        };

        // Pass 1: compute send counts + nInvalid
        size_type nInvalid    = 0;
        auto& rankSendCount_d = rankSendCount_d_;
        Kokkos::parallel_reduce(
            "PSL::packed_count", policy_type(0, pc.getLocalNum()),
            KOKKOS_LAMBDA(const size_t i, size_type& inval) {
                const size_type dest = destRankOf(i);
                const bool leaves    = (dest != myRank);
                inval += leaves;
                if (leaves)
                    Kokkos::atomic_fetch_add(&rankSendCount_d(dest), size_type(1));
            },
            nInvalid);
        Kokkos::fence();

        // Ensure sendIds capacity after we know nInvalid
        ensureSendCapacity(nInvalid);

        // Exclusive scan to offsets (length nRanks+1)
        auto& sendOffsets_d = sendOffsets_d_;
        Kokkos::parallel_scan(
            "PSL::packed_offsets", policy_type(0, (size_t)nRanks + 1),
            KOKKOS_LAMBDA(const size_t r, size_type& upd, const bool final) {
                if (final)
                    sendOffsets_d(r) = upd;
                if (r < (size_t)nRanks)
                    upd += rankSendCount_d(r);
            });
        Kokkos::fence();

        // Pass 2: fill packed send IDs into sendIds_d_ (prefix [0, nInvalid))
        auto& cursor_d  = cursor_d_;
        auto& sendIds_d = sendIds_d_;
        Kokkos::parallel_for(
            "PSL::packed_fill", policy_type(0, pc.getLocalNum()), KOKKOS_LAMBDA(const size_t i) {
                const size_type dest = destRankOf(i);
                if (dest == myRank)
                    return;

                const size_type pos  = Kokkos::atomic_fetch_add(&cursor_d(dest), size_type(1));
                const size_type base = sendOffsets_d(dest);

                sendIds_d(base + pos) = static_cast<typename hash_type::non_const_value_type>(i);
            });
        Kokkos::fence();

        // Build destination rank list on device (compact), store length in nDest_d_
        auto& destRanks_d = destRanks_d_;
        auto& nDest_d     = nDest_d_;
        Kokkos::parallel_for(
            "PSL::packed_destRanks", policy_type(0, (size_t)nRanks), KOKKOS_LAMBDA(const size_t r) {
                if ((size_type)r == myRank)
                    return;
                if (rankSendCount_d(r) > 0) {
                    const size_type idx = Kokkos::atomic_fetch_add(&nDest_d(), size_type(1));
                    destRanks_d(idx)    = static_cast<int>(r);
                }
            });
        Kokkos::fence();

        return nInvalid;
    }

    /**
     * @brief This function determines to which rank particles need to be sent after the iteration
     * step. It starts by first scanning direct rank neighbors, and only does a global scan if there
     * are still unfound particles. It then calculates how many particles need to be sent to each
     * rank and how many ranks are sent to in total.
     *
     * @param pc           Particle Container
     * @param ranks        A vector the length of the number of particles on the current rank, where
     * each value refers to the new rank of the particle
     * @param invalid      A vector marking the particles that need to be sent away, and thus
     * locally deleted
     * @param nSends_dview Device view the length of number of ranks, where each value determines
     * the number of particles sent to that rank from the current rank
     * @param sends_dview  Device view for the number of ranks that are sent to from current rank
     *
     * @return tuple with the number of particles sent away and the number of ranks sent to
     */
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <typename ParticleContainer>
    std::pair<detail::size_type, detail::size_type>
    ParticleSpatialLayout<T, Dim, Mesh, Properties...>::locateParticles(
        const ParticleContainer& pc, locate_type& ranks, locate_type& nSends_dview,
        locate_type& sends_dview) const {
        auto positions           = pc.R.getView();
        region_view_type Regions = rlayout_m->getdLocalRegions();

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        using policy_type  = Kokkos::RangePolicy<size_t, position_execution_space>;

        size_type myRank = Comm->rank();

        const auto is = std::make_index_sequence<Dim>{};

        const neighbor_list& neighbors = flayout_m.getNeighbors();
        const size_type neighborSize   = getNeighborSize(neighbors);

        locate_type neighbors_view("Nearest neighbors IDs", neighborSize);

        increment_type red_val;
        red_val.init();

        auto neighbors_mirror =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), neighbors_view);

        size_t k = 0;
        for (const auto& componentNeighbors : neighbors) {
            for (size_t j = 0; j < componentNeighbors.size(); ++j) {
                neighbors_mirror(k) = componentNeighbors[j];
                k++;
            }
        }

        Kokkos::deep_copy(neighbors_view, neighbors_mirror);

        /// outsideCount: Tracks the number of particles that travelled outside of the neighborhood.
        size_type outsideCount = 0;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::locateParticles() outside count",
            policy_type(0, pc.getLocalNum()),
            KOKKOS_LAMBDA(const size_t i, size_type& cnt) {
                bool found = positionInRegion(is, positions(i), Regions(myRank));
                for (size_t j = 0; j < neighbors_view.extent(0) && !found; ++j) {
                    found = positionInRegion(is, positions(i), Regions(neighbors_view(j)));
                }
                cnt += !found;
            },
            outsideCount);

        /// outsideIds: Container of particle IDs that travelled outside of the neighborhood.
        locate_type outsideIds("Particles outside of neighborhood", outsideCount);

        /// invalidCount: Tracks the number of particles that need to be sent to other ranks.
        size_type invalidCount = 0;

        /*! Begin Kokkos loop:
         * Step 1: search in current rank
         * Step 2: search in neighbors
         * Step 3: save information on whether the particle was located
         * Step 4: run additional loop on non-located particles
         */
        static IpplTimings::TimerRef neighborSearch = IpplTimings::getTimer("neighborSearch");
        IpplTimings::startTimer(neighborSearch);

        Kokkos::parallel_scan(
            "ParticleSpatialLayout::locateParticles()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_type i, increment_type& val, const bool final) {
                /* Step 1
                 * inCurr: True if the particle hasn't left the current MPI rank.
                 * inNeighbor: True if the particle is found in a neighboring rank.
                 * found: True either if inCurr = True or inNeighbor = True.
                 * increment: Helper variable to update red_val.
                 */
                bool inCurr     = false;
                bool inNeighbor = false;
                bool found      = false;
                bool increment[2];

                inCurr = positionInRegion(is, positions(i), Regions(myRank));

                ranks(i) = inCurr * myRank;
                found    = inCurr || found;

                /// Step 2
                for (size_t j = 0; j < neighbors_view.extent(0); ++j) {
                    size_type rank = neighbors_view(j);

                    inNeighbor = positionInRegion(is, positions(i), Regions(rank));

                    ranks(i) = !(inNeighbor)*ranks(i) + inNeighbor * rank;
                    found    = inNeighbor || found;
                }
                /// Step 3
                /* isOut: When the last thread has finished the search, checks whether the particle
                 * has been found either in the current rank or in a neighboring one. Used to avoid
                 * race conditions when updating outsideIds.
                 */
                if (final && !found) {
                    outsideIds(val.count[1]) = i;
                }
                // outsideIds(val.count[1]) = i * isOut;
                increment[0] = !inCurr;
                increment[1] = !found;
                val += increment;
            },
            red_val);

        Kokkos::fence();

        invalidCount = red_val.count[0];
        outsideCount = red_val.count[1];

        IpplTimings::stopTimer(neighborSearch);

        /// Step 4
        static IpplTimings::TimerRef nonNeighboringParticles =
            IpplTimings::getTimer("nonNeighboringParticles");
        IpplTimings::startTimer(nonNeighboringParticles);
        if (outsideCount > 0) {
            Kokkos::parallel_for(
                "ParticleSpatialLayout::leftParticles()",
                mdrange_type({0, 0}, {outsideCount, Regions.extent(0)}),
                KOKKOS_LAMBDA(const size_t i, const size_type j) {
                    /// pID: (local) ID of the particle that is currently being searched.
                    size_type pId = outsideIds(i);

                    /// inRegion: Checks whether particle pID is inside region j.
                    bool inRegion = positionInRegion(is, positions(pId), Regions(j));
                    if (inRegion) {
                        ranks(pId) = j;
                    }
                });
            Kokkos::fence();
        }

        IpplTimings::stopTimer(nonNeighboringParticles);

        Kokkos::parallel_for(
            "Calculate nSends", policy_type(0, ranks.extent(0)), KOKKOS_LAMBDA(const size_t i) {
                size_type rank = ranks(i);
                Kokkos::atomic_fetch_add(&nSends_dview(rank), 1);
            });

        // Number of Ranks we need to send to
        Kokkos::View<size_type, position_memory_space> rankSends(
            "Number of Ranks we need to send to");

        Kokkos::parallel_for(
            "Calculate sends", policy_type(0, nSends_dview.extent(0)),
            KOKKOS_LAMBDA(const size_t rank) {
                if (nSends_dview(rank) != 0) {
                    size_type index    = Kokkos::atomic_fetch_add(&rankSends(), 1);
                    sends_dview(index) = rank;
                }
            });
        size_type temp;
        Kokkos::deep_copy(temp, rankSends);

        return {invalidCount, temp};
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::fillHash(int rank,
                                                                      const locate_type& ranks,
                                                                      hash_type& hash) {
        /* Compute the prefix sum and fill the hash
         */
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::fillHash()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final) {
                    if (rank == ranks(i)) {
                        hash(idx) = i;
                    }
                }

                if (rank == ranks(i)) {
                    idx += 1;
                }
            });
        Kokkos::fence();
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    size_t ParticleSpatialLayout<T, Dim, Mesh, Properties...>::numberOfSends(
        int rank, const locate_type& ranks) {
        size_t nSends     = 0;
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::numberOfSends()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, size_t& num) { num += size_t(rank == ranks(i)); },
            nSends);
        Kokkos::fence();
        return nSends;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::initScratch(int nRanks) {
        Kokkos::realloc(rankSendCount_d_, nRanks);
        Kokkos::realloc(sendOffsets_d_, nRanks + 1);
        Kokkos::realloc(cursor_d_, nRanks);
        Kokkos::realloc(destRanks_d_, nRanks);
        Kokkos::realloc(recvCounts_d_, nRanks);

        // scalar counter
        nDest_d_ = Kokkos::View<size_type, position_memory_space>("nDest_d");

        // Host mirrors
        Kokkos::realloc(rankSendCount_h_, nRanks);
        Kokkos::realloc(sendOffsets_h_, nRanks + 1);
        Kokkos::realloc(destRanks_h_, nRanks);

        destinationRanks_host_.clear();
        destinationRanks_host_.reserve(nRanks);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::ensureSendCapacity(size_t nInvalid) {
        if (nInvalid <= sendIds_capacity_)
            return;

        // grow geometrically
        size_t newCap = sendIds_capacity_ ? sendIds_capacity_ : size_t(1024);
        while (newCap < nInvalid)
            newCap *= 2;

        sendIds_capacity_ = newCap;
        Kokkos::realloc(sendIds_d_, newCap);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::ensureNeighborsCached() {
        if (!neighbors_dirty_)
            return;

        const neighbor_list& neighbors = flayout_m.getNeighbors();
        const size_type neighborSize   = getNeighborSize(neighbors);
        neighbors_used_                = neighborSize;

        if (neighborSize > neighbors_capacity_) {
            neighbors_capacity_ = neighborSize;
            Kokkos::realloc(neighbors_d_, neighbors_capacity_);
        }

        neighbors_host_.clear();
        neighbors_host_.reserve(neighborSize);

        auto neighbors_h = Kokkos::create_mirror_view(neighbors_d_);
        size_t k         = 0;
        for (const auto& comp : neighbors) {
            for (size_t j = 0; j < comp.size(); ++j) {
                neighbors_h(k) = comp[j];
                neighbors_host_.push_back(comp[j]);
                ++k;
            }
        }

        Kokkos::deep_copy(
            Kokkos::subview(neighbors_d_, std::make_pair(size_t(0), size_t(neighborSize))),
            Kokkos::subview(neighbors_h, std::make_pair(size_t(0), size_t(neighborSize))));

        neighbors_dirty_ = false;
    }

}  // namespace ippl
