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
                                                                              Mesh& mesh, bool fem)
        : rlayout_m(std::make_shared<RegionLayout_t>(fl, mesh, fem))
        , flayout_m(fl) {
        nRecvs_m.resize(Comm->size());
        if (Comm->size() > 1) {
            window_m.create(*Comm, nRecvs_m.begin(), nRecvs_m.end());
        }
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::updateLayout(FieldLayout<Dim>& fl,
                                                                          Mesh& mesh) {
        // flayout_m = fl;
        rlayout_m->changeDomain(fl, mesh);
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

        int nRanks = Comm->size();
        if (nRanks < 2) {
            return;
        }

        /* particle MPI exchange:
         *   1. figure out which particles need to go where -> locateParticles(...)
         *   2. fill send buffer and send particles
         *   3. delete invalidated particles
         *   4. receive particles
         */

        // 1.  figure out which particles need to go where -> locateParticles(...) ============= //

        static IpplTimings::TimerRef locateTimer = IpplTimings::getTimer("locateParticles");
        IpplTimings::startTimer(locateTimer);

        /* The indices are the MPI ranks,
         * the values are the number of particles are sent to that rank from myrank
         */
        locate_type rankSendCount_dview("rankSendCount Device", nRanks);
        locate_type sendOffsets_dview("rankSendCount Device", nRanks);
        hash_type sendIds_dview;

        Kokkos::deep_copy(rankSendCount_dview, size_type(0));

        /* nInvalid is the number of invalid particles
         * nDestinationRanks is the number of MPI ranks we need to send to
         */
        auto [nInvalid, destinationRanks_h] =
            locateParticlesPacked(pc, rankSendCount_dview, sendOffsets_dview, sendIds_dview);

        // Host copies for slicing
        auto rankSendCount_hview =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rankSendCount_dview);
        auto sendOffsets_hview =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sendOffsets_dview);

        IpplTimings::stopTimer(locateTimer);

        // 2. fill send buffer and send particles =============================================== //

        // 2.1 Remote Memory Access window for one-sided communication

        static IpplTimings::TimerRef preprocTimer = IpplTimings::getTimer("sendPreprocess");
        IpplTimings::startTimer(preprocTimer);

        std::fill(nRecvs_m.begin(), nRecvs_m.end(), 0);

        window_m.fence(0);

        // Prepare RMA window for the ranks we need to send to
        for (int rank : destinationRanks_h) {
            if (rank == Comm->rank())
                continue;
            const int* src_ptr = &rankSendCount_hview(rank);
            window_m.put<int>(src_ptr, rank, Comm->rank());
        }
        window_m.fence(0);

        IpplTimings::stopTimer(preprocTimer);

        // 2.2 Particle Sends

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        IpplTimings::startTimer(sendTimer);

        std::vector<MPI_Request> requests(0);
        requests.reserve(destinationRanks_h.size());

        int tag = Comm->next_tag(mpi::tag::P_SPATIAL_LAYOUT, mpi::tag::P_LAYOUT_CYCLE);

        for (int rank : destinationRanks_h) {
            if (rank == Comm->rank())
                continue;

            const size_type count = rankSendCount_hview(rank);
            if (count == 0) {
                continue;
            }

            const size_type begin = sendOffsets_hview(rank);
            auto ids_sub = Kokkos::subview(sendIds_dview, std::make_pair(begin, begin + count));

            pc.sendToRank(rank, tag, requests, ids_sub);
        }

        IpplTimings::stopTimer(sendTimer);

        // 3. Internal destruction of invalid particles ======================================= //

        static IpplTimings::TimerRef destroyTimer = IpplTimings::getTimer("particleDestroy");
        IpplTimings::startTimer(destroyTimer);

        const auto myRank = Comm->rank();

        const neighbor_list& neighbors = flayout_m.getNeighbors();
        const size_type neighborSize   = getNeighborSize(neighbors);
        locate_type neighbors_view("Nearest neighbors IDs", neighborSize);
        {
            auto neighbors_mirror = Kokkos::create_mirror_view(neighbors_view);
            size_t k              = 0;
            for (const auto& componentNeighbors : neighbors) {
                for (size_t j = 0; j < componentNeighbors.size(); ++j) {
                    neighbors_mirror(k++) = componentNeighbors[j];
                }
            }
            Kokkos::deep_copy(neighbors_view, neighbors_mirror);
        }

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

            for (int r = 0; r < Regions.extent(0); ++r) {
                if (positionInRegion(is, positions(i), Regions(r)))
                    return r;
            }

            // Policy if outside all regions
            return myRank;
        };

        pc.template internalDestroy<position_memory_space, position_execution_space>(
            KOKKOS_LAMBDA(size_t i) { return destRankOf(i) != myRank; }, nInvalid);
        Kokkos::fence();

        IpplTimings::stopTimer(destroyTimer);

        // 4. Receive Particles ================================================================ //

        static IpplTimings::TimerRef recvTimer = IpplTimings::getTimer("particleRecv");
        IpplTimings::startTimer(recvTimer);

        for (int rank = 0; rank < nRanks; ++rank) {
            if (nRecvs_m[rank] > 0) {
                pc.recvFromRank(rank, tag, nRecvs_m[rank]);
            }
        }
        IpplTimings::stopTimer(recvTimer);

        // IpplTimings::startTimer(sendTimer);

        if (requests.size() > 0) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        Comm->freeAllBuffers();
        // IpplTimings::stopTimer(sendTimer);

        IpplTimings::stopTimer(ParticleUpdateTimer);
    }

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
    std::pair<detail::size_type, std::vector<int>>
    ParticleSpatialLayout<T, Dim, Mesh, Properties...>::locateParticlesPacked(
        const ParticleContainer& pc, locate_type& rankSendCount_dview,
        locate_type& sendOffsets_dview, hash_type& sendIds_dview) const {
        auto positions           = pc.R.getView();
        region_view_type Regions = rlayout_m->getdLocalRegions();

        using exec_space  = position_execution_space;
        using policy_type = Kokkos::RangePolicy<size_t, exec_space>;

        const size_type myRank = Comm->rank();
        const int nRanks       = Comm->size();
        const auto is          = std::make_index_sequence<Dim>{};

        // neighbors_view (build on host then deep_copy)
        const neighbor_list& neighbors = flayout_m.getNeighbors();
        const size_type neighborSize   = getNeighborSize(neighbors);
        locate_type neighbors_view("Nearest neighbors IDs", neighborSize);
        {
            auto neighbors_mirror = Kokkos::create_mirror_view(neighbors_view);
            size_t k              = 0;
            for (const auto& componentNeighbors : neighbors) {
                for (size_t j = 0; j < componentNeighbors.size(); ++j) {
                    neighbors_mirror(k++) = componentNeighbors[j];
                }
            }
            Kokkos::deep_copy(neighbors_view, neighbors_mirror);
        }

        // Local helper: destination rank for particle i
        auto destRankOf = KOKKOS_LAMBDA(const size_t i)->size_type {
            if (positionInRegion(is, positions(i), Regions(myRank)))
                return myRank;

            for (size_t j = 0; j < neighbors_view.extent(0); ++j) {
                const size_type r = neighbors_view(j);
                if (positionInRegion(is, positions(i), Regions(r)))
                    return r;
            }

            for (size_type r = 0; r < Regions.extent(0); ++r) {
                if (positionInRegion(is, positions(i), Regions(r)))
                    return r;
            }

            return myRank;
        };

        // Pass 1: counts + nInvalid
        size_type nInvalid = 0;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::locateParticlesPacked count", policy_type(0, pc.getLocalNum()),
            KOKKOS_LAMBDA(const size_t i, size_type& inval) {
                const size_type dest = destRankOf(i);
                const bool leaves    = (dest != myRank);
                inval += leaves;

                if (leaves) {
                    Kokkos::atomic_fetch_add(&rankSendCount_dview(dest), size_type(1));
                }
            },
            nInvalid);
        Kokkos::fence();

        // Host counts for destination list (also used by update() for window put)
        auto rankSendCount_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rankSendCount_dview);

        std::vector<int> destinationRanks_h;
        destinationRanks_h.reserve(nRanks);
        for (int r = 0; r < nRanks; ++r) {
            if (r == (int)myRank)
                continue;
            if (rankSendCount_h(r) > 0)
                destinationRanks_h.push_back(r);
        }

        // Offsets: exclusive scan over ranks; sendOffsets size nRanks+1
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::locateParticlesPacked offsets",
            policy_type(0, (size_t)nRanks + 1),
            KOKKOS_LAMBDA(const size_t r, size_type& upd, const bool final) {
                if (final)
                    sendOffsets_dview(r) = upd;
                if (r < (size_t)nRanks)
                    upd += rankSendCount_dview(r);
            });
        Kokkos::fence();

        // Allocate packed IDs
        sendIds_dview = hash_type("sendIdsPacked", nInvalid);

        // Cursor per rank for fill
        locate_type cursor_dview("sendFillCursor", nRanks);
        Kokkos::deep_copy(cursor_dview, size_type(0));

        // Pass 2: fill sendIds grouped by dest rank
        Kokkos::parallel_for(
            "ParticleSpatialLayout::locateParticlesPacked fill", policy_type(0, pc.getLocalNum()),
            KOKKOS_LAMBDA(const size_t i) {
                const size_type dest = destRankOf(i);
                if (dest == myRank)
                    return;

                const size_type pos  = Kokkos::atomic_fetch_add(&cursor_dview(dest), size_type(1));
                const size_type base = sendOffsets_dview(dest);
                sendIds_dview(base + pos) = i;
            });
        Kokkos::fence();

        return {nInvalid, destinationRanks_h};
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

        /// neighborSize: Size of a neighborhood in D dimentions.
        const size_type neighborSize = getNeighborSize(neighbors);

        /// neighbors_view: Kokkos view with the IDs of the neighboring MPI ranks.
        locate_type neighbors_view("Nearest neighbors IDs", neighborSize);

        /* red_val: Used to reduce both the number of invalid particles and the number of particles
         * outside of the neighborhood (Kokkos::parallel_scan doesn't allow multiple reduction
         * values, so we use the helper class increment_type). First element updates InvalidCount,
         * second one updates outsideCount.
         */
        increment_type red_val;
        red_val.init();

        auto neighbors_mirror =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), neighbors_view);

        size_t k = 0;

        for (const auto& componentNeighbors : neighbors) {
            for (size_t j = 0; j < componentNeighbors.size(); ++j) {
                neighbors_mirror(k) = componentNeighbors[j];
                // std::cout << "Neighbor: " << neighbors_mirror(k) << std::endl;
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

}  // namespace ippl
