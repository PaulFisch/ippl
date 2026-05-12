//   Class BConds
//   This is the container class for the field BCs.
//   It calls the findBCNeighbors and apply in the
//   respective BC classes to apply field BCs
//
namespace ippl {
    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::write(std::ostream& os) const {
        os << "BConds: (" << std::endl;
        const_iterator it = bc_m.begin();
        for (; it != bc_m.end() - 1; ++it) {
            (*it)->write(os);
            os << "," << std::endl;
        }
        (*it)->write(os);
        os << std::endl << ")";
    }

    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::findBCNeighbors(Field& field) {
        for (auto& bc : bc_m) {
            bc->findBCNeighbors(field);
        }
        Kokkos::fence();
        field.getCommunicator().barrier();
    }

    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::apply(Field& field) {
#ifdef IPPL_HALO_DEBUG
        // Identify the field by its data view pointer so we can correlate the
        // BC apply with the field that triggered it across ranks.
        char _bcapplybuf[256];
        std::snprintf(_bcapplybuf, sizeof(_bcapplybuf),
                      "[HALO/r=%d][seq=%lu][BConds:apply][nfaces=%zu][view=%p]\n",
                      ippl::Comm->rank(),
                      ippl::detail::haloDebugSeq().fetch_add(1),
                      bc_m.size(),
                      reinterpret_cast<const void*>(field.getView().data()));
        std::fprintf(stderr, "%s", _bcapplybuf);
        std::fflush(stderr);
#endif
        IPPL_HALO_SYNC("BConds:enter");
        std::size_t _bc_idx = 0;
        for (auto& bc : bc_m) {
            IPPL_HALO_SYNC("BConds:bc_pre[" << _bc_idx << "]");
            bc->apply(field);
            IPPL_HALO_SYNC("BConds:bc_post[" << _bc_idx << "]");
            ++_bc_idx;
        }
        Kokkos::fence();
        field.getCommunicator().barrier();
        IPPL_HALO_SYNC("BConds:exit");
    }

    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::assignGhostToPhysical(Field& field) {
        for (auto& bc : bc_m) {
            bc->assignGhostToPhysical(field);
        }
        Kokkos::fence();
        field.getCommunicator().barrier();
    }

    template <typename Field, unsigned Dim>
    bool BConds<Field, Dim>::changesPhysicalCells() const {
        for (const auto& bc : bc_m) {
            if (bc->changesPhysicalCells()) {
                return true;
            }
        }
        return false;
    }
}  // namespace ippl
