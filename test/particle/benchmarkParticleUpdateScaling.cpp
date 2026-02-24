//
// benchmarkParticleUpdateScaling.cpp
//
// Scaling benchmark for particle update, testing RMA / P2P / Alltoall
// count-exchange modes with warmup + timer reset.
//
// Usage:
//   srun ./benchmarkParticleUpdateScaling Nx Ny Nz nParticles nSteps
//       [--warmup N] [--exchange rma|p2p|alltoall]
//       [--overallocate F] [--info N]
//

#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

constexpr unsigned Dim = 3;

typedef ippl::ParticleSpatialLayout<double, Dim> PLayout_t;
typedef ippl::UniformCartesian<double, Dim> Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;

template <typename T, unsigned D>
using Vector = ippl::Vector<T, D>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim> Vector_t;

// ---------------------------------------------------------------------------
// Particle bunch — identical structure to the existing benchmark
// ---------------------------------------------------------------------------
template <class PLayout>
class BenchParticles : public ippl::ParticleBase<PLayout> {
public:
    std::array<bool, Dim> isParallel_m;

    ParticleAttrib<double> qm;
    typename ippl::ParticleBase<PLayout>::particle_position_type P;
    typename ippl::ParticleBase<PLayout>::particle_position_type E;

    BenchParticles(PLayout& pl, std::array<bool, Dim> isParallel)
        : ippl::ParticleBase<PLayout>(pl)
        , isParallel_m(isParallel) {
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
        this->setParticleBC(ippl::BC::PERIODIC);
    }
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static ippl::CountExchange parseMode(const std::string& s) {
    if (s == "rma")
        return ippl::CountExchange::RMA;
    if (s == "p2p")
        return ippl::CountExchange::P2P_GPU;
    if (s == "alltoall")
        return ippl::CountExchange::Alltoall_GPU;
    throw std::invalid_argument("Unknown exchange mode: " + s);
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);

        if (argc < 6) {
            if (ippl::Comm->rank() == 0)
                std::cerr << "Usage: " << argv[0]
                          << " Nx Ny Nz nParticles nSteps"
                             " [--warmup N] [--exchange rma|p2p|alltoall]"
                             " [--overallocate F] [--info N]\n";
            ippl::finalize();
            return 1;
        }

        ippl::Vector<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};
        const unsigned int totalP = std::atoi(argv[4]);
        const unsigned int nt     = std::atoi(argv[5]);

        // optional flags
        int warmupSteps         = 5;
        std::string exchangeStr = "alltoall";
        int infoEvery           = 0;

        for (int i = 6; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--warmup" && i + 1 < argc)
                warmupSteps = std::stoi(argv[++i]);
            else if (flag == "--exchange" && i + 1 < argc)
                exchangeStr = argv[++i];
            else if (flag == "--info" && i + 1 < argc)
                infoEvery = std::stoi(argv[++i]);
            // --overallocate is accepted by ippl::initialize automatically
        }

        const ippl::CountExchange exchangeMode = parseMode(exchangeStr);

        msg << "benchmarkParticleUpdateScaling" << endl
            << "nt=" << nt << " Np=" << totalP << " grid=" << nr << " exchange=" << exchangeStr
            << " warmup=" << warmupSteps << endl;

        // ---------------------------------------------------------------
        // Domain / mesh / layout
        // ---------------------------------------------------------------
        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; ++i)
            domain[i] = ippl::Index(nr[i]);

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        Vector_t rmin(0.0), rmax(1.0);
        double dx       = rmax[0] / double(nr[0]);
        double dy       = rmax[1] / double(nr[1]);
        double dz       = rmax[2] / double(nr[2]);
        Vector_t hr     = {dx, dy, dz};
        Vector_t origin = {rmin[0], rmin[1], rmin[2]};
        double hr_min   = std::min({dx, dy, dz});
        const double dt = 1.0;

        Mesh_t mesh(domain, hr, origin);
        FieldLayout_t FL(MPI_COMM_WORLD, domain, isParallel);
        PLayout_t PL(FL, mesh, /*fem=*/false, exchangeMode);

        // ---------------------------------------------------------------
        // Create particles
        // ---------------------------------------------------------------
        using bunch_type = BenchParticles<PLayout_t>;
        auto P           = std::make_unique<bunch_type>(PL, isParallel);

        unsigned long int nloc = totalP / ippl::Comm->size();

        static IpplTimings::TimerRef tCreate = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(tCreate);

        P->create(nloc);

        // Initialise positions on host exactly as in the existing benchmark
        std::mt19937_64 eng[Dim];
        for (unsigned i = 0; i < Dim; ++i) {
            eng[i].seed(42 + i * Dim);
            eng[i].discard(nloc * ippl::Comm->rank());
        }
        std::uniform_real_distribution<double> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();

        for (unsigned long int i = 0; i < nloc; ++i)
            for (int d = 0; d < 3; ++d)
                R_host(i)[d] = unif(eng[d]);

        Kokkos::deep_copy(P->R.getView(), R_host);
        P->qm = 1.0 / totalP;
        P->E  = 0.0;

        IpplTimings::stopTimer(tCreate);

        // Initial update to put particles on the right ranks
        static IpplTimings::TimerRef tUpdate = IpplTimings::getTimer("ParticleUpdate");
        IpplTimings::startTimer(tUpdate);
        P->update();
        IpplTimings::stopTimer(tUpdate);

        // ---------------------------------------------------------------
        // Warmup loop  — same body as the timed loop; discarded afterwards
        // ---------------------------------------------------------------
        if (ippl::Comm->rank() == 0)
            std::cout << "Running " << warmupSteps << " warmup step(s)...\n";

        Kokkos::Random_XorShift64_Pool<> pool(static_cast<uint64_t>(42 + 100 * ippl::Comm->rank()));

        for (int w = 0; w < warmupSteps; ++w) {
            auto P_view = P->P.getView();
            Kokkos::parallel_for(
                "warmup_randP", Kokkos::RangePolicy<>(0, (int)P->getLocalNum()),
                KOKKOS_LAMBDA(const int i) {
                    auto gen = pool.get_state();
                    for (int d = 0; d < 3; ++d)
                        P_view(i)[d] = gen.drand() * hr_min;
                    pool.free_state(gen);
                });
            Kokkos::fence();

            P->R = P->R + dt * P->P;
            P->update();
            P->P = P->P + dt * P->qm * P->E;
        }

        // Discard all warmup timing so measurements start clean
        IpplTimings::resetAllTimers();

        if (ippl::Comm->rank() == 0)
            std::cout << "Warmup done. Timers reset. Starting timed run.\n";

        // ---------------------------------------------------------------
        // Timed loop
        // ---------------------------------------------------------------
        static IpplTimings::TimerRef tMain  = IpplTimings::getTimer("mainTimer");
        static IpplTimings::TimerRef tRandP = IpplTimings::getTimer("RandomP");
        static IpplTimings::TimerRef tPos   = IpplTimings::getTimer("positionUpdate");
        static IpplTimings::TimerRef tVel   = IpplTimings::getTimer("velocityUpdate");
        // Re-get tUpdate after reset
        tUpdate = IpplTimings::getTimer("ParticleUpdate");

        IpplTimings::startTimer(tMain);

        for (unsigned int it = 0; it < nt; ++it) {
            // randomise velocities on device
            IpplTimings::startTimer(tRandP);
            {
                auto P_view = P->P.getView();
                Kokkos::parallel_for(
                    "randP", Kokkos::RangePolicy<>(0, (int)P->getLocalNum()),
                    KOKKOS_LAMBDA(const int i) {
                        auto gen = pool.get_state();
                        for (int d = 0; d < 3; ++d)
                            P_view(i)[d] = gen.drand() * hr_min;
                        pool.free_state(gen);
                    });
                Kokkos::fence();
            }
            IpplTimings::stopTimer(tRandP);

            // position update
            IpplTimings::startTimer(tPos);
            P->R = P->R + dt * P->P;
            IpplTimings::stopTimer(tPos);

            // particle redistribution
            IpplTimings::startTimer(tUpdate);
            P->update();
            IpplTimings::stopTimer(tUpdate);

            // velocity update
            IpplTimings::startTimer(tVel);
            P->P = P->P + dt * P->qm * P->E;
            IpplTimings::stopTimer(tVel);

            if (infoEvery > 0 && (it + 1) % infoEvery == 0 && ippl::Comm->rank() == 0) {
                std::cout << "  step " << (it + 1) << "/" << nt
                          << "  local particles: " << P->getLocalNum() << "\n";
            }
        }

        IpplTimings::stopTimer(tMain);

        // ---------------------------------------------------------------
        // Output
        // ---------------------------------------------------------------
        IpplTimings::print();
        IpplTimings::print(std::string("timing_") + exchangeStr + ".dat");
    }
    ippl::finalize();
    return 0;
}