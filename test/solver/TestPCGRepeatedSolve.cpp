// TestPCGRepeatedSolve
//
// Minimal standalone reproducer for the HaloCells deadlock observed in
// LandauDamping + PCG/newton on the pif-bo branch.
//
// What it does: build a 3D Field with periodic BCs, run PoissonCG with the
// newton preconditioner in a loop. Before each solve, "scatter-like" the
// rhs with `rhs = 1.0; rhs.accumulateHalo()`. No particles, no scatter
// kernel, no load balancing.
//
// Empirical bisect (all on this binary):
//   accumulateHalo(rhs) + PCG/newton level >= 3   -> HANGS
//   accumulateHalo(rhs) + PCG/newton level 1 or 2 -> ok
//   accumulateHalo(rhs) + plain CG (no precond)   -> ok
//   accumulateHalo(rhs) + PCG/jacobi              -> ok
//   accumulateHalo(rhs) + PCG/chebyshev (deg 31)  -> HANGS
//   fillHalo(rhs) instead of accumulateHalo + PCG/newton/4 -> ok
//   no halo on rhs + PCG/newton/4 in a loop       -> ok
//
// Hang signature (matches what the user saw in nsys):
//   one rank stuck in MPI_Recv inside HaloCells::exchangeBoundaries,
//   one rank in MPI_Waitall in the same fillHalo,
//   other ranks already past that fillHalo and waiting in MPI_Barrier
//   from BConds::apply -> Communicator::barrier.
//
// Usage:
//   mpiexec -n 8 ./TestPCGRepeatedSolve <pt> <nSolves> [preconditioner] [level]
//   e.g.   mpiexec -n 8 ./TestPCGRepeatedSolve 16 3 newton 4
//
// COMPILE_ONLY in test/solver/CMakeLists.txt -> ctest does not run it.

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <iostream>
#include <string>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/PoissonCG.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, 3>;
        using Centering_t          = Mesh_t::DefaultCentering;
        using field_type           = ippl::Field<double, dim, Mesh_t, Centering_t>;

        Inform info("TestPCGRepeatedSolve");

        int pt                          = (argc > 1) ? std::atoi(argv[1]) : 16;
        int nSolves                     = (argc > 2) ? std::atoi(argv[2]) : 3;
        std::string preconditioner_type = (argc > 3) ? argv[3] : "newton";
        int newton_level                = (argc > 4) ? std::atoi(argv[4]) : 4;

        info << "ranks=" << ippl::Comm->size() << " pt=" << pt << " nSolves=" << nSolves
             << " precond=" << preconditioner_type << " level=" << newton_level << endl;

        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        std::array<bool, dim> isParallel;
        for (unsigned int d = 0; d < dim; ++d) isParallel[d] = true;

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel, /*isAllPeriodic=*/true);

        double dx                        = 2.0 / double(pt);
        ippl::Vector<double, dim> hx     = {dx, dx, dx};
        ippl::Vector<double, dim> origin = -1;
        Mesh_t mesh(owned, hx, origin);

        field_type rhs(mesh, layout);
        field_type lhs(mesh, layout);

        typedef ippl::BConds<field_type, dim> bc_type;
        bc_type bcField;
        for (unsigned int i = 0; i < 2 * dim; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
        }
        lhs.setFieldBC(bcField);

        ippl::PoissonCG<field_type> lapsolver;

        ippl::ParameterList params;
        params.add("max_iterations", 500);
        params.add("tolerance", 1e-4);
        params.add("solver", "preconditioned");
        params.add("output_type", ippl::PoissonCG<field_type>::SOL);
        params.add("preconditioner_type", preconditioner_type);
        params.add("gauss_seidel_inner_iterations", 5);
        params.add("gauss_seidel_outer_iterations", 1);
        params.add("newton_level", newton_level);
        params.add("chebyshev_degree", 31);
        params.add("richardson_iterations", 1);
        params.add("communication", 1);
        params.add("ssor_omega", 1.57079632679);
        lapsolver.mergeParameters(params);

        lapsolver.setRhs(rhs);
        lapsolver.setLhs(lhs);

        // stderr-direct progress prints so the hang is visible per-rank
        // without depending on Inform's host-only buffered output.
        int me   = ippl::Comm->rank();
        auto tick = [&](const char* what, int it) {
            std::fprintf(stderr, "[rank %d iter %d] %s\n", me, it, what);
            std::fflush(stderr);
        };

        // newton-like recursion that RETURNS Field by value (matching the
        // real recursive_preconditioner signature exactly).
        std::function<field_type(field_type&, int)> rec = [&](field_type& u, int lvl) {
            field_type res(mesh, layout);
            if (lvl == 0) {
                res = u;
                return res;
            }
            field_type PAPr(mesh, layout);
            field_type Pr(mesh, layout);
            Pr   = rec(u, lvl - 1);
            PAPr = -laplace(Pr);
            PAPr = rec(PAPr, lvl - 1);
            res  = Pr - PAPr;
            return res;
        };

        field_type persistent_r(mesh, layout);
        field_type persistent_d(mesh, layout);
        field_type persistent_q(mesh, layout);
        field_type persistent_s(mesh, layout);

        for (int it = 0; it < nSolves; ++it) {
            tick("rhs.accumulateHalo()", it);
            rhs = 1.0;
            rhs.accumulateHalo();

            tick("PoissonCG solve (the hanging path)", it);
            lhs = 0;
            lapsolver.solve();
            tick("inner done", it);
        }

        info << "all " << nSolves << " solves completed without hang" << endl;
    }
    ippl::finalize();
    return 0;
}
