// Bare minimum: call accumulateHalo (or fillHalo) in a tight loop.
// On pif-bo: hangs after ~100 calls.
// On master: hangs after ~5000 calls.
// Usage:  mpiexec -n 4 ./TestHaloOnly [pt] [nCalls] [mode]
//   mode = "accum" or "fill" (default accum)

#include "Ippl.h"
#include <iostream>
#include "Utility/Inform.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int Dim = 3;
        using Mesh_t      = ippl::UniformCartesian<double, 3>;
        using Centering_t = Mesh_t::DefaultCentering;
        using FT          = ippl::Field<double, Dim, Mesh_t, Centering_t>;

        int pt          = (argc > 1) ? std::atoi(argv[1]) : 16;
        int nCalls      = (argc > 2) ? std::atoi(argv[2]) : 500;
        std::string mode = (argc > 3) ? argv[3] : "accum";

        Inform info("TestHaloOnly");
        info << "pt=" << pt << " nCalls=" << nCalls << " mode=" << mode << endl;

        ippl::Index I(pt);
        ippl::NDIndex<Dim> owned(I, I, I);
        std::array<bool, Dim> isParallel;
        for (unsigned d = 0; d < Dim; ++d) isParallel[d] = true;
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel, true);
        double dx = 2.0 / double(pt);
        ippl::Vector<double, Dim> hx     = {dx, dx, dx};
        ippl::Vector<double, Dim> origin = -1;
        Mesh_t mesh(owned, hx, origin);
        FT f(mesh, layout);
        f = 0.0;

        int r = ippl::Comm->rank();
        std::fprintf(stderr, "[r%d] start %d calls\n", r, nCalls);
        std::fflush(stderr);

        for (int k = 0; k < nCalls; ++k) {
            if (mode == "fill")  f.fillHalo();
            else                 f.accumulateHalo();
            if (k % 100 == 0) {
                std::fprintf(stderr, "[r%d] %d done\n", r, k);
                std::fflush(stderr);
            }
        }

        std::fprintf(stderr, "[r%d] all %d calls done\n", r, nCalls);
        std::fflush(stderr);
        info << "ALL DONE" << endl;
    }
    ippl::finalize();
    return 0;
}
