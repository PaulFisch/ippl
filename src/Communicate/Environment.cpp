//
// Class Environment
//
#include "Ippl.h"

#include "Environment.h"

namespace ippl {
    namespace mpi {

        Environment::Environment(int& argc, char**& argv, const MPI_Comm& comm)
            : comm_m(comm)
            , threadMultiple_m(false) {
            if (!initialized()) {
                int provided = MPI_THREAD_SINGLE;
                MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
                threadMultiple_m = (provided >= MPI_THREAD_MULTIPLE);
                if (!threadMultiple_m) {
                    std::cerr << "MPI doesn't support MPI_THREAD_MULTIPLE!" << std::endl;
                }
            }
        }

        Environment::~Environment() {
            if (!finalized()) {
                MPI_Finalize();
            }
        }

        bool Environment::initialized() {
            int flag = 0;
            MPI_Initialized(&flag);
            return (flag != 0);
        }

        bool Environment::finalized() {
            int flag = 0;
            MPI_Finalized(&flag);
            return (flag != 0);
        }
    }  // namespace mpi
}  // namespace ippl
