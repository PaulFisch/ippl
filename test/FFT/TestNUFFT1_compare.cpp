#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <iomanip>
#include <iostream>
#include <random>
#include <typeinfo>

#include "Utility/ParameterList.h"

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }

    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};

template <unsigned Dim>
bool isOwnedLocally(const ippl::NDIndex<Dim>& lDom, const ippl::Vector<int, Dim>& globalIdx) {
    for (unsigned d = 0; d < Dim; ++d) {
        if (globalIdx[d] < lDom[d].first() || globalIdx[d] > lDom[d].last()) {
            return false;
        }
    }
    return true;
}

template <unsigned Dim>
ippl::Vector<int, Dim> globalToLocal(const ippl::NDIndex<Dim>& lDom,
                                     const ippl::Vector<int, Dim>& globalIdx, int nghost) {
    ippl::Vector<int, Dim> localIdx;
    for (unsigned d = 0; d < Dim; ++d) {
        localIdx[d] = globalIdx[d] - lDom[d].first() + nghost;
    }
    return localIdx;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        const double pi = std::acos(-1.0);

        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;

        int myRank = ippl::Comm->rank();
        int nRanks = ippl::Comm->size();

        // Small grid for debugging
        ippl::Vector<int, dim> n_modes = {8, 8, 8};

        ippl::Index I(n_modes[0]);
        ippl::Index J(n_modes[1]);
        ippl::Index K(n_modes[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        typedef ippl::Vector<double, 3> Vector_t;
        Vector_t minU = {0, 0, 0};
        Vector_t maxU = {2 * pi, 2 * pi, 2 * pi};

        std::array<double, dim> dx = {
            (maxU[0] - minU[0]) / double(n_modes[0]),
            (maxU[1] - minU[1]) / double(n_modes[1]),
            (maxU[2] - minU[2]) / double(n_modes[2]),
        };

        Vector_t hx     = {dx[0], dx[1], dx[2]};
        Vector_t origin = {minU[0], minU[1], minU[2]};
        ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

        playout_type pl(layout, mesh);

        bunch_type bunch(pl);
        bunch.setParticleBC(ippl::BC::PERIODIC);

        using size_type = ippl::detail::size_type;

        // Use a small number of particles with known positions for debugging
        size_type Np = 8;

        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type
            field_type;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type real_field_type;

        ippl::ParameterList fftParams;

        fftParams.add("tolerance", 1e-4);
#ifdef FINUFFT_USE_CUDA
        fftParams.add("gpu_method", 1);
        fftParams.add("gpu_sort", 0);
        fftParams.add("gpu_kerevalmeth", 1);
#else
        fftParams.add("spread_kerevalmeth", 1);
        fftParams.add("spread_sort", 2);
        fftParams.add("nthreads", 0);
#endif

        fftParams.add("use_finufft", false);
        fftParams.add("use_kokkos_nufft", false);
        fftParams.add("use_upsampled_inputs", false);

        typedef ippl::FFT<ippl::NUFFTransform, real_field_type> FFT_type;

        int type       = 1;
        size_type nloc = Np / nRanks;
        if (myRank < static_cast<int>(Np % nRanks)) {
            nloc++;
        }

        bunch.create(nloc);

        // Set known particle positions and charges
        auto Rview = bunch.R.getView();
        auto Qview = bunch.Q.getView();

        // Put particles at known positions for rank 0 only
        if (myRank == 0) {
            auto Rhost = Kokkos::create_mirror_view(Rview);
            auto Qhost = Kokkos::create_mirror_view(Qview);

            for (size_type i = 0; i < nloc; ++i) {
                // Spread particles across the domain
                double frac = double(i) / double(Np);
                Rhost(i)[0] = frac * 2 * pi;
                Rhost(i)[1] = frac * 2 * pi;
                Rhost(i)[2] = frac * 2 * pi;
                Qhost(i)    = 1.0;  // Unit charge
            }

            Kokkos::deep_copy(Rview, Rhost);
            Kokkos::deep_copy(Qview, Qhost);
        }

        // Create FFT object
        std::unique_ptr<FFT_type> fft = std::make_unique<FFT_type>(layout, nloc, type, fftParams);

        const int nghost = 1;
        field_type field_output(mesh, layout, nghost);

        bunch.update();
        field_output = Kokkos::complex(0.0);

        // Print particle info
        if (myRank == 0) {
            std::cout << "\n=== Particle Data ===" << std::endl;
        }

        {
            auto Rhost = Kokkos::create_mirror_view(Rview);
            auto Qhost = Kokkos::create_mirror_view(Qview);
            Kokkos::deep_copy(Rhost, Rview);
            Kokkos::deep_copy(Qhost, Qview);

            size_type actualNloc = bunch.getLocalNum();

            for (int r = 0; r < nRanks; ++r) {
                if (myRank == r) {
                    std::cout << "Rank " << myRank << " has " << actualNloc << " particles:" << std::endl;
                    for (size_type i = 0; i < actualNloc; ++i) {
                        std::cout << "  p[" << i << "]: R=(" << Rhost(i)[0] << ", "
                                  << Rhost(i)[1] << ", " << Rhost(i)[2] << "), Q=" << Qhost(i) << std::endl;
                    }
                }
                ippl::Comm->barrier();
            }
        }

        fft->transform(bunch.R, bunch.Q, field_output);

        // Get local domain info
        const auto& lDom       = layout.getLocalNDIndex();
        const int nghost_field = field_output.getNghost();

        // Copy field to host
        auto field_host = field_output.getHostMirror();
        Kokkos::deep_copy(field_host, field_output.getView());

        if (myRank == 0) {
            std::cout << "\n=== NUFFT Output vs DFT Reference ===" << std::endl;
            std::cout << "Grid size: " << n_modes[0] << " x " << n_modes[1] << " x " << n_modes[2]
                      << std::endl;
            std::cout << std::setw(20) << "Index (i,j,k)"
                      << std::setw(30) << "NUFFT Result"
                      << std::setw(30) << "DFT Reference"
                      << std::setw(15) << "Match?" << std::endl;
            std::cout << std::string(95, '-') << std::endl;
        }

        // For each mode in the grid, compute DFT reference and compare
        for (int ki = 0; ki < n_modes[0]; ++ki) {
            for (int kj = 0; kj < n_modes[1]; ++kj) {
                for (int kk = 0; kk < n_modes[2]; ++kk) {
                    ippl::Vector<int, 3> globalIdx = {ki, kj, kk};

                    // Get NUFFT result from whoever owns this mode
                    Kokkos::complex<double> nufft_result(0.0, 0.0);
                    bool iOwn = isOwnedLocally<dim>(lDom, globalIdx);

                    if (iOwn) {
                        auto localIdx = globalToLocal<dim>(lDom, globalIdx, nghost_field);
                        nufft_result  = field_host(localIdx[0], localIdx[1], localIdx[2]);
                    }

                    // Reduce to rank 0
                    double send_buf[2] = {nufft_result.real(), nufft_result.imag()};
                    double recv_buf[2] = {0.0, 0.0};
                    MPI_Reduce(send_buf, recv_buf, 2, MPI_DOUBLE, MPI_SUM, 0,
                               ippl::Comm->getCommunicator());
                    Kokkos::complex<double> nufft_global(recv_buf[0], recv_buf[1]);

                    // Compute DFT reference with ACTUAL signed frequencies
                    // In corner-DC format: index < N/2 -> freq = index
                    //                      index >= N/2 -> freq = index - N
                    Kokkos::complex<double> dft_local(0.0, 0.0);

                    size_type actualNloc = bunch.getLocalNum();

                    // Convert indices to actual signed frequencies
                    int freq_i = (ki < n_modes[0]/2) ? ki : ki - n_modes[0];
                    int freq_j = (kj < n_modes[1]/2) ? kj : kj - n_modes[1];
                    int freq_k = (kk < n_modes[2]/2) ? kk : kk - n_modes[2];

                    // Compute on host for simplicity
                    {
                        auto Rhost_view = Kokkos::create_mirror_view(bunch.R.getView());
                        auto Qhost_view = Kokkos::create_mirror_view(bunch.Q.getView());
                        Kokkos::deep_copy(Rhost_view, bunch.R.getView());
                        Kokkos::deep_copy(Qhost_view, bunch.Q.getView());

                        for (size_type p = 0; p < actualNloc; ++p) {
                            // Use actual signed frequencies
                            double arg = freq_i * Rhost_view(p)[0]
                                       + freq_j * Rhost_view(p)[1]
                                       + freq_k * Rhost_view(p)[2];
                            // Type 1 NUFFT: sum of c_j * exp(-i * k * x_j)
                            dft_local += Kokkos::complex<double>(std::cos(arg), -std::sin(arg))
                                        * Qhost_view(p);
                        }
                    }

                    // Reduce DFT
                    double dft_send[2] = {dft_local.real(), dft_local.imag()};
                    double dft_recv[2] = {0.0, 0.0};
                    MPI_Reduce(dft_send, dft_recv, 2, MPI_DOUBLE, MPI_SUM, 0,
                               ippl::Comm->getCommunicator());
                    Kokkos::complex<double> dft_global(dft_recv[0], dft_recv[1]);

                    // Print comparison on rank 0
                    if (myRank == 0) {
                        double err = Kokkos::abs(nufft_global - dft_global);
                        bool match = err < 0.1;  // Loose tolerance for visual inspection

                        std::cout << std::setw(6) << "(" << ki << "," << kj << "," << kk << ")"
                                  << " freq(" << freq_i << "," << freq_j << "," << freq_k << ")"
                                  << std::setw(15) << std::setprecision(4) << nufft_global.real()
                                  << " + " << std::setw(10) << nufft_global.imag() << "i"
                                  << std::setw(15) << dft_global.real() << " + " << std::setw(10)
                                  << dft_global.imag() << "i" << std::setw(10)
                                  << (match ? "OK" : "MISMATCH") << " (err=" << err << ")"
                                  << std::endl;
                    }
                }
            }
        }

        // Also try alternative DFT sign conventions
        if (myRank == 0) {
            std::cout << "\n=== Testing Sign Conventions for mode (1,0,0) ===" << std::endl;
        }

        {
            int ki = 1, kj = 0, kk = 0;
            // In corner-DC: index 1 = freq 1 (since 1 < N/2)
            int freq_i = 1, freq_j = 0, freq_k = 0;

            ippl::Vector<int, 3> globalIdx = {ki, kj, kk};

            Kokkos::complex<double> nufft_result(0.0, 0.0);
            bool iOwn = isOwnedLocally<dim>(lDom, globalIdx);
            if (iOwn) {
                auto localIdx = globalToLocal<dim>(lDom, globalIdx, nghost_field);
                nufft_result  = field_host(localIdx[0], localIdx[1], localIdx[2]);
            }
            double send_buf[2] = {nufft_result.real(), nufft_result.imag()};
            double recv_buf[2] = {0.0, 0.0};
            MPI_Reduce(send_buf, recv_buf, 2, MPI_DOUBLE, MPI_SUM, 0,
                       ippl::Comm->getCommunicator());
            Kokkos::complex<double> nufft_global(recv_buf[0], recv_buf[1]);

            if (myRank == 0) {
                std::cout << "NUFFT result at index (1,0,0) [freq (1,0,0)]: " << nufft_global << std::endl;

                // Compute with different sign conventions
                auto Rhost_view = Kokkos::create_mirror_view(bunch.R.getView());
                auto Qhost_view = Kokkos::create_mirror_view(bunch.Q.getView());
                Kokkos::deep_copy(Rhost_view, bunch.R.getView());
                Kokkos::deep_copy(Qhost_view, bunch.Q.getView());

                size_type actualNloc = bunch.getLocalNum();

                // Convention 1: exp(-i k x)
                Kokkos::complex<double> dft1(0.0, 0.0);
                // Convention 2: exp(+i k x)
                Kokkos::complex<double> dft2(0.0, 0.0);

                for (size_type p = 0; p < actualNloc; ++p) {
                    double arg = freq_i * Rhost_view(p)[0] + freq_j * Rhost_view(p)[1] + freq_k * Rhost_view(p)[2];

                    dft1 += Kokkos::complex<double>(std::cos(arg), -std::sin(arg)) * Qhost_view(p);
                    dft2 += Kokkos::complex<double>(std::cos(arg), +std::sin(arg)) * Qhost_view(p);
                }

                std::cout << "DFT exp(-ikx):        " << dft1 << std::endl;
                std::cout << "DFT exp(+ikx):        " << dft2 << std::endl;

                // Check which matches
                std::cout << "\nErrors:" << std::endl;
                std::cout << "  |NUFFT - exp(-ikx)|:       " << Kokkos::abs(nufft_global - dft1) << std::endl;
                std::cout << "  |NUFFT - exp(+ikx)|:       " << Kokkos::abs(nufft_global - dft2) << std::endl;
            }
        }

        ippl::Comm->barrier();
    }
    ippl::finalize();
    return 0;
}