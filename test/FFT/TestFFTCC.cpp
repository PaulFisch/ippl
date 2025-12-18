#include "Ippl.h"

#include <array>
#include <iostream>
#include <random>
#include <typeinfo>

#include "Utility/ParameterList.h"

// int main(int argc, char* argv[]) {
//     ippl::initialize(argc, argv);
//     {
//         constexpr unsigned int dim = 3;
//         using Mesh_t               = ippl::UniformCartesian<double, dim>;
//         using Centering_t          = Mesh_t::DefaultCentering;
//
//         std::array<int, dim> pt = {32, 32, 32};
//         ippl::Index I(pt[0]);
//         ippl::Index J(pt[1]);
//         ippl::Index K(pt[2]);
//         ippl::NDIndex<dim> owned(I, J, K);
//
//         std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
//         isParallel.fill(true);
//
//         ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);
//
//         std::array<double, dim> dx = {
//             1.0 / double(pt[0]),
//             1.0 / double(pt[1]),
//             1.0 / double(pt[2]),
//         };
//         ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
//         ippl::Vector<double, 3> origin = {0, 0, 0};
//         Mesh_t mesh(owned, hx, origin);
//
//         typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;
//
//         field_type field(mesh, layout);
//
//         ippl::ParameterList fftParams;
//
//         fftParams.add("use_heffte_defaults", true);
//
//         typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;
//
//         std::unique_ptr<FFT_type> fft;
//
//         fft = std::make_unique<FFT_type>(layout, fftParams);
//
//         typename field_type::view_type& view       = field.getView();
//         typename field_type::HostMirror field_host = field.getHostMirror();
//
//         const int nghost = field.getNghost();
//         std::mt19937_64 engReal(42 + ippl::Comm->rank());
//         std::uniform_real_distribution<double> unifReal(0, 1);
//
//         std::mt19937_64 engImag(43 + ippl::Comm->rank());
//         std::uniform_real_distribution<double> unifImag(0, 1);
//
//         for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
//             for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
//                 for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
//                     field_host(i, j, k).real() = unifReal(engReal);  // 1.0;
//                     field_host(i, j, k).imag() = unifImag(engImag);  // 1.0;
//                 }
//             }
//         }
//
//         Kokkos::deep_copy(field.getView(), field_host);
//
//         // Forward transform
//         fft->transform(ippl::FORWARD, field);
//         // Reverse transform
//         fft->transform(ippl::BACKWARD, field);
//
//         auto field_result =
//             Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());
//
//         Kokkos::complex<double> max_error_local(0.0, 0.0);
//         for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
//             for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
//                 for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
//                     Kokkos::complex<double> error(
//                         std::fabs(field_host(i, j, k).real() - field_result(i, j, k).real()),
//                         std::fabs(field_host(i, j, k).imag() - field_result(i, j, k).imag()));
//
//                     if (error.real() > max_error_local.real()) {
//                         max_error_local.real() = error.real();
//                     }
//
//                     if (error.imag() > max_error_local.imag()) {
//                         max_error_local.imag() = error.imag();
//                     }
//                     // std::cout << "Error: " << std::setprecision(16) << error << std::endl;
//                 }
//             }
//         }
//
//         Kokkos::complex<double> max_error(0.0, 0.0);
//         // MPI_Reduce(&max_error_local, &max_error, 1, MPI_C_DOUBLE_COMPLEX, MPI_MAX, 0,
//         //            ippl::Comm->getCommunicator());
//
//         // if (ippl::Comm->rank() == 0) {
//             std::cout << "Rank:" << ippl::Comm->rank() << "Max. error " << std::setprecision(16)
//                       << max_error_local << std::endl;
//         // }
//     }
//     ippl::finalize();
//
//     return 0;
// }
//
// //
#include "Ippl.h"

#include <array>
#include <iostream>
#include <random>
#include <typeinfo>

#include "Utility/ParameterList.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        std::array<int, dim> pt = {32, 32, 32};
        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::cout << "=== CC FFT TEST DEBUG ===" << std::endl;
        std::cout << "Grid size: " << pt[0] << " x " << pt[1] << " x " << pt[2] << std::endl;

        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        std::cout << "Layout domain: ";
        for (unsigned d = 0; d < dim; ++d) {
            std::cout << layout.getDomain()[d].length() << " ";
        }
        std::cout << std::endl;

        std::cout << "Local domain: ";
        const auto& lDom = layout.getLocalNDIndex();
        for (unsigned d = 0; d < dim; ++d) {
            std::cout << "[" << lDom[d].first() << "," << lDom[d].last() << "] ";
        }
        std::cout << std::endl;

        std::array<double, dim> dx = {
            1.0 / double(pt[0]),
            1.0 / double(pt[1]),
            1.0 / double(pt[2]),
        };
        ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
        ippl::Vector<double, 3> origin = {0, 0, 0};
        Mesh_t mesh(owned, hx, origin);

        std::cout << "Mesh spacing: " << hx[0] << " x " << hx[1] << " x " << hx[2] << std::endl;

        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;

        field_type field(mesh, layout);

        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", true);

        typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;

        std::cout << "Creating FFT object..." << std::endl;
        std::unique_ptr<FFT_type> fft;
        fft = std::make_unique<FFT_type>(layout, fftParams);
        std::cout << "FFT object created." << std::endl;

        typename field_type::view_type& view       = field.getView();
        typename field_type::HostMirror field_host = field.getHostMirror();

        const int nghost = field.getNghost();
        std::cout << "Field nghost: " << nghost << std::endl;
        std::cout << "Field view extents: " << view.extent(0) << " x " << view.extent(1) << " x " << view.extent(2) << std::endl;

        std::mt19937_64 engReal(42 + ippl::Comm->rank());
        std::uniform_real_distribution<double> unifReal(0, 1);

        std::mt19937_64 engImag(43 + ippl::Comm->rank());
        std::uniform_real_distribution<double> unifImag(0, 1);

        for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
            for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
                for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
                    field_host(i, j, k).real() = unifReal(engReal);
                    field_host(i, j, k).imag() = unifImag(engImag);
                }
            }
        }

        Kokkos::deep_copy(field.getView(), field_host);

        // Print initial field values (first few)
        std::cout << "=== INITIAL FIELD VALUES ===" << std::endl;
        for (int i = nghost; i < std::min(nghost + 3, int(view.extent(0)) - nghost); ++i) {
            for (int j = nghost; j < std::min(nghost + 3, int(view.extent(1)) - nghost); ++j) {
                for (int k = nghost; k < std::min(nghost + 3, int(view.extent(2)) - nghost); ++k) {
                    std::cout << "  field_host(" << i << "," << j << "," << k << ") = "
                              << field_host(i, j, k).real() << " + " << field_host(i, j, k).imag() << "i" << std::endl;
                }
            }
        }

        // Forward transform
        std::cout << "=== FORWARD TRANSFORM ===" << std::endl;
        fft->transform(ippl::FORWARD, field);
        Kokkos::fence();

        // Print field values after forward transform
        auto field_after_forward = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());
        std::cout << "=== FIELD AFTER FORWARD ===" << std::endl;
        for (int i = nghost; i < std::min(nghost + 3, int(view.extent(0)) - nghost); ++i) {
            for (int j = nghost; j < std::min(nghost + 3, int(view.extent(1)) - nghost); ++j) {
                for (int k = nghost; k < std::min(nghost + 3, int(view.extent(2)) - nghost); ++k) {
                    std::cout << "  field(" << i << "," << j << "," << k << ") = "
                              << field_after_forward(i, j, k).real() << " + " << field_after_forward(i, j, k).imag() << "i" << std::endl;
                }
            }
        }

        // Compute sum/norm of forward result for sanity check
        double sum_real = 0.0, sum_imag = 0.0;
        for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
            for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
                for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
                    sum_real += std::abs(field_after_forward(i, j, k).real());
                    sum_imag += std::abs(field_after_forward(i, j, k).imag());
                }
            }
        }
        std::cout << "Sum of |real| after forward: " << sum_real << std::endl;
        std::cout << "Sum of |imag| after forward: " << sum_imag << std::endl;

        // Reverse transform
        std::cout << "=== BACKWARD TRANSFORM ===" << std::endl;
        fft->transform(ippl::BACKWARD, field);
        Kokkos::fence();

        auto field_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());

        // Print field values after backward transform
        std::cout << "=== FIELD AFTER BACKWARD ===" << std::endl;
        for (int i = nghost; i < std::min(nghost + 3, int(view.extent(0)) - nghost); ++i) {
            for (int j = nghost; j < std::min(nghost + 3, int(view.extent(1)) - nghost); ++j) {
                for (int k = nghost; k < std::min(nghost + 3, int(view.extent(2)) - nghost); ++k) {
                    std::cout << "  field(" << i << "," << j << "," << k << ") = "
                              << field_result(i, j, k).real() << " + " << field_result(i, j, k).imag() << "i" << std::endl;
                }
            }
        }

        // Compare with original
        std::cout << "=== COMPARISON (original vs result) ===" << std::endl;
        for (int i = nghost; i < std::min(nghost + 3, int(view.extent(0)) - nghost); ++i) {
            for (int j = nghost; j < std::min(nghost + 3, int(view.extent(1)) - nghost); ++j) {
                for (int k = nghost; k < std::min(nghost + 3, int(view.extent(2)) - nghost); ++k) {
                    std::cout << "  (" << i << "," << j << "," << k << "): original=("
                              << field_host(i, j, k).real() << "," << field_host(i, j, k).imag()
                              << ") result=("
                              << field_result(i, j, k).real() << "," << field_result(i, j, k).imag()
                              << ") diff=("
                              << std::abs(field_host(i, j, k).real() - field_result(i, j, k).real()) << ","
                              << std::abs(field_host(i, j, k).imag() - field_result(i, j, k).imag()) << ")"
                              << std::endl;
                }
            }
        }

        Kokkos::complex<double> max_error_local(0.0, 0.0);
        Kokkos::complex<double> max_error_location_real(-1, -1);
        Kokkos::complex<double> max_error_location_imag(-1, -1);

        for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
            for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
                for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
                    Kokkos::complex<double> error(
                        std::fabs(field_host(i, j, k).real() - field_result(i, j, k).real()),
                        std::fabs(field_host(i, j, k).imag() - field_result(i, j, k).imag()));

                    if (error.real() > max_error_local.real()) {
                        max_error_local.real() = error.real();
                        max_error_location_real = Kokkos::complex<double>(i, j);
                    }

                    if (error.imag() > max_error_local.imag()) {
                        max_error_local.imag() = error.imag();
                        max_error_location_imag = Kokkos::complex<double>(i, j);
                    }
                }
            }
        }

        std::cout << "=== FINAL ERROR SUMMARY ===" << std::endl;
        std::cout << "Rank:" << ippl::Comm->rank() << " Max. error " << std::setprecision(16)
                  << max_error_local << std::endl;
        std::cout << "Max real error at approx (" << max_error_location_real.real() << "," << max_error_location_real.imag() << ")" << std::endl;
        std::cout << "Max imag error at approx (" << max_error_location_imag.real() << "," << max_error_location_imag.imag() << ")" << std::endl;

        if (max_error_local.real() < 1e-10 && max_error_local.imag() < 1e-10) {
            std::cout << "CC FFT Test PASSED" << std::endl;
        } else {
            std::cout << "CC FFT Test FAILED" << std::endl;
        }
    }
    ippl::finalize();

    return 0;
}