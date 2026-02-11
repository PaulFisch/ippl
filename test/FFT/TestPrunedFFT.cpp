// #include "Ippl.h"
//
// #include <array>
// #include <cmath>
// #include <iomanip>
// #include <iostream>
// #include <random>
//
// #include "Utility/ParameterList.h"
//
// // Test pruned C2C FFT by comparing with full FFT
// bool testPrunedCC() {
//     constexpr unsigned int dim = 3;
//     using Mesh_t               = ippl::UniformCartesian<double, dim>;
//     using Centering_t          = Mesh_t::DefaultCentering;
//
//     std::array<int, dim> pt_full   = {64, 128, 32};
//     std::array<int, dim> pt_pruned = {32, 64, 16};
//
//     // Create layouts
//     ippl::Index I_full(pt_full[0]);
//     ippl::Index J_full(pt_full[1]);
//     ippl::Index K_full(pt_full[2]);
//     ippl::NDIndex<dim> owned_full(I_full, J_full, K_full);
//
//     ippl::Index I_pruned(pt_pruned[0]);
//     ippl::Index J_pruned(pt_pruned[1]);
//     ippl::Index K_pruned(pt_pruned[2]);
//     ippl::NDIndex<dim> owned_pruned(I_pruned, J_pruned, K_pruned);
//
//     std::array<bool, dim> isParallel;
//     isParallel.fill(true);
//
//     ippl::FieldLayout<dim> layout_full(MPI_COMM_WORLD, owned_full, isParallel);
//     ippl::FieldLayout<dim> layout_pruned(MPI_COMM_WORLD, owned_pruned, isParallel);
//
//     std::array<double, dim> dx = {
//         1.0 / double(pt_full[0]),
//         1.0 / double(pt_full[1]),
//         1.0 / double(pt_full[2]),
//     };
//     ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
//     ippl::Vector<double, 3> origin = {0, 0, 0};
//     Mesh_t mesh_full(owned_full, hx, origin);
//     Mesh_t mesh_pruned(owned_pruned, hx, origin);
//
//     typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;
//
//     field_type field_input(mesh_full, layout_full);
//     field_type field_full_result(mesh_full, layout_full);
//     field_type field_pruned_result(mesh_pruned, layout_pruned);
//
//     // Setup pruning parameters
//     ippl::PruningParams<dim> pruning;
//     pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_pruned[0]),
//                                                 static_cast<size_t>(pt_pruned[1]),
//                                                 static_cast<size_t>(pt_pruned[2])};
//
//     ippl::ParameterList fftParams;
//     fftParams.add("use_heffte_defaults", true);
//
//     // Create FFTs
//     typedef ippl::FFT<ippl::PrunedCCTransform, field_type> PrunedFFT_type;
//     typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;
//
//     auto pruned_fft =
//         std::make_unique<PrunedFFT_type>(layout_full, layout_pruned, pruning, fftParams);
//     auto regular_fft = std::make_unique<FFT_type>(layout_full, fftParams);
//
//     if (ippl::Comm->rank() == 0) {
//         std::cout << "\n=== Testing Pruned C2C FFT ===" << std::endl;
//         std::cout << "Full grid: " << pt_full[0] << "x" << pt_full[1] << "x" << pt_full[2]
//                   << std::endl;
//         std::cout << "Pruned to: " << pt_pruned[0] << "x" << pt_pruned[1] << "x" << pt_pruned[2]
//                   << std::endl;
//     }
//
//     // Initialize with random data
//     const int nghost                           = field_input.getNghost();
//     auto& view_full                            = field_input.getView();
//     typename field_type::HostMirror field_host = field_input.getHostMirror();
//
//     std::mt19937_64 eng(42 + ippl::Comm->rank());
//     std::uniform_real_distribution<double> unif(-1.0, 1.0);
//
//     for (size_t i = nghost; i < view_full.extent(0) - nghost; ++i) {
//         for (size_t j = nghost; j < view_full.extent(1) - nghost; ++j) {
//             for (size_t k = nghost; k < view_full.extent(2) - nghost; ++k) {
//                 field_host(i, j, k) = Kokkos::complex<double>(unif(eng), unif(eng));
//             }
//         }
//     }
//     Kokkos::deep_copy(field_input.getView(), field_host);
//
//     // Save input and compute both FFTs
//     field_full_result = field_input;
//
//     pruned_fft->transform(ippl::FORWARD, field_input, field_pruned_result);
//     regular_fft->transform(ippl::FORWARD, field_full_result);
//
//     // Compare pruned result with corresponding modes from full result
//     auto view_pruned      = field_pruned_result.getView();
//     auto view_full_result = field_full_result.getView();
//
//     const auto& lDom_pruned = layout_pruned.getLocalNDIndex();
//     const auto& lDom_full   = layout_full.getLocalNDIndex();
//     const int nghost_pruned = field_pruned_result.getNghost();
//
//     const int N0 = pt_full[0], K0 = pt_pruned[0];
//     const int N1 = pt_full[1], K1 = pt_pruned[1];
//     const int N2 = pt_full[2], K2 = pt_pruned[2];
//
//     const int p0_first = lDom_pruned[0].first();
//     const int p1_first = lDom_pruned[1].first();
//     const int p2_first = lDom_pruned[2].first();
//
//     const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
//     const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
//     const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();
//
//
//     const int ng   = nghost;
//     const int ng_p = nghost_pruned;
//
//     double max_error = 0.0;
//     size_t count     = 0;
//
//     using exec_space = typename field_type::execution_space;
//     using mdrange_t  = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;
//
//     Kokkos::parallel_reduce(
//         "ComparePrunedWithFull",
//         mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
//                                        view_pruned.extent(2) - ng_p}),
//         KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, double& local_max) {
//             int gi_p = li_p - ng_p + p0_first;
//             int gj_p = lj_p - ng_p + p1_first;
//             int gk_p = lk_p - ng_p + p2_first;
//
//             int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
//             int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
//             int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);
//
//             if (gi_f < f0_first || gi_f > f0_last || gj_f < f1_first || gj_f > f1_last
//                 || gk_f < f2_first || gk_f > f2_last) {
//                 return;
//             }
//
//             int li_f = gi_f - f0_first + ng;
//             int lj_f = gj_f - f1_first + ng;
//             int lk_f = gk_f - f2_first + ng;
//
//             auto val_pruned = view_pruned(li_p, lj_p, lk_p);
//             auto val_full   = view_full_result(li_f, lj_f, lk_f);
//
//             double error = Kokkos::abs(val_pruned - val_full);
//             if (error > local_max) {
//                 local_max = error;
//             }
//         },
//         Kokkos::Max<double>(max_error));
//
//     Kokkos::fence();
//
//     Kokkos::parallel_reduce(
//         "CountComparisons",
//         mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
//                                        view_pruned.extent(2) - ng_p}),
//         KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, size_t& local_count) {
//             int gi_p = li_p - ng_p + p0_first;
//             int gj_p = lj_p - ng_p + p1_first;
//             int gk_p = lk_p - ng_p + p2_first;
//
//             int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
//             int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
//             int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);
//
//             if (gi_f >= f0_first && gi_f <= f0_last && gj_f >= f1_first && gj_f <= f1_last
//                 && gk_f >= f2_first && gk_f <= f2_last) {
//                 ++local_count;
//             }
//         },
//         count);
//
//     Kokkos::fence();
//
//     double global_max_error;
//     size_t global_count;
//     MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX,
//                   ippl::Comm->getCommunicator());
//     MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM,
//                   ippl::Comm->getCommunicator());
//
//     bool passed = (global_max_error < 1e-10);
//
//     if (ippl::Comm->rank() == 0) {
//         std::cout << "Compared " << global_count << " modes" << std::endl;
//         std::cout << "Max error vs full FFT: " << std::scientific << std::setprecision(6)
//                   << global_max_error << std::endl;
//         std::cout << "Pruned C2C FFT test: " << (passed ? "PASSED" : "FAILED") << std::endl;
//     }
//
//     return passed;
// }
//
// bool testPrunedCCBackward() {
//     constexpr unsigned int dim = 3;
//     using Mesh_t               = ippl::UniformCartesian<double, dim>;
//     using Centering_t          = Mesh_t::DefaultCentering;
//
//     std::array<int, dim> pt_full   = {16, 64, 32};
//     std::array<int, dim> pt_pruned = {8, 32, 16};
//
//     // Create layouts
//     ippl::Index I_full(pt_full[0]);
//     ippl::Index J_full(pt_full[1]);
//     ippl::Index K_full(pt_full[2]);
//     ippl::NDIndex<dim> owned_full(I_full, J_full, K_full);
//
//     ippl::Index I_pruned(pt_pruned[0]);
//     ippl::Index J_pruned(pt_pruned[1]);
//     ippl::Index K_pruned(pt_pruned[2]);
//     ippl::NDIndex<dim> owned_pruned(I_pruned, J_pruned, K_pruned);
//
//     std::array<bool, dim> isParallel;
//     isParallel.fill(true);
//
//     ippl::FieldLayout<dim> layout_full(MPI_COMM_WORLD, owned_full, isParallel);
//     ippl::FieldLayout<dim> layout_pruned(MPI_COMM_WORLD, owned_pruned, isParallel);
//
//     std::array<double, dim> dx = {
//         1.0 / double(pt_full[0]),
//         1.0 / double(pt_full[1]),
//         1.0 / double(pt_full[2]),
//     };
//     ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
//     ippl::Vector<double, 3> origin = {0, 0, 0};
//     Mesh_t mesh_full(owned_full, hx, origin);
//     Mesh_t mesh_pruned(owned_pruned, hx, origin);
//
//     typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;
//
//     // For backward: input is pruned (frequency), output is full (spatial)
//     field_type field_pruned_input(mesh_pruned, layout_pruned);
//     field_type field_full_output(mesh_full, layout_full);
//     field_type field_full_freq(mesh_full, layout_full);       // Full frequency for reference
//     field_type field_full_reference(mesh_full, layout_full);  // Reference spatial output
//
//     // Setup pruning parameters
//     ippl::PruningParams<dim> pruning;
//     pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_pruned[0]),
//                                                 static_cast<size_t>(pt_pruned[1]),
//                                                 static_cast<size_t>(pt_pruned[2])};
//
//     ippl::ParameterList fftParams;
//     fftParams.add("use_heffte_defaults", true);
//
//     // Create FFTs
//     typedef ippl::FFT<ippl::PrunedCCTransform, field_type> PrunedFFT_type;
//     typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;
//
//     auto pruned_fft =
//         std::make_unique<PrunedFFT_type>(layout_full, layout_pruned, pruning, fftParams);
//     auto regular_fft = std::make_unique<FFT_type>(layout_full, fftParams);
//
//     if (ippl::Comm->rank() == 0) {
//         std::cout << "\n=== Testing Pruned C2C Backward FFT ===" << std::endl;
//         std::cout << "Pruned frequency: " << pt_pruned[0] << "x" << pt_pruned[1] << "x"
//                   << pt_pruned[2] << std::endl;
//         std::cout << "Full spatial: " << pt_full[0] << "x" << pt_full[1] << "x" << pt_full[2]
//                   << std::endl;
//     }
//
//     const int nghost        = field_full_output.getNghost();
//     const int nghost_pruned = field_pruned_input.getNghost();
//
//     const int N0 = pt_full[0], K0 = pt_pruned[0];
//     const int N1 = pt_full[1], K1 = pt_pruned[1];
//     const int N2 = pt_full[2], K2 = pt_pruned[2];
//
//     // Initialize pruned field with random frequency data
//     typename field_type::HostMirror pruned_host = field_pruned_input.getHostMirror();
//
//     std::mt19937_64 eng(42 + ippl::Comm->rank());
//     std::uniform_real_distribution<double> unif(-1.0, 1.0);
//
//     auto& view_pruned = field_pruned_input.getView();
//     for (size_t i = nghost_pruned; i < view_pruned.extent(0) - nghost_pruned; ++i) {
//         for (size_t j = nghost_pruned; j < view_pruned.extent(1) - nghost_pruned; ++j) {
//             for (size_t k = nghost_pruned; k < view_pruned.extent(2) - nghost_pruned; ++k) {
//                 // pruned_host(i, j, k) = Kokkos::complex<double>(unif(eng), unif(eng));
//                 pruned_host(i, j, k) = 1.0;
//             }
//         }
//     }
//     Kokkos::deep_copy(field_pruned_input.getView(), pruned_host);
//
//     // Initialize full frequency field: zero everywhere, then copy pruned modes
//     // to correct positions (low freq at start, high freq at end)
//     Kokkos::deep_copy(field_full_freq.getView(), Kokkos::complex<double>(0.0, 0.0));
//
//     const auto& lDom_pruned = layout_pruned.getLocalNDIndex();
//     const auto& lDom_full   = layout_full.getLocalNDIndex();
//
//     const int p0_first = lDom_pruned[0].first();
//     const int p1_first = lDom_pruned[1].first();
//     const int p2_first = lDom_pruned[2].first();
//
//     const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
//     const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
//     const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();
//
//     // Copy pruned modes to full frequency field at correct positions
//     auto view_full_freq = field_full_freq.getView();
//
//     using exec_space = typename field_type::execution_space;
//     using mdrange_t  = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;
//
//     Kokkos::parallel_for(
//         "CopyPrunedToFull",
//         mdrange_t({nghost_pruned, nghost_pruned, nghost_pruned},
//                   {view_pruned.extent(0) - nghost_pruned, view_pruned.extent(1) - nghost_pruned,
//                    view_pruned.extent(2) - nghost_pruned}),
//         KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p) {
//             // Global pruned indices
//             int gi_p = li_p - nghost_pruned + p0_first;
//             int gj_p = lj_p - nghost_pruned + p1_first;
//             int gk_p = lk_p - nghost_pruned + p2_first;
//
//             // Map to global full indices
//             int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
//             int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
//             int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);
//
//             // Check if this full index is owned by this rank
//             if (gi_f >= f0_first && gi_f <= f0_last && gj_f >= f1_first && gj_f <= f1_last
//                 && gk_f >= f2_first && gk_f <= f2_last) {
//                 // Local full indices
//                 int li_f = gi_f - f0_first + nghost;
//                 int lj_f = gj_f - f1_first + nghost;
//                 int lk_f = gk_f - f2_first + nghost;
//
//                 view_full_freq(li_f, lj_f, lk_f) = view_pruned(li_p, lj_p, lk_p);
//             }
//         });
//
//     Kokkos::fence();
//
//     eng.seed(42 + ippl::Comm->rank());
//
//     typename field_type::HostMirror full_freq_host = field_full_freq.getHostMirror();
//     Kokkos::deep_copy(full_freq_host, Kokkos::complex<double>(0.0, 0.0));
//
//     // We need to ensure consistency: generate the same random values and place them
//     // For a proper test, we reinitialize both fields with matching data
//
//     // Reinitialize pruned field
//     eng.seed(123);  // Fixed seed for reproducibility across ranks
//     for (size_t i = nghost_pruned; i < view_pruned.extent(0) - nghost_pruned; ++i) {
//         for (size_t j = nghost_pruned; j < view_pruned.extent(1) - nghost_pruned; ++j) {
//             for (size_t k = nghost_pruned; k < view_pruned.extent(2) - nghost_pruned; ++k) {
//                 int gi_p = i - nghost_pruned + p0_first;
//                 int gj_p = j - nghost_pruned + p1_first;
//                 int gk_p = k - nghost_pruned + p2_first;
//
//                 // Use global index as seed for deterministic values
//                 std::mt19937_64 local_eng(gi_p * 10000 + gj_p * 100 + gk_p);
//                 std::uniform_real_distribution<double> local_unif(-1.0, 1.0);
//
//                 pruned_host(i, j, k) =
//                     Kokkos::complex<double>(local_unif(local_eng), local_unif(local_eng));
//             }
//         }
//     }
//     Kokkos::deep_copy(field_pruned_input.getView(), pruned_host);
//
//     // Initialize full frequency field with same values at mapped positions
//     for (size_t i = nghost; i < full_freq_host.extent(0) - nghost; ++i) {
//         for (size_t j = nghost; j < full_freq_host.extent(1) - nghost; ++j) {
//             for (size_t k = nghost; k < full_freq_host.extent(2) - nghost; ++k) {
//                 int gi_f = i - nghost + f0_first;
//                 int gj_f = j - nghost + f1_first;
//                 int gk_f = k - nghost + f2_first;
//
//                 // Reverse map: full index -> pruned index (if it exists)
//                 int gi_p = -1, gj_p = -1, gk_p = -1;
//
//                 if (gi_f < K0 / 2) {
//                     gi_p = gi_f;
//                 } else if (gi_f >= N0 - K0 / 2) {
//                     gi_p = gi_f - (N0 - K0);
//                 }
//
//                 if (gj_f < K1 / 2) {
//                     gj_p = gj_f;
//                 } else if (gj_f >= N1 - K1 / 2) {
//                     gj_p = gj_f - (N1 - K1);
//                 }
//
//                 if (gk_f < K2 / 2) {
//                     gk_p = gk_f;
//                 } else if (gk_f >= N2 - K2 / 2) {
//                     gk_p = gk_f - (N2 - K2);
//                 }
//
//                 if (gi_p >= 0 && gi_p < K0 && gj_p >= 0 && gj_p < K1 && gk_p >= 0 && gk_p < K2) {
//                     // This full index corresponds to a pruned mode
//                     std::mt19937_64 local_eng(gi_p * 10000 + gj_p * 100 + gk_p);
//                     std::uniform_real_distribution<double> local_unif(-1.0, 1.0);
//
//                     full_freq_host(i, j, k) =
//                         Kokkos::complex<double>(local_unif(local_eng), local_unif(local_eng));
//                 } else {
//                     full_freq_host(i, j, k) = Kokkos::complex<double>(0.0, 0.0);
//                 }
//             }
//         }
//     }
//     Kokkos::deep_copy(field_full_reference.getView(), full_freq_host);
//
//     // Apply backward transforms
//     pruned_fft->transform(ippl::BACKWARD, field_pruned_input, field_full_output);
//     regular_fft->transform(ippl::BACKWARD, field_full_reference);
//
//     // Compare results
//     auto view_output    = field_full_output.getView();
//     auto view_reference = field_full_reference.getView();
//
//     double max_error = 0.0;
//     size_t count     = 0;
//
//     Kokkos::parallel_reduce(
//         "CompareBackwardResults",
//         mdrange_t({nghost, nghost, nghost},
//                   {view_output.extent(0) - nghost, view_output.extent(1) - nghost,
//                    view_output.extent(2) - nghost}),
//         KOKKOS_LAMBDA(const int i, const int j, const int k, double& local_max) {
//             auto val_pruned = view_output(i, j, k);
//             auto val_full   = view_reference(i, j, k);
//
//             double error = Kokkos::abs(val_pruned - val_full);
//             if (error > local_max) {
//                 local_max = error;
//             }
//         },
//         Kokkos::Max<double>(max_error));
//
//     Kokkos::fence();
//
//     Kokkos::parallel_reduce(
//         "CountBackwardComparisons",
//         mdrange_t({nghost, nghost, nghost},
//                   {view_output.extent(0) - nghost, view_output.extent(1) - nghost,
//                    view_output.extent(2) - nghost}),
//         KOKKOS_LAMBDA(int, int, int, size_t& local_count) {
//             ++local_count;
//         },
//         count);
//
//     Kokkos::fence();
//
//     double global_max_error;
//     size_t global_count;
//     MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX,
//                   ippl::Comm->getCommunicator());
//     MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM,
//                   ippl::Comm->getCommunicator());
//
//     bool passed = (global_max_error < 1e-10);
//
//     if (ippl::Comm->rank() == 0) {
//         std::cout << "Compared " << global_count << " spatial points" << std::endl;
//         std::cout << "Max error vs full IFFT: " << std::scientific << std::setprecision(6)
//                   << global_max_error << std::endl;
//         std::cout << "Pruned C2C Backward FFT test: " << (passed ? "PASSED" : "FAILED")
//                   << std::endl;
//     }
//
//     // {
//     //     typename field_type::HostMirror host_output    = field_full_output.getHostMirror();
//     //     typename field_type::HostMirror host_reference = field_full_reference.getHostMirror();
//     //     Kokkos::deep_copy(host_output, field_full_output.getView());
//     //     Kokkos::deep_copy(host_reference, field_full_reference.getView());
//
//     //     if (ippl::Comm->rank() == 0) {
//     //         std::cout << "\n=== Comparing Pruned Backward vs Full Backward FFT values ==="
//     //                   << std::endl;
//     //         std::cout << std::setw(20) << "Spatial Index" << std::setw(35) << "Pruned IFFT"
//     //                   << std::setw(35) << "Full IFFT" << std::setw(15) << "Error" << std::endl;
//     //         std::cout << std::string(105, '-') << std::endl;
//     //
//     //         // Print a subset to avoid overwhelming output
//     //         int print_limit = 10;  // per dimension, or set to pt_full[d] for all
//     //         int printed     = 0;
//     //         int max_print   = 100;  // total max entries to print
//     //
//     //         for (int i = 0; i < pt_full[0] && printed < max_print; ++i) {
//     //             for (int j = 0; j < pt_full[1] && printed < max_print; ++j) {
//     //                 for (int k = 0; k < pt_full[2] && printed < max_print; ++k) {
//     //                     // Only print if this index is local to rank 0
//     //                     if (i < f0_first || i > f0_last || j < f1_first || j > f1_last
//     //                         || k < f2_first || k > f2_last) {
//     //                         continue;
//     //                     }
//     //
//     //                     int li = i - f0_first + nghost;
//     //                     int lj = j - f1_first + nghost;
//     //                     int lk = k - f2_first + nghost;
//     //
//     //                     auto val_pruned = host_output(li, lj, lk);
//     //                     auto val_full   = host_reference(li, lj, lk);
//     //                     double err      = Kokkos::abs(val_pruned - val_full);
//     //
//     //                     std::cout << std::setw(6) << "(" << i << "," << j << "," << k << ")"
//     //                               << std::setw(15) << std::fixed << std::setprecision(6) << "("
//     //                               << val_pruned.real() << ", " << val_pruned.imag() << ")"
//     //                               << std::setw(15) << "(" << val_full.real() << ", "
//     //                               << val_full.imag() << ")" << std::setw(15) << std::scientific
//     //                               << err << std::endl;
//     //
//     //                     ++printed;
//     //                 }
//     //             }
//     //         }
//     //
//     //         // Also print some samples from the middle and end
//     //         std::cout << "\n--- Samples from middle/end of domain ---" << std::endl;
//     //
//     //         std::vector<std::array<int, 3>> sample_points = {
//     //             {N0 / 4, N1 / 4, N2 / 4},
//     //             {N0 / 2, N1 / 2, N2 / 2},
//     //             {3 * N0 / 4, 3 * N1 / 4, 3 * N2 / 4},
//     //             {N0 - 1, N1 - 1, N2 - 1},
//     //             {0, N1 / 2, N2 - 1},
//     //             {N0 / 2, 0, N2 / 2},
//     //         };
//     //
//     //         for (const auto& pt : sample_points) {
//     //             int i = pt[0], j = pt[1], k = pt[2];
//     //
//     //             if (i < f0_first || i > f0_last || j < f1_first || j > f1_last || k < f2_first
//     //                 || k > f2_last) {
//     //                 continue;
//     //             }
//     //
//     //             int li = i - f0_first + nghost;
//     //             int lj = j - f1_first + nghost;
//     //             int lk = k - f2_first + nghost;
//     //
//     //             auto val_pruned = host_output(li, lj, lk);
//     //             auto val_full   = host_reference(li, lj, lk);
//     //             double err      = Kokkos::abs(val_pruned - val_full);
//     //
//     //             std::cout << std::setw(6) << "(" << i << "," << j << "," << k << ")"
//     //                       << std::setw(15) << std::fixed << std::setprecision(6) << "("
//     //                       << val_pruned.real() << ", " << val_pruned.imag() << ")" << std::setw(15)
//     //                       << "(" << val_full.real() << ", " << val_full.imag() << ")"
//     //                       << std::setw(15) << std::scientific << err << std::endl;
//     //         }
//     //
//     //         std::cout << std::endl;
//     //     }
//     // }
//
//     return passed;
// }
//
// // Test pruned R2C FFT by comparing with full R2C FFT
// bool testPrunedRC() {
//     constexpr unsigned int dim = 3;
//     using Mesh_t               = ippl::UniformCartesian<double, dim>;
//     using Centering_t          = Mesh_t::DefaultCentering;
//
//     std::array<int, dim> pt_real           = {64, 64, 64};
//     std::array<int, dim> pt_complex_full   = {33, 64, 64};  // R2C in dimension 0: N/2+1
//     std::array<int, dim> pt_complex_pruned = {17, 32, 32};  // Pruned modes
//
//     // Create layouts
//     ippl::Index I_real(pt_real[0]);
//     ippl::Index J_real(pt_real[1]);
//     ippl::Index K_real(pt_real[2]);
//     ippl::NDIndex<dim> owned_real(I_real, J_real, K_real);
//
//     ippl::Index I_cfull(pt_complex_full[0]);
//     ippl::Index J_cfull(pt_complex_full[1]);
//     ippl::Index K_cfull(pt_complex_full[2]);
//     ippl::NDIndex<dim> owned_complex_full(I_cfull, J_cfull, K_cfull);
//
//     ippl::Index I_cpruned(pt_complex_pruned[0]);
//     ippl::Index J_cpruned(pt_complex_pruned[1]);
//     ippl::Index K_cpruned(pt_complex_pruned[2]);
//     ippl::NDIndex<dim> owned_complex_pruned(I_cpruned, J_cpruned, K_cpruned);
//
//     std::array<bool, dim> isParallel;
//     isParallel.fill(true);
//
//     ippl::FieldLayout<dim> layout_real(MPI_COMM_WORLD, owned_real, isParallel);
//     ippl::FieldLayout<dim> layout_complex_full(MPI_COMM_WORLD, owned_complex_full, isParallel);
//     ippl::FieldLayout<dim> layout_complex_pruned(MPI_COMM_WORLD, owned_complex_pruned, isParallel);
//
//     std::array<double, dim> dx = {
//         1.0 / double(pt_real[0]),
//         1.0 / double(pt_real[1]),
//         1.0 / double(pt_real[2]),
//     };
//     ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
//     ippl::Vector<double, 3> origin = {0, 0, 0};
//
//     Mesh_t mesh_real(owned_real, hx, origin);
//     Mesh_t mesh_complex_full(owned_complex_full, hx, origin);
//     Mesh_t mesh_complex_pruned(owned_complex_pruned, hx, origin);
//
//     typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type field_type_real;
//     typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type
//         field_type_complex;
//
//     field_type_real field_real_input(mesh_real, layout_real);
//     field_type_real field_real_copy(mesh_real, layout_real);
//     field_type_complex field_complex_full(mesh_complex_full, layout_complex_full);
//     field_type_complex field_complex_pruned(mesh_complex_pruned, layout_complex_pruned);
//
//     // Setup pruning parameters
//     ippl::PruningParams<dim> pruning;
//     pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_complex_pruned[0]),
//                                                 static_cast<size_t>(pt_complex_pruned[1]),
//                                                 static_cast<size_t>(pt_complex_pruned[2])};
//
//     ippl::ParameterList fftParams;
//     fftParams.add("use_heffte_defaults", true);
//     fftParams.add("r2c_direction", 0);
//
//     // Create FFTs
//     typedef ippl::FFT<ippl::PrunedRCTransform, field_type_real> PrunedRCFFT_type;
//     typedef ippl::FFT<ippl::RCTransform, field_type_real> RCFFT_type;
//
//     auto pruned_fft  = std::make_unique<PrunedRCFFT_type>(layout_real, layout_complex_full,
//                                                           layout_complex_pruned, pruning, fftParams);
//     auto regular_fft = std::make_unique<RCFFT_type>(layout_real, layout_complex_full, fftParams);
//
//     if (ippl::Comm->rank() == 0) {
//         std::cout << "\n=== Testing Pruned R2C FFT ===" << std::endl;
//         std::cout << "Real grid: " << pt_real[0] << "x" << pt_real[1] << "x" << pt_real[2]
//                   << std::endl;
//         std::cout << "Full complex: " << pt_complex_full[0] << "x" << pt_complex_full[1] << "x"
//                   << pt_complex_full[2] << std::endl;
//         std::cout << "Pruned complex: " << pt_complex_pruned[0] << "x" << pt_complex_pruned[1]
//                   << "x" << pt_complex_pruned[2] << std::endl;
//     }
//
//     // Initialize with random data
//     const int nghost                                = field_real_input.getNghost();
//     auto& view_real                                 = field_real_input.getView();
//     typename field_type_real::HostMirror field_host = field_real_input.getHostMirror();
//
//     std::mt19937_64 eng(42 + ippl::Comm->rank());
//     std::uniform_real_distribution<double> unif(-1.0, 1.0);
//
//     for (size_t i = nghost; i < view_real.extent(0) - nghost; ++i) {
//         for (size_t j = nghost; j < view_real.extent(1) - nghost; ++j) {
//             for (size_t k = nghost; k < view_real.extent(2) - nghost; ++k) {
//                 field_host(i, j, k) = unif(eng);
//             }
//         }
//     }
//     Kokkos::deep_copy(field_real_input.getView(), field_host);
//
//     // Save input and compute both FFTs
//     field_real_copy = field_real_input;
//
//     pruned_fft->transform(ippl::FORWARD, field_real_input, field_complex_pruned);
//     regular_fft->transform(ippl::FORWARD, field_real_copy, field_complex_full);
//
//     // Compare pruned result with corresponding modes from full result
//     // R2C: dimension 0 has only positive freqs [0, N/2], direct mapping
//     //      dimensions 1,2 have [0, ..., N/2-1, -N/2, ..., -1], need wrapping
//     auto view_pruned = field_complex_pruned.getView();
//     auto view_full   = field_complex_full.getView();
//
//     const auto& lDom_pruned = layout_complex_pruned.getLocalNDIndex();
//     const auto& lDom_full   = layout_complex_full.getLocalNDIndex();
//     const int nghost_pruned = field_complex_pruned.getNghost();
//     const int nghost_full   = field_complex_full.getNghost();
//
//     // Full complex dimensions (N/2+1, N, N)
//     const int N1 = pt_real[1], K1 = pt_complex_pruned[1];
//     const int N2 = pt_real[2], K2 = pt_complex_pruned[2];
//
//     const int p0_first = lDom_pruned[0].first();
//     const int p1_first = lDom_pruned[1].first();
//     const int p2_first = lDom_pruned[2].first();
//
//     const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
//     const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
//     const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();
//
//     const int ng_p = nghost_pruned;
//     const int ng_f = nghost_full;
//
//     double max_error = 0.0;
//     size_t count     = 0;
//
//     using exec_space = typename field_type_complex::execution_space;
//     using mdrange_t  = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;
//
//     Kokkos::parallel_reduce(
//         "ComparePrunedWithFullRC",
//         mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
//                                        view_pruned.extent(2) - ng_p}),
//         KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, double& local_max) {
//             int gi_p = li_p - ng_p + p0_first;
//             int gj_p = lj_p - ng_p + p1_first;
//             int gk_p = lk_p - ng_p + p2_first;
//
//             // Dimension 0: direct mapping (R2C only has positive freqs)
//             int gi_f = gi_p;
//             // Dimensions 1,2: wrap negative frequencies
//             int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
//             int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);
//
//             if (gi_f < f0_first || gi_f > f0_last || gj_f < f1_first || gj_f > f1_last
//                 || gk_f < f2_first || gk_f > f2_last) {
//                 return;
//             }
//
//             int li_f = gi_f - f0_first + ng_f;
//             int lj_f = gj_f - f1_first + ng_f;
//             int lk_f = gk_f - f2_first + ng_f;
//
//             auto val_pruned = view_pruned(li_p, lj_p, lk_p);
//             auto val_full   = view_full(li_f, lj_f, lk_f);
//
//             double error = Kokkos::abs(val_pruned - val_full);
//             if (error > local_max) {
//                 local_max = error;
//             }
//         },
//         Kokkos::Max<double>(max_error));
//
//     Kokkos::fence();
//
//     Kokkos::parallel_reduce(
//         "CountComparisonsRC",
//         mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
//                                        view_pruned.extent(2) - ng_p}),
//         KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, size_t& local_count) {
//             int gi_p = li_p - ng_p + p0_first;
//             int gj_p = lj_p - ng_p + p1_first;
//             int gk_p = lk_p - ng_p + p2_first;
//
//             int gi_f = gi_p;
//             int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
//             int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);
//
//             if (gi_f >= f0_first && gi_f <= f0_last && gj_f >= f1_first && gj_f <= f1_last
//                 && gk_f >= f2_first && gk_f <= f2_last) {
//                 ++local_count;
//             }
//         },
//         count);
//
//     Kokkos::fence();
//
//     double global_max_error;
//     size_t global_count;
//     MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX,
//                   ippl::Comm->getCommunicator());
//     MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM,
//                   ippl::Comm->getCommunicator());
//
//     bool passed = (global_max_error < 1e-10);
//
//     if (ippl::Comm->rank() == 0) {
//         std::cout << "Compared " << global_count << " modes" << std::endl;
//         std::cout << "Max error vs full R2C FFT: " << std::scientific << std::setprecision(6)
//                   << global_max_error << std::endl;
//         std::cout << "Pruned R2C FFT test: " << (passed ? "PASSED" : "FAILED") << std::endl;
//     }
//
//     return passed;
// }
//
// int main(int argc, char* argv[]) {
//     ippl::initialize(argc, argv);
//
//     bool cc_passed      = testPrunedCC();
//     bool cc_back_passed = testPrunedCCBackward();
//     bool rc_passed = testPrunedRC();
//
//     if (ippl::Comm->rank() == 0) {
//         std::cout << "\n=== Overall Results ===" << std::endl;
//         std::cout << "Pruned C2C: " << (cc_passed ? "PASSED" : "FAILED") << std::endl;
//         std::cout << "Pruned C2C backwards: " << (cc_back_passed ? "PASSED" : "FAILED")
//                   << std::endl;
//         std::cout << "Pruned R2C: " << (rc_passed ? "PASSED" : "FAILED") << std::endl;
//         std::cout << "All tests: " << ((cc_passed && rc_passed) ? "PASSED" : "FAILED") << std::endl;
//     }
//
//     ippl::finalize();
//     return (cc_passed && rc_passed && cc_back_passed) ? 0 : 1;
// }

#include "Ippl.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "Utility/ParameterList.h"

// Helper function to compute field statistics
template <typename Field>
void printFieldStats(const std::string& name, Field& field, int rank) {
    auto view = field.getView();
    const int nghost = field.getNghost();

    using value_type = typename Field::value_type;

    double sum_real = 0.0, sum_imag = 0.0;
    double max_abs = 0.0;
    size_t count = 0;

    using exec_space = typename Field::execution_space;
    using mdrange_t = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    // For complex fields
    if constexpr (std::is_same_v<value_type, Kokkos::complex<double>> ||
                  std::is_same_v<value_type, Kokkos::complex<float>>) {
        Kokkos::parallel_reduce(
            "FieldStats",
            mdrange_t({nghost, nghost, nghost},
                      {view.extent(0) - nghost, view.extent(1) - nghost, view.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                         double& lsum_r, double& lsum_i, double& lmax, size_t& lcount) {
                auto val = view(i, j, k);
                lsum_r += val.real();
                lsum_i += val.imag();
                double abs_val = Kokkos::abs(val);
                if (abs_val > lmax) lmax = abs_val;
                ++lcount;
            },
            sum_real, sum_imag, Kokkos::Max<double>(max_abs), count);
    } else {
        // For real fields
        Kokkos::parallel_reduce(
            "FieldStats",
            mdrange_t({nghost, nghost, nghost},
                      {view.extent(0) - nghost, view.extent(1) - nghost, view.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                         double& lsum_r, double& /*lsum_i*/, double& lmax, size_t& lcount) {
                double val = view(i, j, k);
                lsum_r += val;
                double abs_val = Kokkos::fabs(val);
                if (abs_val > lmax) lmax = abs_val;
                ++lcount;
            },
            sum_real, sum_imag, Kokkos::Max<double>(max_abs), count);
    }

    Kokkos::fence();

    // Gather global statistics
    double global_sum_real, global_sum_imag, global_max_abs;
    size_t global_count;
    MPI_Allreduce(&sum_real, &global_sum_real, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
    MPI_Allreduce(&sum_imag, &global_sum_imag, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_abs, &global_max_abs, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, ippl::Comm->getCommunicator());

    if (rank == 0) {
        std::cout << "  " << name << ": count=" << global_count
                  << ", sum=(" << std::scientific << std::setprecision(6)
                  << global_sum_real << ", " << global_sum_imag << ")"
                  << ", max_abs=" << global_max_abs << std::endl;
    }
}

// Print first few values of a field
template <typename Field>
void printFieldSample(const std::string& name, Field& field, int rank, int max_print = 5) {
    if (rank != 0) return;

    auto host_view = field.getHostMirror();
    Kokkos::deep_copy(host_view, field.getView());

    const int nghost = field.getNghost();

    std::cout << "  " << name << " first " << max_print << " values:" << std::endl;
    int printed = 0;
    for (size_t i = nghost; i < host_view.extent(0) - nghost && printed < max_print; ++i) {
        for (size_t j = nghost; j < host_view.extent(1) - nghost && printed < max_print; ++j) {
            for (size_t k = nghost; k < host_view.extent(2) - nghost && printed < max_print; ++k) {
                auto val = host_view(i, j, k);
                if constexpr (std::is_same_v<std::decay_t<decltype(val)>, Kokkos::complex<double>> ||
                              std::is_same_v<std::decay_t<decltype(val)>, Kokkos::complex<float>>) {
                    std::cout << "    [" << i-nghost << "," << j-nghost << "," << k-nghost << "] = ("
                              << std::scientific << std::setprecision(6)
                              << val.real() << ", " << val.imag() << ")" << std::endl;
                } else {
                    std::cout << "    [" << i-nghost << "," << j-nghost << "," << k-nghost << "] = "
                              << std::scientific << std::setprecision(6) << val << std::endl;
                }
                ++printed;
            }
        }
    }
}

// Test pruned C2C FFT by comparing with full FFT
bool testPrunedCC() {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;

    std::array<int, dim> pt_full   = {64, 128, 32};
    std::array<int, dim> pt_pruned = {32, 64, 16};

    int myRank = ippl::Comm->rank();
    int nProcs = ippl::Comm->size();

    if (myRank == 0) {
        std::cout << "\n=== Testing Pruned C2C FFT ===" << std::endl;
        std::cout << "Running on " << nProcs << " MPI ranks" << std::endl;
        std::cout << "Full grid: " << pt_full[0] << "x" << pt_full[1] << "x" << pt_full[2] << std::endl;
        std::cout << "Pruned to: " << pt_pruned[0] << "x" << pt_pruned[1] << "x" << pt_pruned[2] << std::endl;
    }

    // Create layouts
    ippl::Index I_full(pt_full[0]);
    ippl::Index J_full(pt_full[1]);
    ippl::Index K_full(pt_full[2]);
    ippl::NDIndex<dim> owned_full(I_full, J_full, K_full);

    ippl::Index I_pruned(pt_pruned[0]);
    ippl::Index J_pruned(pt_pruned[1]);
    ippl::Index K_pruned(pt_pruned[2]);
    ippl::NDIndex<dim> owned_pruned(I_pruned, J_pruned, K_pruned);

    std::array<bool, dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<dim> layout_full(MPI_COMM_WORLD, owned_full, isParallel);
    ippl::FieldLayout<dim> layout_pruned(MPI_COMM_WORLD, owned_pruned, isParallel);

    // Debug: print layout info
    const auto& lDom_full = layout_full.getLocalNDIndex();
    const auto& lDom_pruned = layout_pruned.getLocalNDIndex();

    std::cout << "Rank " << myRank << " layout_full: ["
              << lDom_full[0].first() << "-" << lDom_full[0].last() << ", "
              << lDom_full[1].first() << "-" << lDom_full[1].last() << ", "
              << lDom_full[2].first() << "-" << lDom_full[2].last() << "]" << std::endl;
    std::cout << "Rank " << myRank << " layout_pruned: ["
              << lDom_pruned[0].first() << "-" << lDom_pruned[0].last() << ", "
              << lDom_pruned[1].first() << "-" << lDom_pruned[1].last() << ", "
              << lDom_pruned[2].first() << "-" << lDom_pruned[2].last() << "]" << std::endl;

    std::array<double, dim> dx = {
        1.0 / double(pt_full[0]),
        1.0 / double(pt_full[1]),
        1.0 / double(pt_full[2]),
    };
    ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    Mesh_t mesh_full(owned_full, hx, origin);
    Mesh_t mesh_pruned(owned_pruned, hx, origin);

    typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;

    field_type field_input(mesh_full, layout_full);
    field_type field_full_result(mesh_full, layout_full);
    field_type field_pruned_result(mesh_pruned, layout_pruned);

    // Setup pruning parameters
    ippl::PruningParams<dim> pruning;
    pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_pruned[0]),
                                                static_cast<size_t>(pt_pruned[1]),
                                                static_cast<size_t>(pt_pruned[2])};

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", true);

    // Create FFTs
    typedef ippl::FFT<ippl::PrunedCCTransform, field_type> PrunedFFT_type;
    typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;

    if (myRank == 0) std::cout << "Creating pruned FFT..." << std::endl;
    auto pruned_fft = std::make_unique<PrunedFFT_type>(layout_full, layout_pruned, pruning, fftParams);

    if (myRank == 0) std::cout << "Creating regular FFT..." << std::endl;
    auto regular_fft = std::make_unique<FFT_type>(layout_full, fftParams);

    // Initialize with random data using FIXED seed for reproducibility
    const int nghost = field_input.getNghost();
    auto& view_full = field_input.getView();
    typename field_type::HostMirror field_host = field_input.getHostMirror();

    // Use global index as seed for deterministic values across ranks
    for (size_t i = nghost; i < view_full.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view_full.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view_full.extent(2) - nghost; ++k) {
                int gi = i - nghost + lDom_full[0].first();
                int gj = j - nghost + lDom_full[1].first();
                int gk = k - nghost + lDom_full[2].first();

                // Deterministic based on global index
                std::mt19937_64 local_eng(gi * 100000 + gj * 1000 + gk + 42);
                std::uniform_real_distribution<double> unif(-1.0, 1.0);
                field_host(i, j, k) = Kokkos::complex<double>(unif(local_eng), unif(local_eng));
            }
        }
    }
    Kokkos::deep_copy(field_input.getView(), field_host);

    // Print input statistics
    if (myRank == 0) std::cout << "\nInput field statistics:" << std::endl;
    printFieldStats("field_input", field_input, myRank);
    printFieldSample("field_input", field_input, myRank);

    // Save input and compute both FFTs
    field_full_result = field_input;

    if (myRank == 0) std::cout << "\nExecuting pruned FFT forward..." << std::endl;
    MPI_Barrier(ippl::Comm->getCommunicator());
    pruned_fft->transform(ippl::FORWARD, field_input, field_pruned_result);
    MPI_Barrier(ippl::Comm->getCommunicator());

    if (myRank == 0) std::cout << "Executing regular FFT forward..." << std::endl;
    MPI_Barrier(ippl::Comm->getCommunicator());
    regular_fft->transform(ippl::FORWARD, field_full_result);
    MPI_Barrier(ippl::Comm->getCommunicator());

    // Print output statistics
    if (myRank == 0) std::cout << "\nOutput field statistics:" << std::endl;
    printFieldStats("field_pruned_result", field_pruned_result, myRank);
    printFieldStats("field_full_result", field_full_result, myRank);
    printFieldSample("field_pruned_result", field_pruned_result, myRank);
    printFieldSample("field_full_result", field_full_result, myRank);

    // Compare pruned result with corresponding modes from full result
    auto view_pruned = field_pruned_result.getView();
    auto view_full_result = field_full_result.getView();

    const int nghost_pruned = field_pruned_result.getNghost();

    const int N0 = pt_full[0], K0 = pt_pruned[0];
    const int N1 = pt_full[1], K1 = pt_pruned[1];
    const int N2 = pt_full[2], K2 = pt_pruned[2];

    const int p0_first = lDom_pruned[0].first();
    const int p1_first = lDom_pruned[1].first();
    const int p2_first = lDom_pruned[2].first();

    const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
    const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
    const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();

    const int ng = nghost;
    const int ng_p = nghost_pruned;

    // Find worst errors with their indices
    double max_error = 0.0;
    size_t count = 0;

    using exec_space = typename field_type::execution_space;
    using mdrange_t = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    // First pass: find max error
    Kokkos::parallel_reduce(
        "ComparePrunedWithFull",
        mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
                                       view_pruned.extent(2) - ng_p}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, double& local_max) {
            int gi_p = li_p - ng_p + p0_first;
            int gj_p = lj_p - ng_p + p1_first;
            int gk_p = lk_p - ng_p + p2_first;

            int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f < f0_first || gi_f > f0_last || gj_f < f1_first || gj_f > f1_last
                || gk_f < f2_first || gk_f > f2_last) {
                return;
            }

            int li_f = gi_f - f0_first + ng;
            int lj_f = gj_f - f1_first + ng;
            int lk_f = gk_f - f2_first + ng;

            auto val_pruned = view_pruned(li_p, lj_p, lk_p);
            auto val_full = view_full_result(li_f, lj_f, lk_f);

            double error = Kokkos::abs(val_pruned - val_full);
            if (error > local_max) {
                local_max = error;
            }
        },
        Kokkos::Max<double>(max_error));

    Kokkos::fence();

    // Count comparisons
    Kokkos::parallel_reduce(
        "CountComparisons",
        mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
                                       view_pruned.extent(2) - ng_p}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, size_t& local_count) {
            int gi_p = li_p - ng_p + p0_first;
            int gj_p = lj_p - ng_p + p1_first;
            int gk_p = lk_p - ng_p + p2_first;

            int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f >= f0_first && gi_f <= f0_last && gj_f >= f1_first && gj_f <= f1_last
                && gk_f >= f2_first && gk_f <= f2_last) {
                ++local_count;
            }
        },
        count);

    Kokkos::fence();

    double global_max_error;
    size_t global_count;
    MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, ippl::Comm->getCommunicator());

    // Print worst mismatches on each rank
    if (max_error > 1e-10) {
        auto host_pruned = field_pruned_result.getHostMirror();
        auto host_full = field_full_result.getHostMirror();
        Kokkos::deep_copy(host_pruned, field_pruned_result.getView());
        Kokkos::deep_copy(host_full, field_full_result.getView());

        std::cout << "\nRank " << myRank << " worst mismatches (error > 1e-10):" << std::endl;
        int printed = 0;
        int max_print = 10;

        for (size_t li_p = ng_p; li_p < host_pruned.extent(0) - ng_p && printed < max_print; ++li_p) {
            for (size_t lj_p = ng_p; lj_p < host_pruned.extent(1) - ng_p && printed < max_print; ++lj_p) {
                for (size_t lk_p = ng_p; lk_p < host_pruned.extent(2) - ng_p && printed < max_print; ++lk_p) {
                    int gi_p = li_p - ng_p + p0_first;
                    int gj_p = lj_p - ng_p + p1_first;
                    int gk_p = lk_p - ng_p + p2_first;

                    int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
                    int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
                    int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

                    if (gi_f < f0_first || gi_f > f0_last || gj_f < f1_first || gj_f > f1_last
                        || gk_f < f2_first || gk_f > f2_last) {
                        continue;
                    }

                    int li_f = gi_f - f0_first + ng;
                    int lj_f = gj_f - f1_first + ng;
                    int lk_f = gk_f - f2_first + ng;

                    auto val_pruned = host_pruned(li_p, lj_p, lk_p);
                    auto val_full = host_full(li_f, lj_f, lk_f);
                    double error = Kokkos::abs(val_pruned - val_full);

                    if (error > 1e-10) {
                        std::cout << "  pruned[" << gi_p << "," << gj_p << "," << gk_p << "]="
                                  << "(" << val_pruned.real() << "," << val_pruned.imag() << ")"
                                  << " vs full[" << gi_f << "," << gj_f << "," << gk_f << "]="
                                  << "(" << val_full.real() << "," << val_full.imag() << ")"
                                  << " error=" << std::scientific << error << std::endl;
                        ++printed;
                    }
                }
            }
        }
    }

    bool passed = (global_max_error < 1e-10);

    if (myRank == 0) {
        std::cout << "\n=== Pruned C2C FFT Results ===" << std::endl;
        std::cout << "Compared " << global_count << " modes" << std::endl;
        std::cout << "Max error vs full FFT: " << std::scientific << std::setprecision(6)
                  << global_max_error << std::endl;
        std::cout << "Pruned C2C FFT test: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    return passed;
}

bool testPrunedCCBackward() {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;

    std::array<int, dim> pt_full   = {16, 64, 32};
    std::array<int, dim> pt_pruned = {8, 32, 16};

    int myRank = ippl::Comm->rank();
    int nProcs = ippl::Comm->size();

    if (myRank == 0) {
        std::cout << "\n=== Testing Pruned C2C Backward FFT ===" << std::endl;
        std::cout << "Running on " << nProcs << " MPI ranks" << std::endl;
        std::cout << "Pruned frequency: " << pt_pruned[0] << "x" << pt_pruned[1] << "x" << pt_pruned[2] << std::endl;
        std::cout << "Full spatial: " << pt_full[0] << "x" << pt_full[1] << "x" << pt_full[2] << std::endl;
    }

    // Create layouts
    ippl::Index I_full(pt_full[0]);
    ippl::Index J_full(pt_full[1]);
    ippl::Index K_full(pt_full[2]);
    ippl::NDIndex<dim> owned_full(I_full, J_full, K_full);

    ippl::Index I_pruned(pt_pruned[0]);
    ippl::Index J_pruned(pt_pruned[1]);
    ippl::Index K_pruned(pt_pruned[2]);
    ippl::NDIndex<dim> owned_pruned(I_pruned, J_pruned, K_pruned);

    std::array<bool, dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<dim> layout_full(MPI_COMM_WORLD, owned_full, isParallel);
    ippl::FieldLayout<dim> layout_pruned(MPI_COMM_WORLD, owned_pruned, isParallel);

    // Debug: print layout info
    const auto& lDom_full = layout_full.getLocalNDIndex();
    const auto& lDom_pruned = layout_pruned.getLocalNDIndex();

    std::cout << "Rank " << myRank << " layout_full: ["
              << lDom_full[0].first() << "-" << lDom_full[0].last() << ", "
              << lDom_full[1].first() << "-" << lDom_full[1].last() << ", "
              << lDom_full[2].first() << "-" << lDom_full[2].last() << "]" << std::endl;
    std::cout << "Rank " << myRank << " layout_pruned: ["
              << lDom_pruned[0].first() << "-" << lDom_pruned[0].last() << ", "
              << lDom_pruned[1].first() << "-" << lDom_pruned[1].last() << ", "
              << lDom_pruned[2].first() << "-" << lDom_pruned[2].last() << "]" << std::endl;

    std::array<double, dim> dx = {
        1.0 / double(pt_full[0]),
        1.0 / double(pt_full[1]),
        1.0 / double(pt_full[2]),
    };
    ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    Mesh_t mesh_full(owned_full, hx, origin);
    Mesh_t mesh_pruned(owned_pruned, hx, origin);

    typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;

    field_type field_pruned_input(mesh_pruned, layout_pruned);
    field_type field_full_output(mesh_full, layout_full);
    field_type field_full_freq(mesh_full, layout_full);
    field_type field_full_reference(mesh_full, layout_full);

    // Setup pruning parameters
    ippl::PruningParams<dim> pruning;
    pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_pruned[0]),
                                                static_cast<size_t>(pt_pruned[1]),
                                                static_cast<size_t>(pt_pruned[2])};

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", true);

    typedef ippl::FFT<ippl::PrunedCCTransform, field_type> PrunedFFT_type;
    typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;

    if (myRank == 0) std::cout << "Creating FFTs..." << std::endl;
    auto pruned_fft = std::make_unique<PrunedFFT_type>(layout_full, layout_pruned, pruning, fftParams);
    auto regular_fft = std::make_unique<FFT_type>(layout_full, fftParams);

    const int nghost = field_full_output.getNghost();
    const int nghost_pruned = field_pruned_input.getNghost();

    const int N0 = pt_full[0], K0 = pt_pruned[0];
    const int N1 = pt_full[1], K1 = pt_pruned[1];
    const int N2 = pt_full[2], K2 = pt_pruned[2];

    const int p0_first = lDom_pruned[0].first();
    const int p1_first = lDom_pruned[1].first();
    const int p2_first = lDom_pruned[2].first();

    const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
    const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
    const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();

    // Initialize fields with deterministic data
    typename field_type::HostMirror pruned_host = field_pruned_input.getHostMirror();
    typename field_type::HostMirror full_freq_host = field_full_freq.getHostMirror();

    if (myRank == 0) std::cout << "Initializing input fields..." << std::endl;

    // Initialize pruned field with deterministic values
    auto& view_pruned = field_pruned_input.getView();
    for (size_t i = nghost_pruned; i < view_pruned.extent(0) - nghost_pruned; ++i) {
        for (size_t j = nghost_pruned; j < view_pruned.extent(1) - nghost_pruned; ++j) {
            for (size_t k = nghost_pruned; k < view_pruned.extent(2) - nghost_pruned; ++k) {
                int gi_p = i - nghost_pruned + p0_first;
                int gj_p = j - nghost_pruned + p1_first;
                int gk_p = k - nghost_pruned + p2_first;

                std::mt19937_64 local_eng(gi_p * 10000 + gj_p * 100 + gk_p + 123);
                std::uniform_real_distribution<double> local_unif(-1.0, 1.0);
                pruned_host(i, j, k) = Kokkos::complex<double>(local_unif(local_eng), local_unif(local_eng));
            }
        }
    }
    Kokkos::deep_copy(field_pruned_input.getView(), pruned_host);

    // Initialize full frequency field
    Kokkos::deep_copy(full_freq_host, Kokkos::complex<double>(0.0, 0.0));

    for (size_t i = nghost; i < full_freq_host.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < full_freq_host.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < full_freq_host.extent(2) - nghost; ++k) {
                int gi_f = i - nghost + f0_first;
                int gj_f = j - nghost + f1_first;
                int gk_f = k - nghost + f2_first;

                // Reverse map: full index -> pruned index
                int gi_p = -1, gj_p = -1, gk_p = -1;

                if (gi_f < K0 / 2) {
                    gi_p = gi_f;
                } else if (gi_f >= N0 - K0 / 2) {
                    gi_p = gi_f - (N0 - K0);
                }

                if (gj_f < K1 / 2) {
                    gj_p = gj_f;
                } else if (gj_f >= N1 - K1 / 2) {
                    gj_p = gj_f - (N1 - K1);
                }

                if (gk_f < K2 / 2) {
                    gk_p = gk_f;
                } else if (gk_f >= N2 - K2 / 2) {
                    gk_p = gk_f - (N2 - K2);
                }

                if (gi_p >= 0 && gi_p < K0 && gj_p >= 0 && gj_p < K1 && gk_p >= 0 && gk_p < K2) {
                    std::mt19937_64 local_eng(gi_p * 10000 + gj_p * 100 + gk_p + 123);
                    std::uniform_real_distribution<double> local_unif(-1.0, 1.0);
                    full_freq_host(i, j, k) = Kokkos::complex<double>(local_unif(local_eng), local_unif(local_eng));
                } else {
                    full_freq_host(i, j, k) = Kokkos::complex<double>(0.0, 0.0);
                }
            }
        }
    }
    Kokkos::deep_copy(field_full_reference.getView(), full_freq_host);

    // Print input statistics
    if (myRank == 0) std::cout << "\nInput field statistics:" << std::endl;
    printFieldStats("field_pruned_input", field_pruned_input, myRank);
    printFieldStats("field_full_reference (freq)", field_full_reference, myRank);

    // Apply backward transforms
    if (myRank == 0) std::cout << "\nExecuting pruned backward FFT..." << std::endl;
    MPI_Barrier(ippl::Comm->getCommunicator());
    pruned_fft->transform(ippl::BACKWARD, field_pruned_input, field_full_output);
    MPI_Barrier(ippl::Comm->getCommunicator());

    if (myRank == 0) std::cout << "Executing regular backward FFT..." << std::endl;
    MPI_Barrier(ippl::Comm->getCommunicator());
    regular_fft->transform(ippl::BACKWARD, field_full_reference);
    MPI_Barrier(ippl::Comm->getCommunicator());

    // Print output statistics
    if (myRank == 0) std::cout << "\nOutput field statistics:" << std::endl;
    printFieldStats("field_full_output (pruned)", field_full_output, myRank);
    printFieldStats("field_full_reference (regular)", field_full_reference, myRank);

    // Compare results
    auto view_output = field_full_output.getView();
    auto view_reference = field_full_reference.getView();

    double max_error = 0.0;
    size_t count = 0;

    using exec_space = typename field_type::execution_space;
    using mdrange_t = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    Kokkos::parallel_reduce(
        "CompareBackwardResults",
        mdrange_t({nghost, nghost, nghost},
                  {view_output.extent(0) - nghost, view_output.extent(1) - nghost, view_output.extent(2) - nghost}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, double& local_max) {
            auto val_pruned = view_output(i, j, k);
            auto val_full = view_reference(i, j, k);
            double error = Kokkos::abs(val_pruned - val_full);
            if (error > local_max) {
                local_max = error;
            }
        },
        Kokkos::Max<double>(max_error));

    Kokkos::fence();

    Kokkos::parallel_reduce(
        "CountBackwardComparisons",
        mdrange_t({nghost, nghost, nghost},
                  {view_output.extent(0) - nghost, view_output.extent(1) - nghost, view_output.extent(2) - nghost}),
        KOKKOS_LAMBDA(int, int, int, size_t& local_count) { ++local_count; },
        count);

    Kokkos::fence();

    double global_max_error;
    size_t global_count;
    MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, ippl::Comm->getCommunicator());

    // Print worst mismatches if error is large
    if (max_error > 1e-10) {
        auto host_output = field_full_output.getHostMirror();
        auto host_reference = field_full_reference.getHostMirror();
        Kokkos::deep_copy(host_output, field_full_output.getView());
        Kokkos::deep_copy(host_reference, field_full_reference.getView());

        std::cout << "\nRank " << myRank << " worst backward mismatches (error > 1e-10):" << std::endl;
        int printed = 0;
        int max_print = 10;

        for (size_t i = nghost; i < host_output.extent(0) - nghost && printed < max_print; ++i) {
            for (size_t j = nghost; j < host_output.extent(1) - nghost && printed < max_print; ++j) {
                for (size_t k = nghost; k < host_output.extent(2) - nghost && printed < max_print; ++k) {
                    auto val_pruned = host_output(i, j, k);
                    auto val_full = host_reference(i, j, k);
                    double error = Kokkos::abs(val_pruned - val_full);

                    if (error > 1e-10) {
                        int gi = i - nghost + f0_first;
                        int gj = j - nghost + f1_first;
                        int gk = k - nghost + f2_first;
                        std::cout << "  [" << gi << "," << gj << "," << gk << "]"
                                  << " pruned=(" << val_pruned.real() << "," << val_pruned.imag() << ")"
                                  << " full=(" << val_full.real() << "," << val_full.imag() << ")"
                                  << " error=" << std::scientific << error << std::endl;
                        ++printed;
                    }
                }
            }
        }
    }

    bool passed = (global_max_error < 1e-10);

    if (myRank == 0) {
        std::cout << "\n=== Pruned C2C Backward FFT Results ===" << std::endl;
        std::cout << "Compared " << global_count << " spatial points" << std::endl;
        std::cout << "Max error vs full IFFT: " << std::scientific << std::setprecision(6) << global_max_error << std::endl;
        std::cout << "Pruned C2C Backward FFT test: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    return passed;
}

// Test pruned R2C FFT (keeping original but adding debug output)
bool testPrunedRC() {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;

    std::array<int, dim> pt_real           = {64, 64, 64};
    std::array<int, dim> pt_complex_full   = {33, 64, 64};
    std::array<int, dim> pt_complex_pruned = {17, 32, 32};

    int myRank = ippl::Comm->rank();
    int nProcs = ippl::Comm->size();

    if (myRank == 0) {
        std::cout << "\n=== Testing Pruned R2C FFT ===" << std::endl;
        std::cout << "Running on " << nProcs << " MPI ranks" << std::endl;
        std::cout << "Real grid: " << pt_real[0] << "x" << pt_real[1] << "x" << pt_real[2] << std::endl;
        std::cout << "Full complex: " << pt_complex_full[0] << "x" << pt_complex_full[1] << "x" << pt_complex_full[2] << std::endl;
        std::cout << "Pruned complex: " << pt_complex_pruned[0] << "x" << pt_complex_pruned[1] << "x" << pt_complex_pruned[2] << std::endl;
    }

    // Create layouts
    ippl::Index I_real(pt_real[0]);
    ippl::Index J_real(pt_real[1]);
    ippl::Index K_real(pt_real[2]);
    ippl::NDIndex<dim> owned_real(I_real, J_real, K_real);

    ippl::Index I_cfull(pt_complex_full[0]);
    ippl::Index J_cfull(pt_complex_full[1]);
    ippl::Index K_cfull(pt_complex_full[2]);
    ippl::NDIndex<dim> owned_complex_full(I_cfull, J_cfull, K_cfull);

    ippl::Index I_cpruned(pt_complex_pruned[0]);
    ippl::Index J_cpruned(pt_complex_pruned[1]);
    ippl::Index K_cpruned(pt_complex_pruned[2]);
    ippl::NDIndex<dim> owned_complex_pruned(I_cpruned, J_cpruned, K_cpruned);

    std::array<bool, dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<dim> layout_real(MPI_COMM_WORLD, owned_real, isParallel);
    ippl::FieldLayout<dim> layout_complex_full(MPI_COMM_WORLD, owned_complex_full, isParallel);
    ippl::FieldLayout<dim> layout_complex_pruned(MPI_COMM_WORLD, owned_complex_pruned, isParallel);

    // Debug: print layout info
    const auto& lDom_real = layout_real.getLocalNDIndex();
    const auto& lDom_cfull = layout_complex_full.getLocalNDIndex();
    const auto& lDom_cpruned = layout_complex_pruned.getLocalNDIndex();

    std::cout << "Rank " << myRank << " layout_real: ["
              << lDom_real[0].first() << "-" << lDom_real[0].last() << ", "
              << lDom_real[1].first() << "-" << lDom_real[1].last() << ", "
              << lDom_real[2].first() << "-" << lDom_real[2].last() << "]" << std::endl;
    std::cout << "Rank " << myRank << " layout_complex_full: ["
              << lDom_cfull[0].first() << "-" << lDom_cfull[0].last() << ", "
              << lDom_cfull[1].first() << "-" << lDom_cfull[1].last() << ", "
              << lDom_cfull[2].first() << "-" << lDom_cfull[2].last() << "]" << std::endl;
    std::cout << "Rank " << myRank << " layout_complex_pruned: ["
              << lDom_cpruned[0].first() << "-" << lDom_cpruned[0].last() << ", "
              << lDom_cpruned[1].first() << "-" << lDom_cpruned[1].last() << ", "
              << lDom_cpruned[2].first() << "-" << lDom_cpruned[2].last() << "]" << std::endl;

    std::array<double, dim> dx = {
        1.0 / double(pt_real[0]),
        1.0 / double(pt_real[1]),
        1.0 / double(pt_real[2]),
    };
    ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};

    Mesh_t mesh_real(owned_real, hx, origin);
    Mesh_t mesh_complex_full(owned_complex_full, hx, origin);
    Mesh_t mesh_complex_pruned(owned_complex_pruned, hx, origin);

    typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type field_type_real;
    typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type field_type_complex;

    field_type_real field_real_input(mesh_real, layout_real);
    field_type_real field_real_copy(mesh_real, layout_real);
    field_type_complex field_complex_full(mesh_complex_full, layout_complex_full);
    field_type_complex field_complex_pruned(mesh_complex_pruned, layout_complex_pruned);

    // Setup pruning parameters
    ippl::PruningParams<dim> pruning;
    pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_complex_pruned[0]),
                                                static_cast<size_t>(pt_complex_pruned[1]),
                                                static_cast<size_t>(pt_complex_pruned[2])};

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", true);
    fftParams.add("r2c_direction", 0);

    typedef ippl::FFT<ippl::PrunedRCTransform, field_type_real> PrunedRCFFT_type;
    typedef ippl::FFT<ippl::RCTransform, field_type_real> RCFFT_type;

    if (myRank == 0) std::cout << "Creating pruned R2C FFT..." << std::endl;
    auto pruned_fft = std::make_unique<PrunedRCFFT_type>(layout_real, layout_complex_full,
                                                         layout_complex_pruned, pruning, fftParams);
    if (myRank == 0) std::cout << "Creating regular R2C FFT..." << std::endl;
    auto regular_fft = std::make_unique<RCFFT_type>(layout_real, layout_complex_full, fftParams);

    // Initialize with deterministic random data
    const int nghost = field_real_input.getNghost();
    auto& view_real = field_real_input.getView();
    typename field_type_real::HostMirror field_host = field_real_input.getHostMirror();

    for (size_t i = nghost; i < view_real.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view_real.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view_real.extent(2) - nghost; ++k) {
                int gi = i - nghost + lDom_real[0].first();
                int gj = j - nghost + lDom_real[1].first();
                int gk = k - nghost + lDom_real[2].first();

                std::mt19937_64 local_eng(gi * 100000 + gj * 1000 + gk + 42);
                std::uniform_real_distribution<double> unif(-1.0, 1.0);
                field_host(i, j, k) = unif(local_eng);
            }
        }
    }
    Kokkos::deep_copy(field_real_input.getView(), field_host);

    // Print input statistics
    if (myRank == 0) std::cout << "\nInput field statistics:" << std::endl;
    printFieldStats("field_real_input", field_real_input, myRank);

    // Save input and compute both FFTs
    field_real_copy = field_real_input;

    if (myRank == 0) std::cout << "\nExecuting pruned R2C FFT forward..." << std::endl;
    MPI_Barrier(ippl::Comm->getCommunicator());
    pruned_fft->transform(ippl::FORWARD, field_real_input, field_complex_pruned);
    MPI_Barrier(ippl::Comm->getCommunicator());

    if (myRank == 0) std::cout << "Executing regular R2C FFT forward..." << std::endl;
    MPI_Barrier(ippl::Comm->getCommunicator());
    regular_fft->transform(ippl::FORWARD, field_real_copy, field_complex_full);
    MPI_Barrier(ippl::Comm->getCommunicator());

    // Print output statistics
    if (myRank == 0) std::cout << "\nOutput field statistics:" << std::endl;
    printFieldStats("field_complex_pruned", field_complex_pruned, myRank);
    printFieldStats("field_complex_full", field_complex_full, myRank);

    // Compare
    auto view_pruned = field_complex_pruned.getView();
    auto view_full   = field_complex_full.getView();

    const auto& lDom_pruned = layout_complex_pruned.getLocalNDIndex();
    const auto& lDom_full   = layout_complex_full.getLocalNDIndex();
    const int nghost_pruned = field_complex_pruned.getNghost();
    const int nghost_full   = field_complex_full.getNghost();

    const int N1 = pt_real[1], K1 = pt_complex_pruned[1];
    const int N2 = pt_real[2], K2 = pt_complex_pruned[2];

    const int p0_first = lDom_pruned[0].first();
    const int p1_first = lDom_pruned[1].first();
    const int p2_first = lDom_pruned[2].first();

    const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
    const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
    const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();

    const int ng_p = nghost_pruned;
    const int ng_f = nghost_full;

    double max_error = 0.0;
    size_t count = 0;

    using exec_space = typename field_type_complex::execution_space;
    using mdrange_t  = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    Kokkos::parallel_reduce(
        "ComparePrunedWithFullRC",
        mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p, view_pruned.extent(2) - ng_p}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, double& local_max) {
            int gi_p = li_p - ng_p + p0_first;
            int gj_p = lj_p - ng_p + p1_first;
            int gk_p = lk_p - ng_p + p2_first;

            int gi_f = gi_p;
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f < f0_first || gi_f > f0_last || gj_f < f1_first || gj_f > f1_last
                || gk_f < f2_first || gk_f > f2_last) {
                return;
            }

            int li_f = gi_f - f0_first + ng_f;
            int lj_f = gj_f - f1_first + ng_f;
            int lk_f = gk_f - f2_first + ng_f;

            auto val_pruned = view_pruned(li_p, lj_p, lk_p);
            auto val_full   = view_full(li_f, lj_f, lk_f);

            double error = Kokkos::abs(val_pruned - val_full);
            if (error > local_max) {
                local_max = error;
            }
        },
        Kokkos::Max<double>(max_error));

    Kokkos::fence();

    Kokkos::parallel_reduce(
        "CountComparisonsRC",
        mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p, view_pruned.extent(2) - ng_p}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, size_t& local_count) {
            int gi_p = li_p - ng_p + p0_first;
            int gj_p = lj_p - ng_p + p1_first;
            int gk_p = lk_p - ng_p + p2_first;

            int gi_f = gi_p;
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f >= f0_first && gi_f <= f0_last && gj_f >= f1_first && gj_f <= f1_last
                && gk_f >= f2_first && gk_f <= f2_last) {
                ++local_count;
            }
        },
        count);

    Kokkos::fence();

    double global_max_error;
    size_t global_count;
    MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, ippl::Comm->getCommunicator());

    // Print worst mismatches if error is large
    if (max_error > 1e-10) {
        auto host_pruned = field_complex_pruned.getHostMirror();
        auto host_full = field_complex_full.getHostMirror();
        Kokkos::deep_copy(host_pruned, field_complex_pruned.getView());
        Kokkos::deep_copy(host_full, field_complex_full.getView());

        std::cout << "\nRank " << myRank << " worst R2C mismatches (error > 1e-10):" << std::endl;
        int printed = 0;
        int max_print = 10;

        for (size_t li_p = ng_p; li_p < host_pruned.extent(0) - ng_p && printed < max_print; ++li_p) {
            for (size_t lj_p = ng_p; lj_p < host_pruned.extent(1) - ng_p && printed < max_print; ++lj_p) {
                for (size_t lk_p = ng_p; lk_p < host_pruned.extent(2) - ng_p && printed < max_print; ++lk_p) {
                    int gi_p = li_p - ng_p + p0_first;
                    int gj_p = lj_p - ng_p + p1_first;
                    int gk_p = lk_p - ng_p + p2_first;

                    int gi_f = gi_p;
                    int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
                    int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

                    if (gi_f < f0_first || gi_f > f0_last || gj_f < f1_first || gj_f > f1_last
                        || gk_f < f2_first || gk_f > f2_last) {
                        continue;
                    }

                    int li_f = gi_f - f0_first + ng_f;
                    int lj_f = gj_f - f1_first + ng_f;
                    int lk_f = gk_f - f2_first + ng_f;

                    auto val_pruned = host_pruned(li_p, lj_p, lk_p);
                    auto val_full = host_full(li_f, lj_f, lk_f);
                    double error = Kokkos::abs(val_pruned - val_full);

                    if (error > 1e-10) {
                        std::cout << "  pruned[" << gi_p << "," << gj_p << "," << gk_p << "]="
                                  << "(" << val_pruned.real() << "," << val_pruned.imag() << ")"
                                  << " vs full[" << gi_f << "," << gj_f << "," << gk_f << "]="
                                  << "(" << val_full.real() << "," << val_full.imag() << ")"
                                  << " error=" << std::scientific << error << std::endl;
                        ++printed;
                    }
                }
            }
        }
    }

    bool passed = (global_max_error < 1e-10);

    if (myRank == 0) {
        std::cout << "\n=== Pruned R2C FFT Results ===" << std::endl;
        std::cout << "Compared " << global_count << " modes" << std::endl;
        std::cout << "Max error vs full R2C FFT: " << std::scientific << std::setprecision(6) << global_max_error << std::endl;
        std::cout << "Pruned R2C FFT test: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    return passed;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    int myRank = ippl::Comm->rank();

    if (myRank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "Pruned FFT Test Suite" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    bool cc_passed = testPrunedCC();
    MPI_Barrier(ippl::Comm->getCommunicator());

    bool cc_back_passed = testPrunedCCBackward();
    MPI_Barrier(ippl::Comm->getCommunicator());

    bool rc_passed = testPrunedRC();
    MPI_Barrier(ippl::Comm->getCommunicator());

    if (myRank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "=== Overall Results ===" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Pruned C2C Forward:  " << (cc_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Pruned C2C Backward: " << (cc_back_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Pruned R2C Forward:  " << (rc_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "All tests: " << ((cc_passed && rc_passed && cc_back_passed) ? "PASSED" : "FAILED") << std::endl;
    }

    ippl::finalize();
    return (cc_passed && rc_passed && cc_back_passed) ? 0 : 1;
}