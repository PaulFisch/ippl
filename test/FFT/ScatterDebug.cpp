#include "Ippl.h"

#include <Kokkos_Complex.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Interpolation/Kernels.h"
#include "Interpolation/Scatter/Scatter.h"
#include "Interpolation/Scatter/ScatterConfig.h"

template <typename T, typename PLayout>
class DebugBunch : public ippl::ParticleBase<PLayout> {
public:
    using Base = ippl::ParticleBase<PLayout>;

    ippl::ParticleAttrib<T> weight;

    explicit DebugBunch(PLayout& layout)
        : Base(layout) {
        this->addAttribute(weight);
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        constexpr unsigned Dim = 2;
        using T                = double;
        using ExecSpace        = Kokkos::DefaultExecutionSpace;

        using Mesh_t      = ippl::UniformCartesian<T, Dim>;
        using Centering_t = typename Mesh_t::DefaultCentering;
        using Field_t     = typename ippl::Field<T, Dim, Mesh_t, Centering_t, ExecSpace>::uniform_type;

        using PLayout_t = ippl::ParticleSpatialLayout<T, Dim, Mesh_t, ExecSpace>;
        using Bunch_t   = DebugBunch<T, PLayout_t>;

        int myRank = ippl::Comm->rank();
        int nRanks = ippl::Comm->size();

        // Small grid for visualization
        const int n_grid = 8;
        const T two_pi   = 2.0 * Kokkos::numbers::pi_v<T>;

        // Test with Linear kernel
        using kernel_type = ippl::Interpolation::LinearKernel<T>;
        kernel_type kernel;
        const int nghost = kernel.width() / 2 + 1;

        if (myRank == 0) {
            std::cout << "############################################################\n";
            std::cout << "#     PERIODIC BOUNDARY TILED SCATTER DEBUG               #\n";
            std::cout << "############################################################\n";
            std::cout << "Kernel: Linear (width=" << kernel.width() << ")\n";
            std::cout << "Grid: " << n_grid << "x" << n_grid << "\n";
            std::cout << "nghost: " << nghost << "\n";
            std::cout << "MPI ranks: " << nRanks << "\n";
            std::cout << "Domain: [0, " << two_pi << ") x [0, " << two_pi << ")\n";
            std::cout << "############################################################\n\n";
        }

        // Setup domain
        ippl::Vector<T, Dim> origin = {0.0, 0.0};
        ippl::Vector<T, Dim> hx     = {two_pi / n_grid, two_pi / n_grid};

        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        for (unsigned d = 0; d < Dim; ++d) {
            domains[d] = ippl::Index(n_grid);
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel, true, nghost);
        Mesh_t mesh(owned, hx, origin);

        const auto& lDom = layout.getLocalNDIndex();

        // Scatter configs
        ippl::Interpolation::ScatterConfig<Dim> cfg_atomic;
        cfg_atomic.method = ippl::Interpolation::ScatterMethod::Atomic;
        cfg_atomic.sort   = true;

        ippl::Interpolation::ScatterConfig<Dim> cfg_tiled;
        cfg_tiled.method = ippl::Interpolation::ScatterMethod::Tiled;
        cfg_tiled.sort   = true;

        // ================================================================
        // TEST 1: Multiple boundary particles
        // ================================================================
        if (myRank == 0) {
            std::cout << "=== TEST 1: Multiple boundary particles ===\n";
        }

        {
            PLayout_t playout(layout, mesh);
            Bunch_t bunch(playout);

            Field_t field_atomic(mesh, layout, nghost);
            Field_t field_tiled(mesh, layout, nghost);

            // Create particles near all boundaries
            size_t nParticlesPerBoundary = 3;
            size_t nBoundaries           = 2 * Dim;
            size_t totalParticles        = nBoundaries * nParticlesPerBoundary;

            if (myRank == 0) {
                bunch.create(totalParticles);

                auto R_view      = bunch.R.getView();
                auto weight_view = bunch.weight.getView();

                auto R_host      = Kokkos::create_mirror_view(R_view);
                auto weight_host = Kokkos::create_mirror_view(weight_view);

                size_t idx = 0;

                // Lower X boundary (x near 0)
                for (size_t i = 0; i < nParticlesPerBoundary; ++i) {
                    T x = 0.02 + i * 0.03;
                    T y = 0.5 * two_pi + (i - 1) * 0.2;
                    R_host(idx)      = {x, y};
                    weight_host(idx) = 1.0;
                    std::cout << "  Particle " << idx << " (lower X): (" << x << ", " << y << ")\n";
                    idx++;
                }

                // Upper X boundary (x near 2π)
                for (size_t i = 0; i < nParticlesPerBoundary; ++i) {
                    T x = two_pi - 0.02 - i * 0.03;
                    T y = 0.5 * two_pi + (i - 1) * 0.2;
                    R_host(idx)      = {x, y};
                    weight_host(idx) = 1.0;
                    std::cout << "  Particle " << idx << " (upper X): (" << x << ", " << y << ")\n";
                    idx++;
                }

                // Lower Y boundary (y near 0)
                for (size_t i = 0; i < nParticlesPerBoundary; ++i) {
                    T x = 0.5 * two_pi + (i - 1) * 0.2;
                    T y = 0.02 + i * 0.03;
                    R_host(idx)      = {x, y};
                    weight_host(idx) = 1.0;
                    std::cout << "  Particle " << idx << " (lower Y): (" << x << ", " << y << ")\n";
                    idx++;
                }

                // Upper Y boundary (y near 2π)
                for (size_t i = 0; i < nParticlesPerBoundary; ++i) {
                    T x = 0.5 * two_pi + (i - 1) * 0.2;
                    T y = two_pi - 0.02 - i * 0.03;
                    R_host(idx)      = {x, y};
                    weight_host(idx) = 1.0;
                    std::cout << "  Particle " << idx << " (upper Y): (" << x << ", " << y << ")\n";
                    idx++;
                }

                Kokkos::deep_copy(R_view, R_host);
                Kokkos::deep_copy(weight_view, weight_host);
            }

            bunch.update();
            Kokkos::fence();

            T totalWeight = bunch.weight.sum();

            if (myRank == 0) {
                std::cout << "\n  Total particles: " << totalParticles << "\n";
                std::cout << "  Total weight: " << totalWeight << "\n";
            }

            // Scatter with both methods
            field_atomic = T(0.0);
            field_tiled  = T(0.0);

            auto scatter_atomic = ippl::Scatter(kernel, cfg_atomic);
            auto scatter_tiled  = ippl::Scatter(kernel, cfg_tiled);

            scatter_atomic(field_atomic, bunch.R, bunch.weight);
            scatter_tiled(field_tiled, bunch.R, bunch.weight);
            Kokkos::fence();

            T sum_atomic = ippl::norm(field_atomic, 1);
            T sum_tiled  = ippl::norm(field_tiled, 1);

            if (myRank == 0) {
                std::cout << "  Field sum (Atomic): " << sum_atomic << "\n";
                std::cout << "  Field sum (Tiled): " << sum_tiled << "\n";
                std::cout << "  Expected: " << totalWeight << "\n";
                std::cout << "  Atomic error: " << std::abs(sum_atomic - totalWeight) << "\n";
                std::cout << "  Tiled error: " << std::abs(sum_tiled - totalWeight) << "\n\n";
            }

            // Print difference analysis
            if (myRank == 0 && nRanks == 1) {
                auto atomic_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_atomic.getView());
                auto tiled_host  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_tiled.getView());

                std::cout << "  Cells with differences:\n";
                int diffCount = 0;
                T maxDiff = 0;
                for (int j = -nghost; j < n_grid + nghost; ++j) {
                    for (int i = -nghost; i < n_grid + nghost; ++i) {
                        int li   = i + nghost;
                        int lj   = j + nghost;
                        T diff = atomic_host(li, lj) - tiled_host(li, lj);
                        if (std::abs(diff) > 1e-12) {
                            std::cout << "    grid(" << i << ", " << j << "): Atomic="
                                      << atomic_host(li, lj) << ", Tiled=" << tiled_host(li, lj)
                                      << ", Diff=" << diff << "\n";
                            diffCount++;
                            maxDiff = std::max(maxDiff, std::abs(diff));
                        }
                    }
                }
                if (diffCount == 0) {
                    std::cout << "    None - results match!\n";
                } else {
                    std::cout << "  Total differing cells: " << diffCount << ", max diff: " << maxDiff << "\n";
                }
            }
        }

        // ================================================================
        // TEST 2: Single corner particle near origin
        // ================================================================
        if (myRank == 0) {
            std::cout << "\n=== TEST 2: Single particle near origin (0.1, 0.1) ===\n";
        }

        {
            PLayout_t playout(layout, mesh);
            Bunch_t bunch(playout);

            Field_t field_atomic(mesh, layout, nghost);
            Field_t field_tiled(mesh, layout, nghost);

            if (myRank == 0) {
                bunch.create(1);

                auto R_view      = bunch.R.getView();
                auto weight_view = bunch.weight.getView();

                auto R_host      = Kokkos::create_mirror_view(R_view);
                auto weight_host = Kokkos::create_mirror_view(weight_view);

                T epsilon        = 0.1;
                R_host(0)        = {epsilon, epsilon};
                weight_host(0)   = 1.0;

                Kokkos::deep_copy(R_view, R_host);
                Kokkos::deep_copy(weight_view, weight_host);

                std::cout << "  Particle at (" << epsilon << ", " << epsilon << ")\n";
                std::cout << "  Grid coordinate: (" << epsilon / hx[0] << ", " << epsilon / hx[1] << ")\n";
            }

            bunch.update();
            Kokkos::fence();

            field_atomic = T(0.0);
            field_tiled  = T(0.0);

            auto scatter_atomic = ippl::Scatter(kernel, cfg_atomic);
            auto scatter_tiled  = ippl::Scatter(kernel, cfg_tiled);

            scatter_atomic(field_atomic, bunch.R, bunch.weight);
            scatter_tiled(field_tiled, bunch.R, bunch.weight);
            Kokkos::fence();

            T sum_atomic = ippl::norm(field_atomic, 1);
            T sum_tiled  = ippl::norm(field_tiled, 1);

            if (myRank == 0) {
                std::cout << "  Atomic sum: " << sum_atomic << "\n";
                std::cout << "  Tiled sum: " << sum_tiled << "\n";
                std::cout << "  Expected: 1.0\n";
            }

            if (myRank == 0 && nRanks == 1) {
                auto atomic_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_atomic.getView());
                auto tiled_host  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_tiled.getView());

                std::cout << "\n  Non-zero Atomic values:\n";
                for (int j = -nghost; j < n_grid + nghost; ++j) {
                    for (int i = -nghost; i < n_grid + nghost; ++i) {
                        T val = atomic_host(i + nghost, j + nghost);
                        if (std::abs(val) > 1e-12) {
                            std::cout << "    grid(" << i << ", " << j << ") = " << val << "\n";
                        }
                    }
                }

                std::cout << "\n  Non-zero Tiled values:\n";
                for (int j = -nghost; j < n_grid + nghost; ++j) {
                    for (int i = -nghost; i < n_grid + nghost; ++i) {
                        T val = tiled_host(i + nghost, j + nghost);
                        if (std::abs(val) > 1e-12) {
                            std::cout << "    grid(" << i << ", " << j << ") = " << val << "\n";
                        }
                    }
                }

                std::cout << "\n  Differences:\n";
                bool anyDiff = false;
                for (int j = -nghost; j < n_grid + nghost; ++j) {
                    for (int i = -nghost; i < n_grid + nghost; ++i) {
                        T diff = atomic_host(i + nghost, j + nghost) - tiled_host(i + nghost, j + nghost);
                        if (std::abs(diff) > 1e-12) {
                            std::cout << "    grid(" << i << ", " << j << "): Atomic="
                                      << atomic_host(i + nghost, j + nghost) << ", Tiled="
                                      << tiled_host(i + nghost, j + nghost) << ", Diff=" << diff << "\n";
                            anyDiff = true;
                        }
                    }
                }
                if (!anyDiff) {
                    std::cout << "    None - results match!\n";
                }
            }
        }

        // ================================================================
        // TEST 3: Particle exactly at x=0 boundary
        // ================================================================
        if (myRank == 0) {
            std::cout << "\n=== TEST 3: Particle at x≈0 (should wrap to x=n-1) ===\n";
        }

        {
            PLayout_t playout(layout, mesh);
            Bunch_t bunch(playout);

            Field_t field_atomic(mesh, layout, nghost);
            Field_t field_tiled(mesh, layout, nghost);

            if (myRank == 0) {
                bunch.create(1);

                auto R_view      = bunch.R.getView();
                auto weight_view = bunch.weight.getView();

                auto R_host      = Kokkos::create_mirror_view(R_view);
                auto weight_host = Kokkos::create_mirror_view(weight_view);

                T x = 1e-10;  // Effectively 0
                T y = Kokkos::numbers::pi_v<T>;  // Center in y
                R_host(0)        = {x, y};
                weight_host(0)   = 1.0;

                Kokkos::deep_copy(R_view, R_host);
                Kokkos::deep_copy(weight_view, weight_host);

                std::cout << "  Particle at (" << x << ", " << y << ")\n";
                std::cout << "  Grid coordinate: (" << x / hx[0] << ", " << y / hx[1] << ")\n";
            }

            bunch.update();
            Kokkos::fence();

            field_atomic = T(0.0);
            field_tiled  = T(0.0);

            auto scatter_atomic = ippl::Scatter(kernel, cfg_atomic);
            auto scatter_tiled  = ippl::Scatter(kernel, cfg_tiled);

            scatter_atomic(field_atomic, bunch.R, bunch.weight);
            scatter_tiled(field_tiled, bunch.R, bunch.weight);
            Kokkos::fence();

            T sum_atomic = ippl::norm(field_atomic, 1);
            T sum_tiled  = ippl::norm(field_tiled, 1);

            if (myRank == 0) {
                std::cout << "  Atomic sum: " << sum_atomic << "\n";
                std::cout << "  Tiled sum: " << sum_tiled << "\n";
                std::cout << "  Expected: 1.0\n";
                std::cout << "  Atomic error: " << std::abs(sum_atomic - 1.0) << "\n";
                std::cout << "  Tiled error: " << std::abs(sum_tiled - 1.0) << "\n";
            }

            if (myRank == 0 && nRanks == 1) {
                auto atomic_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_atomic.getView());
                auto tiled_host  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_tiled.getView());

                std::cout << "\n  Non-zero Atomic values:\n";
                for (int j = -nghost; j < n_grid + nghost; ++j) {
                    for (int i = -nghost; i < n_grid + nghost; ++i) {
                        T val = atomic_host(i + nghost, j + nghost);
                        if (std::abs(val) > 1e-12) {
                            std::cout << "    grid(" << i << ", " << j << ") = " << val << "\n";
                        }
                    }
                }

                std::cout << "\n  Non-zero Tiled values:\n";
                for (int j = -nghost; j < n_grid + nghost; ++j) {
                    for (int i = -nghost; i < n_grid + nghost; ++i) {
                        T val = tiled_host(i + nghost, j + nghost);
                        if (std::abs(val) > 1e-12) {
                            std::cout << "    grid(" << i << ", " << j << ") = " << val << "\n";
                        }
                    }
                }

                std::cout << "\n  Critical analysis - periodic wrapping:\n";
                std::cout << "    For particle at x≈0, stencil base = -1 or 0\n";
                std::cout << "    x=-1 should wrap to x=" << n_grid - 1 << " (periodic)\n";

                // Check specific cells
                std::cout << "\n  Key cell values:\n";
                std::cout << "    Atomic grid(-1, 4) [ghost]: " << atomic_host(nghost - 1, 4 + nghost) << "\n";
                std::cout << "    Tiled grid(-1, 4) [ghost]: " << tiled_host(nghost - 1, 4 + nghost) << "\n";
                std::cout << "    Atomic grid(0, 4): " << atomic_host(nghost, 4 + nghost) << "\n";
                std::cout << "    Tiled grid(0, 4): " << tiled_host(nghost, 4 + nghost) << "\n";
                std::cout << "    Atomic grid(" << n_grid - 1 << ", 4) [wrap target]: "
                          << atomic_host(n_grid - 1 + nghost, 4 + nghost) << "\n";
                std::cout << "    Tiled grid(" << n_grid - 1 << ", 4) [wrap target]: "
                          << tiled_host(n_grid - 1 + nghost, 4 + nghost) << "\n";
            }
        }

        // ================================================================
        // TEST 4: Particle at upper boundary x≈2π
        // ================================================================
        if (myRank == 0) {
            std::cout << "\n=== TEST 4: Particle at x≈2π (should wrap to x=0) ===\n";
        }

        {
            PLayout_t playout(layout, mesh);
            Bunch_t bunch(playout);

            Field_t field_atomic(mesh, layout, nghost);
            Field_t field_tiled(mesh, layout, nghost);

            if (myRank == 0) {
                bunch.create(1);

                auto R_view      = bunch.R.getView();
                auto weight_view = bunch.weight.getView();

                auto R_host      = Kokkos::create_mirror_view(R_view);
                auto weight_host = Kokkos::create_mirror_view(weight_view);

                T x = two_pi - 1e-10;  // Just before upper boundary
                T y = Kokkos::numbers::pi_v<T>;  // Center in y
                R_host(0)        = {x, y};
                weight_host(0)   = 1.0;

                Kokkos::deep_copy(R_view, R_host);
                Kokkos::deep_copy(weight_view, weight_host);

                std::cout << "  Particle at (" << x << ", " << y << ")\n";
                std::cout << "  Grid coordinate: (" << x / hx[0] << ", " << y / hx[1] << ")\n";
            }

            bunch.update();
            Kokkos::fence();

            field_atomic = T(0.0);
            field_tiled  = T(0.0);

            auto scatter_atomic = ippl::Scatter(kernel, cfg_atomic);
            auto scatter_tiled  = ippl::Scatter(kernel, cfg_tiled);

            scatter_atomic(field_atomic, bunch.R, bunch.weight);
            scatter_tiled(field_tiled, bunch.R, bunch.weight);
            Kokkos::fence();

            T sum_atomic = ippl::norm(field_atomic, 1);
            T sum_tiled  = ippl::norm(field_tiled, 1);

            if (myRank == 0) {
                std::cout << "  Atomic sum: " << sum_atomic << "\n";
                std::cout << "  Tiled sum: " << sum_tiled << "\n";
                std::cout << "  Expected: 1.0\n";
                std::cout << "  Atomic error: " << std::abs(sum_atomic - 1.0) << "\n";
                std::cout << "  Tiled error: " << std::abs(sum_tiled - 1.0) << "\n";
            }

            if (myRank == 0 && nRanks == 1) {
                auto atomic_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_atomic.getView());
                auto tiled_host  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_tiled.getView());

                std::cout << "\n  Non-zero Atomic values:\n";
                for (int j = -nghost; j < n_grid + nghost; ++j) {
                    for (int i = -nghost; i < n_grid + nghost; ++i) {
                        T val = atomic_host(i + nghost, j + nghost);
                        if (std::abs(val) > 1e-12) {
                            std::cout << "    grid(" << i << ", " << j << ") = " << val << "\n";
                        }
                    }
                }

                std::cout << "\n  Non-zero Tiled values:\n";
                for (int j = -nghost; j < n_grid + nghost; ++j) {
                    for (int i = -nghost; i < n_grid + nghost; ++i) {
                        T val = tiled_host(i + nghost, j + nghost);
                        if (std::abs(val) > 1e-12) {
                            std::cout << "    grid(" << i << ", " << j << ") = " << val << "\n";
                        }
                    }
                }

                std::cout << "\n  Key cell values:\n";
                std::cout << "    Atomic grid(" << n_grid - 1 << ", 4): "
                          << atomic_host(n_grid - 1 + nghost, 4 + nghost) << "\n";
                std::cout << "    Tiled grid(" << n_grid - 1 << ", 4): "
                          << tiled_host(n_grid - 1 + nghost, 4 + nghost) << "\n";
                std::cout << "    Atomic grid(" << n_grid << ", 4) [ghost]: "
                          << atomic_host(n_grid + nghost, 4 + nghost) << "\n";
                std::cout << "    Tiled grid(" << n_grid << ", 4) [ghost]: "
                          << tiled_host(n_grid + nghost, 4 + nghost) << "\n";
                std::cout << "    Atomic grid(0, 4) [wrap target]: "
                          << atomic_host(nghost, 4 + nghost) << "\n";
                std::cout << "    Tiled grid(0, 4) [wrap target]: "
                          << tiled_host(nghost, 4 + nghost) << "\n";
            }
        }

        // ================================================================
        // TEST 5: Interior particle (should work for both)
        // ================================================================
        if (myRank == 0) {
            std::cout << "\n=== TEST 5: Interior particle (sanity check) ===\n";
        }

        {
            PLayout_t playout(layout, mesh);
            Bunch_t bunch(playout);

            Field_t field_atomic(mesh, layout, nghost);
            Field_t field_tiled(mesh, layout, nghost);

            if (myRank == 0) {
                bunch.create(1);

                auto R_view      = bunch.R.getView();
                auto weight_view = bunch.weight.getView();

                auto R_host      = Kokkos::create_mirror_view(R_view);
                auto weight_host = Kokkos::create_mirror_view(weight_view);

                // Center of domain - no boundary issues
                T x = Kokkos::numbers::pi_v<T>;
                T y = Kokkos::numbers::pi_v<T>;
                R_host(0)        = {x, y};
                weight_host(0)   = 1.0;

                Kokkos::deep_copy(R_view, R_host);
                Kokkos::deep_copy(weight_view, weight_host);

                std::cout << "  Particle at (" << x << ", " << y << ")\n";
                std::cout << "  Grid coordinate: (" << x / hx[0] << ", " << y / hx[1] << ")\n";
            }

            bunch.update();
            Kokkos::fence();

            field_atomic = T(0.0);
            field_tiled  = T(0.0);

            auto scatter_atomic = ippl::Scatter(kernel, cfg_atomic);
            auto scatter_tiled  = ippl::Scatter(kernel, cfg_tiled);

            scatter_atomic(field_atomic, bunch.R, bunch.weight);
            scatter_tiled(field_tiled, bunch.R, bunch.weight);
            Kokkos::fence();

            T sum_atomic = ippl::norm(field_atomic, 1);
            T sum_tiled  = ippl::norm(field_tiled, 1);

            if (myRank == 0) {
                std::cout << "  Atomic sum: " << sum_atomic << "\n";
                std::cout << "  Tiled sum: " << sum_tiled << "\n";
                std::cout << "  Expected: 1.0\n";

                T diff = std::abs(sum_atomic - sum_tiled);
                if (diff < 1e-12) {
                    std::cout << "  ✓ Results match for interior particle\n";
                } else {
                    std::cout << "  ✗ Results differ by: " << diff << "\n";
                }
            }
        }

        std::cout << "\n############################################################\n";
    }

    ippl::finalize();
    return 0;
}