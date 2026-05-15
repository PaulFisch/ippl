// TestMinimal: progressively strip the PCG/newton hang reproducer until it
// stops hanging. Run from CLI with a "level" arg (1..10) that selects which
// stripped variant to run. Higher level = simpler.
//
//   Level 1: full PoissonCG with newton/4    -- known to HANG
//   Level 2: manual PCG-newton recursion     -- minimal manual replica
//   Level 3: manual without PeriodicFace BC on workspace fields
//   Level 4: only the inner laplace loop, no outer PCG
//   Level 5: only laplace on a single Field, no recursion
//   Level 6: only accumulateHalo + fillHalo in tight loop
//   Level 7: only accumulateHalo in tight loop, nothing else
//   ...
//
// Usage: mpiexec -n 4 ./TestMinimal <level> [nIters]

#include "Ippl.h"
#include <Kokkos_MathematicalConstants.hpp>
#include <iostream>
#include "Utility/Inform.h"
#include "PoissonSolvers/PoissonCG.h"

constexpr unsigned int Dim = 3;
using Mesh_t      = ippl::UniformCartesian<double, 3>;
using Centering_t = Mesh_t::DefaultCentering;
using FT          = ippl::Field<double, Dim, Mesh_t, Centering_t>;

static int g_rank = 0;
static void tick(const char* what, int it) {
    std::fprintf(stderr, "[r%d %d] %s\n", g_rank, it, what);
    std::fflush(stderr);
}

// Manual newton-style recursion, returns Field by value, calls -laplace.
static FT manual_newton(FT& u, int lvl, Mesh_t& mesh,
                        ippl::FieldLayout<Dim>& layout) {
    FT res(mesh, layout);
    if (lvl == 0) {
        res = 0.5 * u;
        return res;
    }
    FT PAPr(mesh, layout);
    FT Pr(mesh, layout);
    Pr   = manual_newton(u, lvl - 1, mesh, layout);
    PAPr = -laplace(Pr);
    PAPr = manual_newton(PAPr, lvl - 1, mesh, layout);
    res  = Pr - PAPr;
    return res;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        int level   = (argc > 1) ? std::atoi(argv[1]) : 1;
        int nIters  = (argc > 2) ? std::atoi(argv[2]) : 3;
        int pt      = (argc > 3) ? std::atoi(argv[3]) : 16;
        int maxIter = (argc > 4) ? std::atoi(argv[4]) : 500;

        Inform info("TestMinimal");
        g_rank = ippl::Comm->rank();

        ippl::Index I(pt);
        ippl::NDIndex<Dim> owned(I, I, I);
        std::array<bool, Dim> isParallel;
        for (unsigned d = 0; d < Dim; ++d) isParallel[d] = true;
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel, true);
        double dx = 2.0 / double(pt);
        ippl::Vector<double, Dim> hx     = {dx, dx, dx};
        ippl::Vector<double, Dim> origin = -1;
        Mesh_t mesh(owned, hx, origin);

        FT rhs(mesh, layout);
        FT lhs(mesh, layout);
        ippl::BConds<FT, Dim> bcField;
        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<FT>>(i);
        }
        lhs.setFieldBC(bcField);

        info << "level=" << level << " nIters=" << nIters << " pt=" << pt
             << " maxIter=" << maxIter << endl;

        if (level == 1) {
            // PoissonCG newton/4  --  known HANGS
            ippl::PoissonCG<FT> solver;
            ippl::ParameterList params;
            params.add("max_iterations", maxIter);
            params.add("tolerance", 1e-4);
            params.add("solver", "preconditioned");
            params.add("output_type", ippl::PoissonCG<FT>::SOL);
            params.add("preconditioner_type", std::string("newton"));
            params.add("gauss_seidel_inner_iterations", 5);
            params.add("gauss_seidel_outer_iterations", 1);
            params.add("newton_level", 4);
            params.add("chebyshev_degree", 31);
            params.add("richardson_iterations", 1);
            params.add("communication", 1);
            params.add("ssor_omega", 1.57079632679);
            solver.mergeParameters(params);
            solver.setRhs(rhs);
            solver.setLhs(lhs);
            for (int it = 0; it < nIters; ++it) {
                tick("level1 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                lhs = 0;
                solver.solve();
                tick("level1 done", it);
            }
        }
        else if (level == 2) {
            // Manual newton/4 with PCG-like outer loop. No PCG class.
            FT r(mesh, layout), d(mesh, layout), q(mesh, layout), s(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level2 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                lhs = 0;
                r = rhs - (-laplace(lhs));
                d = manual_newton(r, 4, mesh, layout).deepCopy();
                d.setFieldBC(bcField);
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    double inner_dq = innerProduct(d, q);
                    if (inner_dq == 0.0) inner_dq = 1.0;
                    double alpha = innerProduct(r, d) / inner_dq;
                    lhs = lhs + alpha * d;
                    r = r - alpha * q;
                    s = manual_newton(r, 4, mesh, layout).deepCopy();
                    double delta1 = innerProduct(r, s);
                    d = s + 0.5 * d;
                    (void)delta1;
                }
                tick("level2 done", it);
            }
        }
        else if (level == 3) {
            // Same as 2 but NO setFieldBC on d (no PeriodicFace anywhere).
            // Workspace fields keep default NoBcFace.
            FT r(mesh, layout), d(mesh, layout), q(mesh, layout), s(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level3 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                lhs = 0;
                r = rhs - (-laplace(lhs));
                d = manual_newton(r, 4, mesh, layout).deepCopy();
                // d.setFieldBC(bcField);                            // <-- removed
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    double inner_dq = innerProduct(d, q);
                    if (inner_dq == 0.0) inner_dq = 1.0;
                    double alpha = innerProduct(r, d) / inner_dq;
                    lhs = lhs + alpha * d;
                    r = r - alpha * q;
                    s = manual_newton(r, 4, mesh, layout).deepCopy();
                    d = s + 0.5 * d;
                }
                tick("level3 done", it);
            }
        }
        else if (level == 4) {
            // Strip outer PCG loop. Only the inner laplace loop on a
            // persistent field d (with PeriodicFace BC).
            FT d(mesh, layout), q(mesh, layout);
            d.setFieldBC(bcField);
            for (int it = 0; it < nIters; ++it) {
                tick("level4 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                d = rhs;
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    d = q + 0.5 * d;
                }
                tick("level4 done", it);
            }
        }
        else if (level == 5) {
            // Even simpler: accumulateHalo + many laplace on a single Field.
            FT d(mesh, layout);
            d.setFieldBC(bcField);
            for (int it = 0; it < nIters; ++it) {
                tick("level5 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                d = rhs;
                for (int k = 0; k < maxIter; ++k) {
                    d = -laplace(d);
                    if (k % 1000 == 0 && g_rank == 0) {
                        std::fprintf(stderr, "[r%d %d] level5 k=%d\n", g_rank, it, k);
                        std::fflush(stderr);
                    }
                }
                tick("level5 done", it);
            }
        }
        else if (level == 6) {
            // accumulateHalo on rhs then many fillHalo on lhs (no laplace).
            for (int it = 0; it < nIters; ++it) {
                tick("level6 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                for (int k = 0; k < maxIter; ++k) {
                    lhs.fillHalo();
                }
                tick("level6 done", it);
            }
        }
        else if (level == 7) {
            // Only accumulateHalo in tight loop.
            for (int it = 0; it < nIters; ++it) {
                tick("level7 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    rhs.accumulateHalo();
                }
                tick("level7 done", it);
            }
        }
        else if (level == 8) {
            // Like level 5 but with FRESH fields per iteration of the
            // inner loop (mirrors newton's `Field Pr(mesh,layout);` churn).
            rhs.accumulateHalo();
            for (int it = 0; it < nIters; ++it) {
                tick("level8 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT fresh(mesh, layout);
                    fresh = -laplace(rhs);
                }
                tick("level8 done", it);
            }
        }
        else if (level == 9) {
            // Like level 8 but the fresh field calls accumulateHalo
            // instead of laplace.
            for (int it = 0; it < nIters; ++it) {
                tick("level9 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT fresh(mesh, layout);
                    fresh.accumulateHalo();
                }
                tick("level9 done", it);
            }
        }
        else if (level == 12) {
            // Like level 11 but NO rhs.accumulateHalo()
            FT r(mesh, layout), d(mesh, layout), q(mesh, layout), s(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level12 start", it);
                rhs = 1.0;
                lhs = 0;
                r = rhs - (-laplace(lhs));
                d = manual_newton(r, 4, mesh, layout).deepCopy();
                d.setFieldBC(bcField);
                double delta1 = innerProduct(r, d);
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    double inner_dq = innerProduct(d, q);
                    if (inner_dq == 0.0) inner_dq = 1.0;
                    double alpha = delta1 / inner_dq;
                    lhs = lhs + alpha * d;
                    r = r - alpha * q;
                    s = manual_newton(r, 4, mesh, layout).deepCopy();
                    double delta0 = delta1;
                    delta1 = innerProduct(r, s);
                    double beta = delta1 / delta0;
                    d = s + beta * d;
                }
                tick("level12 done", it);
            }
        }
        else if (level == 13) {
            // Like level 11 but newton level=1 (much less halo churn).
            FT r(mesh, layout), d(mesh, layout), q(mesh, layout), s(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level13 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                lhs = 0;
                r = rhs - (-laplace(lhs));
                d = manual_newton(r, 1, mesh, layout).deepCopy();
                d.setFieldBC(bcField);
                double delta1 = innerProduct(r, d);
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    double inner_dq = innerProduct(d, q);
                    if (inner_dq == 0.0) inner_dq = 1.0;
                    double alpha = delta1 / inner_dq;
                    lhs = lhs + alpha * d;
                    r = r - alpha * q;
                    s = manual_newton(r, 1, mesh, layout).deepCopy();
                    double delta0 = delta1;
                    delta1 = innerProduct(r, s);
                    double beta = delta1 / delta0;
                    d = s + beta * d;
                }
                tick("level13 done", it);
            }
        }
        else if (level == 14) {
            // Like level 11 but no setFieldBC (no PeriodicFace on d).
            FT r(mesh, layout), d(mesh, layout), q(mesh, layout), s(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level14 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                lhs = 0;
                r = rhs - (-laplace(lhs));
                d = manual_newton(r, 4, mesh, layout).deepCopy();
                // d.setFieldBC(bcField);                  // removed
                double delta1 = innerProduct(r, d);
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    double inner_dq = innerProduct(d, q);
                    if (inner_dq == 0.0) inner_dq = 1.0;
                    double alpha = delta1 / inner_dq;
                    lhs = lhs + alpha * d;
                    r = r - alpha * q;
                    s = manual_newton(r, 4, mesh, layout).deepCopy();
                    double delta0 = delta1;
                    delta1 = innerProduct(r, s);
                    double beta = delta1 / delta0;
                    d = s + beta * d;
                }
                tick("level14 done", it);
            }
        }
        else if (level == 11) {
            // Replicate EXACTLY PoissonCG newton-level-4 inline (no PCG class).
            // The persistent workspace, the field-by-value newton recursion,
            // the deepCopy, and the setFieldBC are all here.
            FT r(mesh, layout), d(mesh, layout), q(mesh, layout), s(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level11 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                lhs = 0;
                r = rhs - (-laplace(lhs));
                d = manual_newton(r, 4, mesh, layout).deepCopy();
                d.setFieldBC(bcField);
                double delta1 = innerProduct(r, d);
                (void)delta1;
                auto t_start = std::chrono::steady_clock::now();
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    double inner_dq = innerProduct(d, q);
                    if (inner_dq == 0.0) inner_dq = 1.0;
                    double alpha = delta1 / inner_dq;
                    lhs = lhs + alpha * d;
                    r = r - alpha * q;
                    s = manual_newton(r, 4, mesh, layout).deepCopy();
                    double delta0 = delta1;
                    delta1 = innerProduct(r, s);
                    double beta = delta1 / delta0;
                    d = s + beta * d;
                    if ((k + 1) % 100 == 0 && g_rank == 0) {
                        auto t_now = std::chrono::steady_clock::now();
                        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     t_now - t_start).count();
                        std::fprintf(stderr, "[r0 %d] level11 k=%d (%lld ms)\n",
                                     it, k + 1, (long long)ms);
                        std::fflush(stderr);
                    }
                }
                tick("level11 done", it);
            }
        }
        else if (level == 10) {
            // Like level 8 but ALSO declare ENTIRE workspace inside loop
            // body, like the newton recursion does at every level.
            for (int it = 0; it < nIters; ++it) {
                tick("level10 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT a(mesh, layout);
                    FT b(mesh, layout);
                    FT c(mesh, layout);
                    a = rhs;
                    b = -laplace(a);
                    c = -laplace(b);
                }
                tick("level10 done", it);
            }
        }
        else if (level == 15) {
            // No PCG, no innerProduct, no halo exchange. Just call
            // manual_newton(level=4) in a maxIter loop. Tests whether the
            // pure recursion volume is sufficient to trigger the hang.
            rhs = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level15 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT s = manual_newton(rhs, 4, mesh, layout);
                    (void)s;
                }
                tick("level15 done", it);
            }
        }
        else if (level == 16) {
            // Like 15 but newton level=3.
            rhs = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level16 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT s = manual_newton(rhs, 3, mesh, layout);
                    (void)s;
                }
                tick("level16 done", it);
            }
        }
        else if (level == 17) {
            // Like 15 but newton level=2.
            rhs = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level17 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT s = manual_newton(rhs, 2, mesh, layout);
                    (void)s;
                }
                tick("level17 done", it);
            }
        }
        else if (level == 18) {
            // Like 11 but newton level=3 (not 4). Tests depth boundary.
            FT r(mesh, layout), d(mesh, layout), q(mesh, layout), s(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level18 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                lhs = 0;
                r = rhs - (-laplace(lhs));
                d = manual_newton(r, 3, mesh, layout).deepCopy();
                d.setFieldBC(bcField);
                double delta1 = innerProduct(r, d);
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    double inner_dq = innerProduct(d, q);
                    if (inner_dq == 0.0) inner_dq = 1.0;
                    double alpha = delta1 / inner_dq;
                    lhs = lhs + alpha * d;
                    r = r - alpha * q;
                    s = manual_newton(r, 3, mesh, layout).deepCopy();
                    double delta0 = delta1;
                    delta1 = innerProduct(r, s);
                    double beta = delta1 / delta0;
                    d = s + beta * d;
                }
                tick("level18 done", it);
            }
        }
        else if (level == 20) {
            // Like 15 but newton lvl=0 (no laplace, just `res = 0.5 * u`).
            // Tests if the bare return-by-value pattern + fresh-field
            // allocation is enough.
            rhs = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level20 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT s = manual_newton(rhs, 0, mesh, layout);
                    (void)s;
                }
                tick("level20 done", it);
            }
        }
        else if (level == 21) {
            // Like 15 but newton lvl=1 (one laplace, one return).
            rhs = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level21 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT s = manual_newton(rhs, 1, mesh, layout);
                    (void)s;
                }
                tick("level21 done", it);
            }
        }
        else if (level == 22) {
            // Mimic manual_newton(lvl=4) UNROLLED. No recursion, no
            // return-by-value. Same set of allocs and laplaces.
            rhs = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level22 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    // 16 leaf operations, each touches a fresh Field.
                    for (int i = 0; i < 16; ++i) {
                        FT a(mesh, layout);
                        FT b(mesh, layout);
                        a = 0.5 * rhs;
                        b = -laplace(a);
                    }
                }
                tick("level22 done", it);
            }
        }
        else if (level == 23) {
            // Like 22 but only 1 fresh-field+laplace per inner step.
            // Tests if total work or per-iter density is what matters.
            rhs = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level23 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    FT a(mesh, layout);
                    FT b(mesh, layout);
                    a = 0.5 * rhs;
                    b = -laplace(a);
                }
                tick("level23 done", it);
            }
        }
        else if (level == 24) {
            // Like 22 but without `a = 0.5 * rhs`. Just fresh a + laplace(a).
            // a's data is uninitialized but that doesn't matter — we just
            // want to know if the assignment step contributes.
            rhs = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level24 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    for (int i = 0; i < 16; ++i) {
                        FT a(mesh, layout);
                        FT b(mesh, layout);
                        b = -laplace(a);
                    }
                }
                tick("level24 done", it);
            }
        }
        else if (level == 25) {
            // Like 22 but use PERSISTENT a and b. Tests if fresh-field
            // creation in the inner-inner loop is the trigger.
            rhs = 1.0;
            FT a(mesh, layout);
            FT b(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level25 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    for (int i = 0; i < 16; ++i) {
                        a = 0.5 * rhs;
                        b = -laplace(a);
                    }
                }
                tick("level25 done", it);
            }
        }
        else if (level == 26) {
            // Level 5 with 16 laplaces in immediate succession per outer
            // iter. Only ONE persistent field d. If this hangs, it's
            // density of fillHalo per outer iter that matters; if it
            // passes, it's the alternation between two fields a, b.
            FT d(mesh, layout);
            d.setFieldBC(bcField);
            d = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level26 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    for (int i = 0; i < 16; ++i) {
                        d = -laplace(d);
                    }
                }
                tick("level26 done", it);
            }
        }
        else if (level == 27) {
            // Level 26 but no setFieldBC (so it's exactly like level 5
            // but with 16 laplaces per outer iter). Removes the BC apply.
            FT d(mesh, layout);
            d = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level27 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    for (int i = 0; i < 16; ++i) {
                        d = -laplace(d);
                    }
                }
                tick("level27 done", it);
            }
        }
        else if (level == 28) {
            // Like 25 but no `a = 0.5*rhs` — just 16 laplaces between
            // two persistent fields, alternating a → b → a.
            FT a(mesh, layout);
            FT b(mesh, layout);
            a = 1.0;
            for (int it = 0; it < nIters; ++it) {
                tick("level28 start", it);
                for (int k = 0; k < maxIter; ++k) {
                    // alternate so both fields' halos get exchanged
                    for (int i = 0; i < 8; ++i) {
                        b = -laplace(a);
                        a = -laplace(b);
                    }
                }
                tick("level28 done", it);
            }
        }
        else if (level == 19) {
            // Like 11 but newton level=2.
            FT r(mesh, layout), d(mesh, layout), q(mesh, layout), s(mesh, layout);
            for (int it = 0; it < nIters; ++it) {
                tick("level19 start", it);
                rhs = 1.0;
                rhs.accumulateHalo();
                lhs = 0;
                r = rhs - (-laplace(lhs));
                d = manual_newton(r, 2, mesh, layout).deepCopy();
                d.setFieldBC(bcField);
                double delta1 = innerProduct(r, d);
                for (int k = 0; k < maxIter; ++k) {
                    q = -laplace(d);
                    double inner_dq = innerProduct(d, q);
                    if (inner_dq == 0.0) inner_dq = 1.0;
                    double alpha = delta1 / inner_dq;
                    lhs = lhs + alpha * d;
                    r = r - alpha * q;
                    s = manual_newton(r, 2, mesh, layout).deepCopy();
                    double delta0 = delta1;
                    delta1 = innerProduct(r, s);
                    double beta = delta1 / delta0;
                    d = s + beta * d;
                }
                tick("level19 done", it);
            }
        }
        else {
            info << "unknown level " << level << endl;
        }

        info << "level " << level << " ALL DONE" << endl;
    }
    ippl::finalize();
    return 0;
}
