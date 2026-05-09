//
// Class PCG
//   Preconditioned Conjugate Gradient solver algorithm
//

#ifndef IPPL_PCG_H
#define IPPL_PCG_H

#include "Preconditioner.h"
#include "SolverAlgorithm.h"
#include "FEM/FEMVector.h"

#ifdef IPPL_HALO_DEBUG
#include <cstdio>
#include <sstream>
#include <mpi.h>
#define IPPL_PCG_LOG(stream)                                                               \
    do {                                                                                   \
        std::ostringstream _ipplpcglog_oss;                                                \
        _ipplpcglog_oss << "[PCG/r=" << ippl::Comm->rank() << "]" << stream << '\n';       \
        const auto _ipplpcglog_s = _ipplpcglog_oss.str();                                  \
        std::fprintf(stderr, "%s", _ipplpcglog_s.c_str());                                 \
        std::fflush(stderr);                                                               \
    } while (0)
// Phase-boundary barrier inside the CG loop. Same semantics as IPPL_HALO_SYNC
// but tagged for the PCG trace. If a rank stops emitting "post" lines, the
// last "pre id=…" line tells us exactly which CG phase it got stuck on.
#define IPPL_PCG_SYNC(label)                                                               \
    do {                                                                                   \
        IPPL_PCG_LOG("sync:pre id=" << label);                                             \
        MPI_Barrier(ippl::Comm->getCommunicator());                                        \
        IPPL_PCG_LOG("sync:post id=" << label);                                            \
    } while (0)
#else
#define IPPL_PCG_LOG(stream) do {} while (0)
#define IPPL_PCG_SYNC(label) do {} while (0)
#endif

namespace ippl {
    template <typename OperatorRet, typename LowerRet, typename UpperRet, typename UpperLowerRet,
              typename InverseDiagRet, typename DiagRet, typename FieldLHS,
              typename FieldRHS = FieldLHS>
    class CG : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorF    = std::function<OperatorRet(lhs_type)>;
        using LowerF       = std::function<LowerRet(lhs_type)>;
        using UpperF       = std::function<UpperRet(lhs_type)>;
        using UpperLowerF  = std::function<UpperLowerRet(lhs_type)>;
        using InverseDiagF = std::function<InverseDiagRet(lhs_type)>;
        using DiagF        = std::function<DiagRet(lhs_type)>;
        using mesh_type    = typename lhs_type::Mesh_t;
        using layout_type  = typename lhs_type::Layout_t;

        virtual ~CG() = default;

        /*
         * Initializes the fields needed for CG operations
         * and avoids allocating them at each solve step.
         * @param mesh The mesh to initialize the field with
         * @param layout The layout to initialize the field with
         */
        virtual void initializeFields(mesh_type& mesh, layout_type& layout) {
            r.initialize(mesh, layout);
            d.initialize(mesh, layout);
            q.initialize(mesh, layout);
        }

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        virtual void setOperator(OperatorF op) { op_m = std::move(op); }
        virtual void setPreconditioner(
            [[maybe_unused]] OperatorF&& op,  // Operator passed to chebyshev and newton
            [[maybe_unused]] LowerF&& lower,  // Operator passed to 2-step gauss-seidel and ssor
            [[maybe_unused]] UpperF&& upper,  // Operator passed to 2-step gauss-seidel and ssor
            [[maybe_unused]] UpperLowerF&&
                upper_and_lower,  // Operator passed to 2-step gauss-seidel
            [[maybe_unused]] InverseDiagF&&
                inverse_diagonal,  // Operator passed to jacobi, 2-step gauss-seidel and ssor
            [[maybe_unused]] DiagF&& diagonal,  // Operator passed to SSOR
            [[maybe_unused]] double alpha,      // smallest eigenvalue of the operator
            [[maybe_unused]] double beta,       // largest eigenvalue of the operator
            [[maybe_unused]] std::string preconditioner_type =
                "",  // Name of the preconditioner that should be used
            [[maybe_unused]] int level =
                5,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            [[maybe_unused]] int degree =
                31,  // This is a dummy default parameter, actual default parameter should
            // be set in main
            [[maybe_unused]] int richardson_iterations =
                1,  // This is a dummy default parameter, actual default
            // parameter should be set in main
            [[maybe_unused]] int inner =
                5,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            [[maybe_unused]] int outer =
                1,  // This is a dummy default parameter, actual default parameter should be
            [[maybe_unused]] double omega =
                1  // This is a dummy default parameter, actual default parameter should be
                   // set in main
        ) {}
        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        virtual int getIterationCount() { return iterations_m; }

        virtual void operator()(lhs_type& lhs, rhs_type& rhs,
                                const ParameterList& params) override {
            constexpr unsigned Dim = lhs_type::dim;

            static IpplTimings::TimerRef cg_ops = IpplTimings::getTimer("CG");
            static IpplTimings::TimerRef up_layout = IpplTimings::getTimer("updateLayout");
            static IpplTimings::TimerRef apply = IpplTimings::getTimer("applyOp");
            static IpplTimings::TimerRef inner = IpplTimings::getTimer("innerProduct");

            IpplTimings::startTimer(cg_ops);

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            IpplTimings::startTimer(up_layout);
            r.updateLayout(lhs.getLayout());
            d.updateLayout(lhs.getLayout());
            q.updateLayout(lhs.getLayout());
            IpplTimings::stopTimer(up_layout);

            using bc_type  = BConds<lhs_type, Dim>;
            bc_type lhsBCs = lhs.getFieldBC();
            bc_type bc;

            bool allFacesPeriodic = true;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                FieldBC bcType = lhsBCs[i]->getBCType();
                if (bcType == PERIODIC_FACE) {
                    // If the LHS has periodic BCs, so does the residue
                    bc[i] = std::make_shared<PeriodicFace<lhs_type>>(i);
                } else if (bcType & CONSTANT_FACE) {
                    // If the LHS has constant BCs, the residue is zero on the BCs
                    // Bitwise AND with CONSTANT_FACE will succeed for ZeroFace or ConstantFace
                    bc[i]            = std::make_shared<ZeroFace<lhs_type>>(i);
                    allFacesPeriodic = false;
                } else {
                    throw IpplException("PCG::operator()",
                                        "Only periodic or constant BCs for LHS supported.");
                    return;
                }
            }

            IpplTimings::startTimer(apply);
            r = rhs - op_m(lhs);
            IpplTimings::stopTimer(apply);

            d = r.deepCopy();
            d.setFieldBC(bc);

            IpplTimings::startTimer(inner);
            T delta1          = innerProduct(r, d);
            IpplTimings::stopTimer(inner);
            T delta0          = delta1;
            residueNorm       = Kokkos::sqrt(delta1);
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            while (iterations_m < maxIterations && residueNorm > tolerance) {

                IpplTimings::startTimer(apply);
                q = op_m(d);
                IpplTimings::stopTimer(apply);

                IpplTimings::startTimer(inner);
                T alpha = delta1 / innerProduct(d, q);
                IpplTimings::stopTimer(inner);
                lhs     = lhs + alpha * d;

                // The exact residue is given by
                // r = rhs - op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                r      = r - alpha * q;
                delta0 = delta1;
                IpplTimings::startTimer(inner);
                delta1 = innerProduct(r, r);
                IpplTimings::stopTimer(inner);
                T beta = delta1 / delta0;

                residueNorm = Kokkos::sqrt(delta1);
                d           = r + beta * d;
                ++iterations_m;
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
            IpplTimings::stopTimer(cg_ops);
        }

        virtual T getResidue() const { return residueNorm; }

    protected:
        OperatorF op_m;
        T residueNorm    = 0;
        int iterations_m = 0;

        // Workspaces, allocated once via initializeFields() and reused across
        // solves. Protected so derived solvers (e.g. PCG) can extend the
        // workspace set without redeclaring r, d, q as locals on every
        // operator() call.
        lhs_type r;
        lhs_type d;
        lhs_type q;
    };


    template <typename OperatorRet, typename LowerRet, typename UpperRet, typename UpperLowerRet,
              typename InverseDiagRet, typename T>
    class CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, FEMVector<T>, FEMVector<T> >
            : public SolverAlgorithm<FEMVector<T>, FEMVector<T>> {
        using Base = SolverAlgorithm<FEMVector<T>, FEMVector<T>>;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorF    = std::function<OperatorRet(lhs_type)>;
        using LowerF       = std::function<LowerRet(lhs_type)>;
        using UpperF       = std::function<UpperRet(lhs_type)>;
        using UpperLowerF  = std::function<UpperLowerRet(lhs_type)>;
        using InverseDiagF = std::function<InverseDiagRet(lhs_type)>;

        virtual ~CG() = default;

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        virtual void setOperator(OperatorF op) { op_m = std::move(op); }
        virtual void setPreconditioner(
            [[maybe_unused]] OperatorF&& op,  // Operator passed to chebyshev and newton
            [[maybe_unused]] LowerF&& lower,  // Operator passed to 2-step gauss-seidel
            [[maybe_unused]] UpperF&& upper,  // Operator passed to 2-step gauss-seidel
            [[maybe_unused]] UpperLowerF&&
                upper_and_lower,  // Operator passed to 2-step gauss-seidel
            [[maybe_unused]] InverseDiagF&&
                inverse_diagonal,           // Operator passed to jacobi and 2-step gauss-seidel
            [[maybe_unused]] double alpha,  // smallest eigenvalue of the operator
            [[maybe_unused]] double beta,   // largest eigenvalue of the operator
            [[maybe_unused]] std::string preconditioner_type =
                "",  // Name of the preconditioner that should be used
            [[maybe_unused]] int level =
                5,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            [[maybe_unused]] int degree =
                31,  // This is a dummy default parameter, actual default parameter should
            // be set in main
            [[maybe_unused]] int richardson_iterations =
                1,  // This is a dummy default parameter, actual default
            // parameter should be set in main
            [[maybe_unused]] int inner =
                5,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            [[maybe_unused]] int outer = 1  // This is a dummy default parameter, actual default
                                            // parameter should be set in main
        ) {}
        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        virtual int getIterationCount() { return iterations_m; }
        
        virtual void operator()(lhs_type& lhs, rhs_type& rhs,
                                const ParameterList& params) override {
            
            //constexpr unsigned Dim             = lhs_type::dim;
            //typename lhs_type::Mesh_t& mesh     = lhs.get_mesh();
            //typename lhs_type::Layout_t& layout = lhs.getLayout();

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            lhs_type r = lhs.deepCopy();
            r = 0;
            lhs_type d = lhs.deepCopy();
            d = 0;
            
           
            r = rhs - op_m(lhs);
            r.setHalo(0);
            d = r; //.deepCopy();
            //d.setFieldBC(bc);
            T delta1          = innerProduct(r, d);
            T delta0          = delta1;
            residueNorm       = Kokkos::sqrt(delta1);
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            lhs_type q = lhs.deepCopy();
            q = 0;
            

            while (iterations_m < maxIterations && residueNorm > tolerance) {
                q = op_m(d);
                d.setHalo(0);
                T alpha = delta1 / innerProduct(d, q);
                lhs     = lhs + alpha * d;

                // The exact residue is given by
                // r = rhs - op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                r      = r - alpha * q;
                delta0 = delta1;
                r.setHalo(0);
                delta1 = innerProduct(r, r);
                T beta = delta1 / delta0;

                residueNorm = Kokkos::sqrt(delta1);
                d           = r + beta * d;
                ++iterations_m;

                
            }
        }

        virtual T getResidue() const { return residueNorm; }

    protected:
        OperatorF op_m;
        T residueNorm    = 0;
        int iterations_m = 0;
    };


    template <typename OperatorRet, typename LowerRet, typename UpperRet, typename UpperLowerRet,
              typename InverseDiagRet, typename DiagRet, typename FieldLHS,
              typename FieldRHS = FieldLHS>
    class PCG : public CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, DiagRet,
                          FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorF    = std::function<OperatorRet(lhs_type)>;
        using LowerF       = std::function<LowerRet(lhs_type)>;
        using UpperF       = std::function<UpperRet(lhs_type)>;
        using UpperLowerF  = std::function<UpperLowerRet(lhs_type)>;
        using InverseDiagF = std::function<InverseDiagRet(lhs_type)>;
        using DiagF        = std::function<DiagRet(lhs_type)>;

        using mesh_type   = typename lhs_type::Mesh_t;
        using layout_type = typename lhs_type::Layout_t;

        PCG()
            : CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, DiagRet, FieldLHS,
                 FieldRHS>()
            , preconditioner_m(nullptr){};

        // Allocates the PCG workspace. Extends CG by adding the preconditioner
        // result buffer s and a NoBcFace staging buffer pcond_out. Called once
        // by the owning solver (e.g. PoissonCG) so that operator() does not
        // allocate per solve.
        void initializeFields(mesh_type& mesh, layout_type& layout) override {
            CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, DiagRet, FieldLHS,
               FieldRHS>::initializeFields(mesh, layout);
            s.initialize(mesh, layout);
            pcond_out.initialize(mesh, layout);
        }

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setPreconditioner(
            OperatorF&& op,                   // Operator passed to chebyshev and newton
            LowerF&& lower,                   // Operator passed to 2-step gauss-seidel
            UpperF&& upper,                   // Operator passed to 2-step gauss-seidel
            UpperLowerF&& upper_and_lower,    // Operator passed to 2-step gauss-seidel
            InverseDiagF&& inverse_diagonal,  // Operator passed to jacobi and 2-step gauss-seidel
            DiagF&& diagonal,                 // Operator passed to ssor
            double alpha,                     // smallest eigenvalue of the operator
            double beta,                      // largest eigenvalue of the operator
            std::string preconditioner_type = "",  // Name of the preconditioner that should be used
            int level = 5,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            int degree = 31,  // This is a dummy default parameter, actual default parameter should
            // be set in main
            int richardson_iterations = 4,  // This is a dummy default parameter, actual default
            // parameter should be set in main
            int inner = 2,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            int outer = 2,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            double omega = 1.57079632679  // This is a dummy default parameter, actual default
            // parameter should be set in main
            // default = pi/2 as this was found optimal during hyperparameter scan for test case 
            // (see https://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/BSc-mbolliger.pdf)
            ) override {
            if (preconditioner_type == "jacobi") {
                // Turn on damping parameter
                /*
                double w = 2.0 / ((alpha + beta));
                preconditioner_m = std::move(std::make_unique<jacobi_preconditioner<FieldLHS ,
                InverseDiagF>>(std::move(inverse_diagonal), w));
                */
                preconditioner_m =
                    std::move(std::make_unique<jacobi_preconditioner<FieldLHS, InverseDiagF>>(
                        std::move(inverse_diagonal)));
            } else if (preconditioner_type == "newton") {
                preconditioner_m = std::move(
                    std::make_unique<polynomial_newton_preconditioner<FieldLHS, OperatorF>>(
                        std::move(op), alpha, beta, level, 1e-3));
            } else if (preconditioner_type == "chebyshev") {
                preconditioner_m = std::move(
                    std::make_unique<polynomial_chebyshev_preconditioner<FieldLHS, OperatorF>>(
                        std::move(op), alpha, beta, degree, 1e-3));
            } else if (preconditioner_type == "richardson") {
                preconditioner_m =
                    std::move(std::make_unique<
                              richardson_preconditioner<FieldLHS, UpperLowerF, InverseDiagF>>(
                        std::move(upper_and_lower), std::move(inverse_diagonal),
                        richardson_iterations));
            } else if (preconditioner_type == "richardson_alt") {
                preconditioner_m =
                    std::move(std::make_unique<
                              richardson_preconditioner_alt<FieldLHS, OperatorF, InverseDiagF>>(
                        std::move(op), std::move(inverse_diagonal),
                        richardson_iterations));
            } else if (preconditioner_type == "gauss-seidel") {
                preconditioner_m = std::move(
                    std::make_unique<gs_preconditioner<FieldLHS, LowerF, UpperF, InverseDiagF>>(
                        std::move(lower), std::move(upper), std::move(inverse_diagonal), inner,
                        outer));
            } else if (preconditioner_type == "ssor") {
                preconditioner_m =
                    std::move(std::make_unique<
                              ssor_preconditioner<FieldLHS, LowerF, UpperF, InverseDiagF, DiagF>>(
                        std::move(lower), std::move(upper), std::move(inverse_diagonal),
                        std::move(diagonal), inner, outer, omega));
            } else {
                preconditioner_m = std::move(std::make_unique<preconditioner<FieldLHS>>());
            }
        }

        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {
            constexpr unsigned Dim = lhs_type::dim;

            if (preconditioner_m == nullptr) {
                throw IpplException("PCG::operator()",
                                    "Preconditioner has not been set for PCG solver");
            }

            this->iterations_m      = 0;
            const int maxIterations = params.get<int>("max_iterations");

            ++solve_count_m;
            IPPL_PCG_LOG("solve:enter solve#=" << solve_count_m
                         << " maxIter=" << maxIterations
                         << " precond=" << preconditioner_m->get_type());
            IPPL_PCG_SYNC("solve:enter#" << solve_count_m);

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            // r, d, q come from the CG base class; s is a PCG member. All are
            // pre-allocated via initializeFields(); operator() only refreshes
            // their layout so we track load-balancing repartitions of the lhs.
            this->r.updateLayout(lhs.getLayout());
            this->d.updateLayout(lhs.getLayout());
            s.updateLayout(lhs.getLayout());
            pcond_out.updateLayout(lhs.getLayout());
            this->q.updateLayout(lhs.getLayout());

            // Preconditioner scratch must follow the current lhs layout too,
            // otherwise its halo-exchange neighbor list goes out of sync with
            // r/d/s/q after a repartition and halo MPI calls deadlock. Each
            // preconditioner's init_fields() is responsible for being cheap on
            // the steady-state path (refreshing layout, not reallocating).
            preconditioner_m->init_fields(lhs);

            using bc_type  = BConds<lhs_type, Dim>;
            bc_type lhsBCs = lhs.getFieldBC();
            bc_type bc;

            bool allFacesPeriodic = true;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                FieldBC bcType = lhsBCs[i]->getBCType();
                if (bcType == PERIODIC_FACE) {
                    // If the LHS has periodic BCs, so does the residue
                    bc[i] = std::make_shared<PeriodicFace<lhs_type>>(i);
                } else if (bcType & CONSTANT_FACE) {
                    // If the LHS has constant BCs, the residue is zero on the BCs
                    // Bitwise AND with CONSTANT_FACE will succeed for ZeroFace or ConstantFace
                    bc[i]            = std::make_shared<ZeroFace<lhs_type>>(i);
                    allFacesPeriodic = false;
                } else {
                    throw IpplException("PCG::operator()",
                                        "Only periodic or constant BCs for LHS supported.");
                    return;
                }
            }

            IPPL_PCG_LOG("solve#=" << solve_count_m << " step=initial_residual");
            IPPL_PCG_SYNC("initial_residual:pre solve#" << solve_count_m);
            this->r = rhs - this->op_m(lhs);
            IPPL_PCG_SYNC("initial_residual:post solve#" << solve_count_m);
            // The preconditioner writes into pcond_out (NoBcFace, no halo MPI
            // from BC apply), then we hand the result over to d via an
            // expression assignment. d's PeriodicFace BCs must NOT be visible
            // during the preconditioner's internal operator chain — that would
            // trigger PeriodicFace::apply MPI inside pcond, which is what the
            // master code path avoids by returning a fresh NoBcFace field from
            // pcond.
            IPPL_PCG_LOG("solve#=" << solve_count_m << " step=initial_pcond:pre");
            IPPL_PCG_SYNC("initial_pcond:pre solve#" << solve_count_m);
            (*preconditioner_m)(this->r, pcond_out);
            IPPL_PCG_SYNC("initial_pcond:post solve#" << solve_count_m);
            IPPL_PCG_LOG("solve#=" << solve_count_m << " step=initial_pcond:post");
            this->d = T(1) * pcond_out;
            IPPL_PCG_SYNC("initial_setFieldBC:pre solve#" << solve_count_m);
            this->d.setFieldBC(bc);
            IPPL_PCG_SYNC("initial_setFieldBC:post solve#" << solve_count_m);

            T delta1          = innerProduct(this->r, this->d);
            T delta0           = delta1;
            this->residueNorm = Kokkos::sqrt(Kokkos::abs(delta1));
            const T tolerance = params.get<T>("tolerance") * this->residueNorm;
            IPPL_PCG_LOG("solve#=" << solve_count_m << " step=loop_start"
                         << " residue=" << this->residueNorm << " tol=" << tolerance);

            while (this->iterations_m < maxIterations && this->residueNorm > tolerance) {
                IPPL_PCG_LOG("solve#=" << solve_count_m << " iter=" << this->iterations_m
                             << " step=apply_op");
                IPPL_PCG_SYNC("iter_apply_op:pre solve#" << solve_count_m
                              << " iter=" << this->iterations_m);
                // q = op_m(d) writes the expression into q's existing storage
                // via operator=(Expression); no allocation, no extra deep copy.
                this->q = this->op_m(this->d);
                IPPL_PCG_SYNC("iter_apply_op:post solve#" << solve_count_m
                              << " iter=" << this->iterations_m);
                T alpha = delta1 / innerProduct(this->d, this->q);
                lhs     = lhs + alpha * this->d;

                // The exact residue is given by
                // r = rhs - BaseCG::op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                this->r = this->r - alpha * this->q;
                IPPL_PCG_LOG("solve#=" << solve_count_m << " iter=" << this->iterations_m
                             << " step=loop_pcond:pre");
                IPPL_PCG_SYNC("iter_pcond:pre solve#" << solve_count_m
                              << " iter=" << this->iterations_m);
                // s := M^{-1} r; preconditioner writes into s. s has NoBcFace
                // BCs (never set by setFieldBC), so its operator chain matches
                // master's NoBcFace scratch behaviour.
                (*preconditioner_m)(this->r, s);
                IPPL_PCG_SYNC("iter_pcond:post solve#" << solve_count_m
                              << " iter=" << this->iterations_m);
                IPPL_PCG_LOG("solve#=" << solve_count_m << " iter=" << this->iterations_m
                             << " step=loop_pcond:post");

                delta0 = delta1;
                delta1 = innerProduct(this->r, s);

                T beta            = delta1 / delta0;
                this->residueNorm = Kokkos::sqrt(Kokkos::abs(delta1));

                this->d = s + beta * this->d;
                ++this->iterations_m;
                IPPL_PCG_LOG("solve#=" << solve_count_m << " iter=" << this->iterations_m
                             << " step=loop_end residue=" << this->residueNorm);
            }
            IPPL_PCG_LOG("solve#=" << solve_count_m << " step=exit iters=" << this->iterations_m
                         << " residue=" << this->residueNorm);

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

    protected:
        std::unique_ptr<preconditioner<FieldLHS>> preconditioner_m;

        // Preconditioner result buffers, allocated once via initializeFields()
        // and reused across solves. Both deliberately keep their default
        // NoBcFace BCs so that the preconditioner's internal operator chain
        // does NOT trigger PeriodicFace::apply MPI calls -- d gets PeriodicFace
        // via setFieldBC and using d as the pcond result would change the
        // global MPI sequence relative to master, where pcond returned a
        // fresh NoBcFace field.
        lhs_type s;
        lhs_type pcond_out;

        // Counter so debug logs can identify *which* solve is mid-flight when
        // a halo deadlock fires. Per-rank, advances on every operator() call.
        int solve_count_m = 0;
    };

};  // namespace ippl

#endif
