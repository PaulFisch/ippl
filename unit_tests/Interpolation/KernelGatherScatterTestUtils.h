//
// GatherScatterTestUtils.h
//   Shared utilities for gather/scatter unit tests
//
#ifndef IPPL_KERNEL_GATHER_SCATTER_TEST_UTILS_H
#define IPPL_KERNEL_GATHER_SCATTER_TEST_UTILS_H

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_Random.hpp>
#include <random>

namespace ippl {
    namespace test {

        //=======================================================================
        // Tolerance Helper
        //=======================================================================

        template <typename T>
        constexpr T tolerance() {
            if constexpr (std::is_same_v<T, float>) {
                return 1e-5;
            } else {
                return 1e-12;
            }
        }

        //=======================================================================
        // Particle Structure
        //=======================================================================

        template <typename T, class PLayout>
        struct InterpolationBunch : public ippl::ParticleBase<PLayout> {
            InterpolationBunch(PLayout& playout)
                : ippl::ParticleBase<PLayout>(playout) {
                this->addAttribute(Q_scatter);
                this->addAttribute(Q_gather);
            }

            using charge_type = ippl::ParticleAttrib<T>;
            charge_type Q_scatter;  // Input to scatter
            charge_type Q_gather;   // Output from gather
        };

        /**
         * @brief Particle bunch with complex charges for kernel-based scatter/gather
         */
        template <typename T, class PLayout>
        struct KernelInterpolationBunch : public ippl::ParticleBase<PLayout> {
            KernelInterpolationBunch(PLayout& playout)
                : ippl::ParticleBase<PLayout>(playout) {
                this->addAttribute(Q_scatter);
                this->addAttribute(Q_gather);
            }

            using complex_type = Kokkos::complex<T>;
            // Use same execution space properties as position attribute R
            using charge_type = ippl::ParticleAttrib<complex_type, typename PLayout::position_execution_space>;
            charge_type Q_scatter;  // Input to scatter_kernel
            charge_type Q_gather;   // Output from gather with kernel
        };

        //=======================================================================
        // Field Initialization Helpers
        //=======================================================================

        /**
         * @brief Initialize field with constant value
         */
        template <typename Field, typename T>
        void initializeConstantField(Field& field, T value) {
            auto view = field.getView();
            int nghost = field.getNghost();
            using ExecutionSpace = Field::execution_space;

            const auto& lDom = field.getLayout().getLocalNDIndex();
            const unsigned dim = Field::dim;

            if constexpr (dim == 3) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();
                int nk = lDom[2].length();

                Kokkos::parallel_for(
                    "init_constant_field_3d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecutionSpace>({0, 0, 0}, {ni, nj, nk}),
                    KOKKOS_LAMBDA(int i, int j, int k) {
                        view(i + nghost, j + nghost, k + nghost) = value;
                    });
            } else if constexpr (dim == 2) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();

                Kokkos::parallel_for(
                    "init_constant_field_2d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>({0, 0}, {ni, nj}),
                    KOKKOS_LAMBDA(int i, int j) {
                        view(i + nghost, j + nghost) = value;
                    });
            }

            Kokkos::fence();
            field.fillHalo();
        }

        /**
         * @brief Initialize field with random values
         */
        template <typename Field, typename ExecSpace = typename Field::execution_space>
        void initializeRandomField(Field& field, unsigned seed = 42) {
            using value_type = typename Field::value_type;
            using generator_pool = Kokkos::Random_XorShift64_Pool<ExecSpace>;

            generator_pool randPool(seed);
            auto view = field.getView();
            int nghost = field.getNghost();

            const auto& lDom = field.getLayout().getLocalNDIndex();
            const unsigned dim = Field::dim;

            if constexpr (dim == 3) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();
                int nk = lDom[2].length();

                // Check if complex type using host-side check
                using is_complex_type = std::bool_constant<
                    std::is_same_v<value_type, Kokkos::complex<float>> ||
                    std::is_same_v<value_type, Kokkos::complex<double>>
                >;

                if constexpr (is_complex_type::value) {
                    using real_type = typename value_type::value_type;
                    Kokkos::parallel_for(
                        "init_random_field_3d",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>({0, 0, 0}, {ni, nj, nk}),
                        KOKKOS_LAMBDA(int i, int j, int k) {
                            typename generator_pool::generator_type gen = randPool.get_state();
                            view(i + nghost, j + nghost, k + nghost) =
                                value_type(gen.drand(0.0, 1.0), gen.drand(0.0, 1.0));
                            randPool.free_state(gen);
                        });
                } else {
                    Kokkos::parallel_for(
                        "init_random_field_3d",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>({0, 0, 0}, {ni, nj, nk}),
                        KOKKOS_LAMBDA(int i, int j, int k) {
                            typename generator_pool::generator_type gen = randPool.get_state();
                            view(i + nghost, j + nghost, k + nghost) = gen.drand(0.0, 1.0);
                            randPool.free_state(gen);
                        });
                }
            } else if constexpr (dim == 2) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();

                using is_complex_type = std::bool_constant<
                    std::is_same_v<value_type, Kokkos::complex<float>> ||
                    std::is_same_v<value_type, Kokkos::complex<double>>
                >;

                if constexpr (is_complex_type::value) {
                    using real_type = typename value_type::value_type;
                    Kokkos::parallel_for(
                        "init_random_field_2d",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>({0, 0}, {ni, nj}),
                        KOKKOS_LAMBDA(int i, int j) {
                            typename generator_pool::generator_type gen = randPool.get_state();
                            view(i + nghost, j + nghost) =
                                value_type(gen.drand(0.0, 1.0), gen.drand(0.0, 1.0));
                            randPool.free_state(gen);
                        });
                } else {
                    Kokkos::parallel_for(
                        "init_random_field_2d",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>({0, 0}, {ni, nj}),
                        KOKKOS_LAMBDA(int i, int j) {
                            typename generator_pool::generator_type gen = randPool.get_state();
                            view(i + nghost, j + nghost) = gen.drand(0.0, 1.0);
                            randPool.free_state(gen);
                        });
                }
            }

            Kokkos::fence();
            field.fillHalo();
        }

        /**
         * @brief Initialize field with analytic function for testing interpolation order
         * Uses polynomial: f(x,y,z) = x^order + y^order + z^order
         */
        template <typename Field, typename T, unsigned Dim>
        void initializeAnalyticField(Field& field, int order,
                                      const ippl::Vector<T, Dim>& origin,
                                      const ippl::Vector<T, Dim>& hx) {
            auto view = field.getView();
            int nghost = field.getNghost();
            using ExecSpace = Field::execution_space;

            const auto& lDom = field.getLayout().getLocalNDIndex();

            if constexpr (Dim == 3) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();
                int nk = lDom[2].length();
                int iStart = lDom[0].first();
                int jStart = lDom[1].first();
                int kStart = lDom[2].first();

                Kokkos::parallel_for(
                    "init_analytic_field_3d",
                    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {ni, nj, nk}),
                    KOKKOS_LAMBDA(int li, int lj, int lk) {
                        int gi = li + iStart;
                        int gj = lj + jStart;
                        int gk = lk + kStart;

                        T x = origin[0] + (gi + 0.5) * hx[0];  // Cell center
                        T y = origin[1] + (gj + 0.5) * hx[1];
                        T z = origin[2] + (gk + 0.5) * hx[2];

                        T value = Kokkos::pow(x, order) + Kokkos::pow(y, order) + Kokkos::pow(z, order);
                        view(li + nghost, lj + nghost, lk + nghost) = value;
                    });
            } else if constexpr (Dim == 2) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();
                int iStart = lDom[0].first();
                int jStart = lDom[1].first();

                Kokkos::parallel_for(
                    "init_analytic_field_2d",
                    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {ni, nj}),
                    KOKKOS_LAMBDA(int li, int lj) {
                        int gi = li + iStart;
                        int gj = lj + jStart;

                        T x = origin[0] + (gi + 0.5) * hx[0];
                        T y = origin[1] + (gj + 0.5) * hx[1];

                        T value = Kokkos::pow(x, order) + Kokkos::pow(y, order);
                        view(li + nghost, lj + nghost) = value;
                    });
            }

            Kokkos::fence();
            field.fillHalo();
        }

        //=======================================================================
        // Particle Generation Helpers
        //=======================================================================

        /**
         * @brief Create uniformly distributed particles in domain
         */
        template <typename Bunch, typename T, unsigned Dim>
        void createUniformParticles(Bunch& bunch,
                                    const ippl::Vector<T, Dim>& origin,
                                    const ippl::Vector<T, Dim>& extent,
                                    const ippl::Vector<size_t, Dim>& nPerDim,
                                    unsigned seed = 42) {
            size_t nlocTotal = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                nlocTotal *= nPerDim[d];
            }

            size_t nRanks = ippl::Comm->size();
            size_t myRank = ippl::Comm->rank();
            size_t nloc = nlocTotal / nRanks + (myRank < (nlocTotal % nRanks) ? 1 : 0);

            bunch.create(nloc);

            auto RView = bunch.R.getView();
            auto RHost = Kokkos::create_mirror_view(RView);

            size_t globalStart = myRank * (nlocTotal / nRanks) + std::min(myRank, nlocTotal % nRanks);

            for (size_t i = 0; i < nloc; ++i) {
                size_t globalIdx = globalStart + i;

                if constexpr (Dim == 3) {
                    size_t ix = globalIdx % nPerDim[0];
                    size_t iy = (globalIdx / nPerDim[0]) % nPerDim[1];
                    size_t iz = globalIdx / (nPerDim[0] * nPerDim[1]);

                    RHost(i)[0] = origin[0] + (ix + 0.5) * extent[0] / nPerDim[0];
                    RHost(i)[1] = origin[1] + (iy + 0.5) * extent[1] / nPerDim[1];
                    RHost(i)[2] = origin[2] + (iz + 0.5) * extent[2] / nPerDim[2];
                } else if constexpr (Dim == 2) {
                    size_t ix = globalIdx % nPerDim[0];
                    size_t iy = globalIdx / nPerDim[0];

                    RHost(i)[0] = origin[0] + (ix + 0.5) * extent[0] / nPerDim[0];
                    RHost(i)[1] = origin[1] + (iy + 0.5) * extent[1] / nPerDim[1];
                }
            }

            Kokkos::deep_copy(RView, RHost);
            Kokkos::fence();
            bunch.update();
        }

        /**
         * @brief Create randomly distributed particles
         */
        template <typename Bunch, typename T, unsigned Dim>
        void createRandomParticles(Bunch& bunch, size_t nloc,
                                    const ippl::Vector<T, Dim>& origin,
                                    const ippl::Vector<T, Dim>& extent,
                                    unsigned seed = 42) {
            bunch.create(nloc);

            std::mt19937_64 eng(seed + ippl::Comm->rank());
            std::uniform_real_distribution<T> dist(0.0, 1.0);

            auto RView = bunch.R.getView();
            auto RHost = Kokkos::create_mirror_view(RView);

            for (size_t i = 0; i < nloc; ++i) {
                for (unsigned d = 0; d < Dim; ++d) {
                    RHost(i)[d] = origin[d] + dist(eng) * extent[d];
                }
            }

            Kokkos::deep_copy(RView, RHost);
            Kokkos::fence();
            bunch.update();
        }

        /**
         * @brief Create particles at cell centers
         */
        template <typename Bunch, typename T, unsigned Dim>
        void createCellCenteredParticles(Bunch& bunch,
                                          const ippl::NDIndex<Dim>& lDom,
                                          const ippl::Vector<T, Dim>& origin,
                                          const ippl::Vector<T, Dim>& hx) {
            size_t nloc = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                nloc *= lDom[d].length();
            }

            bunch.create(nloc);

            auto RView = bunch.R.getView();
            auto RHost = Kokkos::create_mirror_view(RView);

            size_t idx = 0;
            if constexpr (Dim == 3) {
                for (int i = lDom[0].first(); i <= lDom[0].last(); ++i) {
                    for (int j = lDom[1].first(); j <= lDom[1].last(); ++j) {
                        for (int k = lDom[2].first(); k <= lDom[2].last(); ++k) {
                            RHost(idx)[0] = origin[0] + (i + 0.5) * hx[0];
                            RHost(idx)[1] = origin[1] + (j + 0.5) * hx[1];
                            RHost(idx)[2] = origin[2] + (k + 0.5) * hx[2];
                            ++idx;
                        }
                    }
                }
            } else if constexpr (Dim == 2) {
                for (int i = lDom[0].first(); i <= lDom[0].last(); ++i) {
                    for (int j = lDom[1].first(); j <= lDom[1].last(); ++j) {
                        RHost(idx)[0] = origin[0] + (i + 0.5) * hx[0];
                        RHost(idx)[1] = origin[1] + (j + 0.5) * hx[1];
                        ++idx;
                    }
                }
            }

            Kokkos::deep_copy(RView, RHost);
            Kokkos::fence();
            bunch.update();
        }

        //=======================================================================
        // Inner Product Computation (for Adjointness Testing)
        //=======================================================================

        /**
         * @brief Compute inner product of field values: <f, g>
         */
        template <typename Field>
        typename Field::value_type computeFieldInnerProduct(const Field& f, const Field& g) {
            using value_type = typename Field::value_type;
            using ExecSpace = typename Field::execution_space;

            auto fView = f.getView();
            auto gView = g.getView();
            int nghost = f.getNghost();

            const auto& lDom = f.getLayout().getLocalNDIndex();
            const unsigned dim = Field::dim;

            value_type local_sum(0);

            // Check if value_type is complex
            using is_complex_type = std::bool_constant<
                std::is_same_v<value_type, Kokkos::complex<float>> ||
                std::is_same_v<value_type, Kokkos::complex<double>>
            >;

            if constexpr (dim == 3) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();
                int nk = lDom[2].length();

                if constexpr (is_complex_type::value) {
                    Kokkos::parallel_reduce(
                        "field_inner_product_3d",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>({0, 0, 0}, {ni, nj, nk}),
                        KOKKOS_LAMBDA(int i, int j, int k, value_type& sum) {
                            sum += Kokkos::conj(fView(i + nghost, j + nghost, k + nghost))
                                   * gView(i + nghost, j + nghost, k + nghost);
                        },
                        local_sum);
                } else {
                    Kokkos::parallel_reduce(
                        "field_inner_product_3d",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>({0, 0, 0}, {ni, nj, nk}),
                        KOKKOS_LAMBDA(int i, int j, int k, value_type& sum) {
                            sum += fView(i + nghost, j + nghost, k + nghost)
                                   * gView(i + nghost, j + nghost, k + nghost);
                        },
                        local_sum);
                }
            } else if constexpr (dim == 2) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();

                if constexpr (is_complex_type::value) {
                    Kokkos::parallel_reduce(
                        "field_inner_product_2d",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>({0, 0}, {ni, nj}),
                        KOKKOS_LAMBDA(int i, int j, value_type& sum) {
                            sum += Kokkos::conj(fView(i + nghost, j + nghost))
                                   * gView(i + nghost, j + nghost);
                        },
                        local_sum);
                } else {
                    Kokkos::parallel_reduce(
                        "field_inner_product_2d",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>({0, 0}, {ni, nj}),
                        KOKKOS_LAMBDA(int i, int j, value_type& sum) {
                            sum += fView(i + nghost, j + nghost)
                                   * gView(i + nghost, j + nghost);
                        },
                        local_sum);
                }
            }

            Kokkos::fence();

            // MPI reduce
            value_type global_sum(0);
            if constexpr (is_complex_type::value) {
                using real_type = typename value_type::value_type;
                real_type send_buf[2] = {local_sum.real(), local_sum.imag()};
                real_type recv_buf[2] = {0, 0};
                MPI_Datatype mpi_type = std::is_same_v<real_type, float> ? MPI_FLOAT : MPI_DOUBLE;
                MPI_Allreduce(send_buf, recv_buf, 2, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());
                global_sum = value_type(recv_buf[0], recv_buf[1]);
            } else {
                MPI_Datatype mpi_type = std::is_same_v<value_type, float> ? MPI_FLOAT : MPI_DOUBLE;
                MPI_Allreduce(&local_sum, &global_sum, 1, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());
            }

            return global_sum;
        }

        /**
         * @brief Compute inner product of particle values: <q, p>
         */
        template <typename Bunch>
        typename Bunch::charge_type::value_type
        computeParticleInnerProduct(const Bunch& bunch,
                                     typename Bunch::charge_type& q,
                                     typename Bunch::charge_type& p) {
            using value_type = typename Bunch::charge_type::value_type;
            using ExecSpace = typename Bunch::exec_space;

            // Check if value_type is complex
            using is_complex_type = std::bool_constant<
                std::is_same_v<value_type, Kokkos::complex<float>> ||
                std::is_same_v<value_type, Kokkos::complex<double>>
            >;

            auto qView = q.getView();
            auto pView = p.getView();
            size_t nloc = bunch.getLocalNum();

            value_type local_sum(0);

            if constexpr (is_complex_type::value) {
                Kokkos::parallel_reduce(
                    "particle_inner_product",
                    Kokkos::RangePolicy<ExecSpace>(0,nloc),
                    KOKKOS_LAMBDA(size_t i, value_type& sum) {
                        sum += Kokkos::conj(qView(i)) * pView(i);
                    },
                    local_sum);
            } else {
                Kokkos::parallel_reduce(
                    "particle_inner_product",
                    Kokkos::RangePolicy<ExecSpace>(0,nloc),
                    KOKKOS_LAMBDA(size_t i, value_type& sum) {
                        sum += qView(i) * pView(i);
                    },
                    local_sum);
            }

            Kokkos::fence();

            // MPI reduce
            value_type global_sum(0);
            if constexpr (is_complex_type::value) {
                using real_type = typename value_type::value_type;
                real_type send_buf[2] = {local_sum.real(), local_sum.imag()};
                real_type recv_buf[2] = {0, 0};
                MPI_Datatype mpi_type = std::is_same_v<real_type, float> ? MPI_FLOAT : MPI_DOUBLE;
                MPI_Allreduce(send_buf, recv_buf, 2, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());
                global_sum = value_type(recv_buf[0], recv_buf[1]);
            } else {
                MPI_Datatype mpi_type = std::is_same_v<value_type, float> ? MPI_FLOAT : MPI_DOUBLE;
                MPI_Allreduce(&local_sum, &global_sum, 1, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());
            }

            return global_sum;
        }

        /**
         * @brief Verify adjointness: |<S*q, g> - <q, G*g>| / |<S*q, g>| < tol
         */
        template <typename T>
        bool verifyAdjointness(T leftIP, T rightIP, T tolerance) {
            // Check if T is complex
            using is_complex_type = std::bool_constant<
                std::is_same_v<T, Kokkos::complex<float>> ||
                std::is_same_v<T, Kokkos::complex<double>>
            >;

            T diff, magnitude;

            if constexpr (is_complex_type::value) {
                diff = Kokkos::abs(leftIP - rightIP);
                magnitude = Kokkos::abs(leftIP);
            } else {
                diff = std::abs(leftIP - rightIP);
                magnitude = std::abs(leftIP);
            }

            if (magnitude < tolerance) {
                return diff < tolerance;  // Both are near zero
            }

            T relError = diff / magnitude;
            return relError < tolerance;
        }

        //=======================================================================
        // Complex-Valued Helpers for Kernel-Based Scatter/Gather
        //=======================================================================

        /**
         * @brief Initialize field with random complex values
         */
        template <typename Field>
        void initializeRandomFieldComplex(Field& field, unsigned seed = 123) {
            using value_type = typename Field::value_type;
            using real_type = typename value_type::value_type;
            using exec_space = typename Field::execution_space;
            using generator_pool = Kokkos::Random_XorShift64_Pool<exec_space>;

            generator_pool randPool(seed);
            auto fieldView = field.getView();
            int nghost = field.getNghost();

            const auto& lDom = field.getLayout().getLocalNDIndex();
            const unsigned dim = Field::dim;

            if constexpr (dim == 3) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();
                int nk = lDom[2].length();

                Kokkos::parallel_for(
                    "init_random_complex_field_3d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec_space>({0, 0, 0}, {ni, nj, nk}),
                    KOKKOS_LAMBDA(int i, int j, int k) {
                        typename generator_pool::generator_type gen = randPool.get_state();
                        real_type re = gen.drand(-1.0, 1.0);
                        real_type im = gen.drand(-1.0, 1.0);
                        fieldView(i + nghost, j + nghost, k + nghost) = value_type(re, im);
                        randPool.free_state(gen);
                    });
            } else if constexpr (dim == 2) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();

                Kokkos::parallel_for(
                    "init_random_complex_field_2d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>({0, 0}, {ni, nj}),
                    KOKKOS_LAMBDA(int i, int j) {
                        typename generator_pool::generator_type gen = randPool.get_state();
                        real_type re = gen.drand(-1.0, 1.0);
                        real_type im = gen.drand(-1.0, 1.0);
                        fieldView(i + nghost, j + nghost) = value_type(re, im);
                        randPool.free_state(gen);
                    });
            }

            Kokkos::fence();
            field.fillHalo();
        }

        /**
         * @brief Compute inner product of complex fields: <f, g> = sum(conj(f) * g)
         */
        template <typename Field>
        typename Field::value_type computeFieldInnerProductComplex(const Field& f, const Field& g) {
            using complex_type = typename Field::value_type;
            using real_type = typename complex_type::value_type;
            using ExecSpace = Field::execution_space;

            auto fView = f.getView();
            auto gView = g.getView();
            int nghost = f.getNghost();

            const auto& lDom = f.getLayout().getLocalNDIndex();
            const unsigned dim = Field::dim;

            real_type local_re = 0.0;
            real_type local_im = 0.0;

            if constexpr (dim == 3) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();
                int nk = lDom[2].length();

                Kokkos::parallel_reduce(
                    "complex_field_inner_product_3d",
                    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {ni, nj, nk}),
                    KOKKOS_LAMBDA(int i, int j, int k, real_type& sum_re, real_type& sum_im) {
                        complex_type fval = fView(i + nghost, j + nghost, k + nghost);
                        complex_type gval = gView(i + nghost, j + nghost, k + nghost);
                        complex_type prod = Kokkos::conj(fval) * gval;
                        sum_re += prod.real();
                        sum_im += prod.imag();
                    },
                    local_re, local_im);
            } else if constexpr (dim == 2) {
                int ni = lDom[0].length();
                int nj = lDom[1].length();

                Kokkos::parallel_reduce(
                    "complex_field_inner_product_2d",
                    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {ni, nj}),
                    KOKKOS_LAMBDA(int i, int j, real_type& sum_re, real_type& sum_im) {
                        complex_type fval = fView(i + nghost, j + nghost);
                        complex_type gval = gView(i + nghost, j + nghost);
                        complex_type prod = Kokkos::conj(fval) * gval;
                        sum_re += prod.real();
                        sum_im += prod.imag();
                    },
                    local_re, local_im);
            }

            Kokkos::fence();

            // MPI reduce
            real_type global_re = 0.0;
            real_type global_im = 0.0;
            MPI_Datatype mpi_type = std::is_same_v<real_type, float> ? MPI_FLOAT : MPI_DOUBLE;
            MPI_Allreduce(&local_re, &global_re, 1, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());
            MPI_Allreduce(&local_im, &global_im, 1, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());

            return complex_type(global_re, global_im);
        }

        /**
         * @brief Compute inner product of complex particle values: <q, p> = sum(conj(q) * p)
         */
        template <typename Bunch, typename ChargeAttrib>
        typename ChargeAttrib::value_type
        computeParticleInnerProductComplex(const Bunch& bunch,
                                            ChargeAttrib& q,
                                            ChargeAttrib& p) {
            using complex_type = typename ChargeAttrib::value_type;
            using real_type = typename complex_type::value_type;
            using ExecSpace = ChargeAttrib::execution_space;

            auto qView = q.getView();
            auto pView = p.getView();
            size_t nloc = bunch.getLocalNum();

            real_type local_re = 0.0;
            real_type local_im = 0.0;

            Kokkos::parallel_reduce(
                "complex_particle_inner_product",
                Kokkos::RangePolicy<ExecSpace>(0,nloc),
                KOKKOS_LAMBDA(size_t i, real_type& sum_re, real_type& sum_im) {
                    complex_type qval = qView(i);
                    complex_type pval = pView(i);
                    complex_type prod = Kokkos::conj(qval) * pval;
                    sum_re += prod.real();
                    sum_im += prod.imag();
                },
                local_re, local_im);

            Kokkos::fence();

            // MPI reduce
            real_type global_re = 0.0;
            real_type global_im = 0.0;
            MPI_Datatype mpi_type = std::is_same_v<real_type, float> ? MPI_FLOAT : MPI_DOUBLE;
            MPI_Allreduce(&local_re, &global_re, 1, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());
            MPI_Allreduce(&local_im, &global_im, 1, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());

            return complex_type(global_re, global_im);
        }

        /**
         * @brief Verify adjointness for complex types
         */
        template <typename ComplexT, typename RealT>
        bool verifyAdjointnessComplex(ComplexT leftIP, ComplexT rightIP, RealT tolerance) {
            RealT diff = Kokkos::abs(leftIP - rightIP);
            RealT magnitude = Kokkos::abs(leftIP);

            if (magnitude < tolerance) {
                return diff < tolerance;  // Both are near zero
            }

            RealT relError = diff / magnitude;
            return relError < tolerance;
        }

    }  // namespace test
}  // namespace ippl

#endif  // IPPL_KERNEL_GATHER_SCATTER_TEST_UTILS_H
