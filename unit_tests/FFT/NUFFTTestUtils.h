//
// NUFFTTestUtils.h
//   Shared utilities for NUFFT unit tests
//
#ifndef IPPL_NUFFT_TEST_UTILS_H
#define IPPL_NUFFT_TEST_UTILS_H

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_Random.hpp>
#include <iomanip>
#include <iostream>

#include "Utility/ParameterList.h"

namespace ippl {
    namespace test {

        //=======================================================================
        // Particle Structure
        //=======================================================================

        template <typename T, class PLayout>
        struct Bunch : public ippl::ParticleBase<PLayout> {
            Bunch(PLayout& playout)
                : ippl::ParticleBase<PLayout>(playout) {
                this->addAttribute(Q);
            }

            using charge_container_type = ippl::ParticleAttrib<T>;
            charge_container_type Q;
        };

        //=======================================================================
        // Particle Generation Functors
        //=======================================================================

        template <typename VecT, typename ChargeT, class GeneratorPool, unsigned Dim>
        struct RandomParticleGenerator {
            using view_type        = typename ippl::detail::ViewType<VecT, 1>::view_type;
            using view_type_scalar = typename ippl::detail::ViewType<ChargeT, 1>::view_type;

            view_type x;
            view_type_scalar Q;
            GeneratorPool randPool;
            VecT minU, maxU;

            RandomParticleGenerator(view_type x_, view_type_scalar Q_, GeneratorPool randPool_,
                                    const VecT& minU_, const VecT& maxU_)
                : x(x_)
                , Q(Q_)
                , randPool(randPool_)
                , minU(minU_)
                , maxU(maxU_) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                typename GeneratorPool::generator_type randGen = randPool.get_state();

                for (unsigned d = 0; d < Dim; ++d) {
                    x(i)[d] = randGen.drand(minU[d], maxU[d]);
                }
                Q(i) = static_cast<ChargeT>(randGen.drand(0.0, 1.0));

                randPool.free_state(randGen);
            }
        };

        template <typename VecT, typename ChargeT, class GeneratorPool, unsigned Dim>
        struct ConstantParticleGenerator {
            using view_type        = typename ippl::detail::ViewType<VecT, 1>::view_type;
            using view_type_scalar = typename ippl::detail::ViewType<ChargeT, 1>::view_type;
            using scalar_type      = typename VecT::value_type;

            view_type x;
            view_type_scalar Q;
            scalar_type posValue;
            ChargeT chargeValue;

            ConstantParticleGenerator(view_type x_, view_type_scalar Q_, scalar_type posValue_,
                                      ChargeT chargeValue_)
                : x(x_)
                , Q(Q_)
                , posValue(posValue_)
                , chargeValue(chargeValue_) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                for (unsigned d = 0; d < Dim; ++d) {
                    x(i)[d] = posValue;
                }
                Q(i) = chargeValue;
            }
        };

        //=======================================================================
        // Field Generation Functors
        //=======================================================================

        template <typename FieldView, class GeneratorPool>
        struct RandomFieldGenerator {
            FieldView field;
            GeneratorPool randPool;
            int nghost;

            RandomFieldGenerator(FieldView field_, GeneratorPool randPool_, int nghost_)
                : field(field_)
                , randPool(randPool_)
                , nghost(nghost_) {}

            KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
                typename GeneratorPool::generator_type randGen = randPool.get_state();

                auto& val  = field(i + nghost, j + nghost, k + nghost);
                val.real() = randGen.drand(0.0, 1.0);
                val.imag() = randGen.drand(0.0, 1.0);

                randPool.free_state(randGen);
            }
        };

        //=======================================================================
        // Index Conversion Utilities
        //=======================================================================

        template <unsigned Dim>
        class IndexUtils {
        public:
            /**
             * @brief Check if a global index is owned by this rank's local domain
             */
            static bool isOwnedLocally(const ippl::NDIndex<Dim>& lDom,
                                       const ippl::Vector<int, Dim>& globalIdx) {
                for (unsigned d = 0; d < Dim; ++d) {
                    if (globalIdx[d] < lDom[d].first() || globalIdx[d] > lDom[d].last()) {
                        return false;
                    }
                }
                return true;
            }

            /**
             * @brief Convert global index to local index (with ghost offset)
             */
            static ippl::Vector<int, Dim> globalToLocal(const ippl::NDIndex<Dim>& lDom,
                                                        const ippl::Vector<int, Dim>& globalIdx,
                                                        int nghost) {
                ippl::Vector<int, Dim> localIdx;
                for (unsigned d = 0; d < Dim; ++d) {
                    localIdx[d] = globalIdx[d] - lDom[d].first() + nghost;
                }
                return localIdx;
            }

            /**
             * @brief Convert centered frequency to corner-DC format
             * Centered: k in [-N/2, N/2-1]
             * Corner:   k in [0, N-1] (no upsampling) or [0, 2*N-1] (with upsampling)
             */
            static ippl::Vector<int, Dim> centeredToCornerDC(const ippl::Vector<int, Dim>& kVec,
                                                             const ippl::Vector<int, Dim>& nModes,
                                                             bool useUpsampling = false) {
                ippl::Vector<int, Dim> cornerIdx;
                for (unsigned d = 0; d < Dim; ++d) {
                    if (kVec[d] >= 0) {
                        cornerIdx[d] = kVec[d];
                    } else {
                        // With upsampling, grid is 2x larger, so use 2*nModes for wrapping
                        cornerIdx[d] = (useUpsampling ? 2 : 1) * nModes[d] + kVec[d];
                    }
                }
                return cornerIdx;
            }

            /**
             * @brief Convert corner-DC index to centered frequency
             * Corner:   k in [0, N-1]
             * Centered: k in [-N/2, N/2-1]
             */
            static ippl::Vector<int, Dim> cornerToCenteredDC(const ippl::Vector<int, Dim>& cornerIdx,
                                                             const ippl::Vector<int, Dim>& nModes) {
                ippl::Vector<int, Dim> kVec;
                for (unsigned d = 0; d < Dim; ++d) {
                    if (cornerIdx[d] < nModes[d] / 2) {
                        kVec[d] = cornerIdx[d];
                    } else {
                        kVec[d] = cornerIdx[d] - nModes[d];
                    }
                }
                return kVec;
            }
        };

        //=======================================================================
        // DFT Reference Implementations
        //=======================================================================

        template <typename T, unsigned Dim>
        class DFTReference {
        public:
            /**
             * @brief Compute Type-1 DFT at a single mode
             * Type-1: f_k = sum_j c_j * exp(-i * k * x_j)
             * @param R Particle positions view
             * @param Q Particle charges view
             * @param kVec Frequency vector (centered)
             * @param hx Grid spacing
             * @param nModes Number of modes per dimension
             * @param nloc Number of local particles
             * @return Complex DFT value (local contribution)
             */
            template <typename PosView, typename ChargeView>
            static Kokkos::complex<T> computeType1ModeLocal(const PosView& R, const ChargeView& Q,
                                                            const ippl::Vector<int, Dim>& kVec,
                                                            const ippl::Vector<T, Dim>& hx,
                                                            const ippl::Vector<int, Dim>& nModes,
                                                            size_t nloc) {
                const T pi = Kokkos::numbers::pi_v<T>;
                Kokkos::complex<T> dftLocal(0.0, 0.0);
                const Kokkos::complex<T> imag = {0.0, 1.0};

                Kokkos::parallel_reduce(
                    "DFT_Type1_Local", nloc,
                    KOKKOS_LAMBDA(const size_t idx, Kokkos::complex<T>& val) {
                        T arg = 0.0;
                        for (unsigned d = 0; d < Dim; ++d) {
                            arg += (2 * pi / (hx[d] * nModes[d])) * kVec[d] * R(idx)[d];
                        }
                        // Type-1: exp(-i * k * x_j)
                        val += (Kokkos::cos(arg) - imag * Kokkos::sin(arg)) * Q(idx);
                    },
                    Kokkos::Sum<Kokkos::complex<T>>(dftLocal));

                return dftLocal;
            }

            /**
             * @brief Compute Type-1 DFT at a single mode (with MPI reduction)
             */
            template <typename PosView, typename ChargeView>
            static Kokkos::complex<T> computeType1Mode(const PosView& R, const ChargeView& Q,
                                                       const ippl::Vector<int, Dim>& kVec,
                                                       const ippl::Vector<T, Dim>& hx,
                                                       const ippl::Vector<int, Dim>& nModes,
                                                       size_t nloc) {
                auto dftLocal = computeType1ModeLocal(R, Q, kVec, hx, nModes, nloc);

                // Global reduce
                Kokkos::complex<T> dftGlobal(0.0, 0.0);
                T sendBuf[2] = {dftLocal.real(), dftLocal.imag()};
                T recvBuf[2] = {0.0, 0.0};
                MPI_Allreduce(sendBuf, recvBuf, 2,
                              std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM,
                              ippl::Comm->getCommunicator());
                dftGlobal = Kokkos::complex<T>(recvBuf[0], recvBuf[1]);

                return dftGlobal;
            }

            /**
             * @brief Compute Type-2 DFT at a single particle position
             * Type-2: q(x_j) = sum_k f_k * exp(+i * k * x_j)
             * @param field Field view
             * @param testPos Test particle position
             * @param lDom Local domain
             * @param hx Grid spacing (not used, domain is [0, 2pi])
             * @param nModes Number of modes per dimension
             * @param nghost Number of ghost cells
             * @return Complex DFT value (local contribution)
             */
            template <typename FieldView>
            static Kokkos::complex<T> computeType2ValueLocal(const FieldView& field,
                                                             const ippl::Vector<T, Dim>& testPos,
                                                             const ippl::NDIndex<Dim>& lDom,
                                                             const ippl::Vector<T, Dim>& hx,
                                                             const ippl::Vector<int, Dim>& nModes,
                                                             int nghost) {
                Kokkos::complex<T> dftLocal(0.0, 0.0);
                const Kokkos::complex<T> imag = {0.0, 1.0};

                // Get local domain bounds
                int localIStart = lDom[0].first();
                int localJStart = lDom[1].first();
                int localKStart = lDom[2].first();
                int localNi     = lDom[0].length();
                int localNj     = lDom[1].length();
                int localNk     = lDom[2].length();

                auto testPosD = testPos;
                auto nModesD  = nModes;

                using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
                Kokkos::parallel_reduce(
                    "DFT_Type2_Local", mdrange_type({0, 0, 0}, {localNi, localNj, localNk}),
                    KOKKOS_LAMBDA(const int li, const int lj, const int lk,
                                  Kokkos::complex<T>& val) {
                        // Convert local index to global index
                        int i = li + localIStart;
                        int j = lj + localJStart;
                        int k = lk + localKStart;

                        // Corner indexing -> centered integer frequency
                        int kc0 = (i < nModesD[0] / 2 ? i : i - nModesD[0]);
                        int kc1 = (j < nModesD[1] / 2 ? j : j - nModesD[1]);
                        int kc2 = (k < nModesD[2] / 2 ? k : k - nModesD[2]);

                        // Domain length is 2*pi, so frequency factor is 1
                        T arg = 0.0;
                        arg += kc0 * testPosD[0];
                        arg += kc1 * testPosD[1];
                        arg += kc2 * testPosD[2];

                        auto fk = field(li + nghost, lj + nghost, lk + nghost);

                        // Type-2: exp(+i * k * x_j)
                        val += (Kokkos::cos(arg) + imag * Kokkos::sin(arg)) * fk;
                    },
                    Kokkos::Sum<Kokkos::complex<T>>(dftLocal));

                return dftLocal;
            }

            /**
             * @brief Compute Type-2 DFT at a single particle position (with MPI reduction)
             */
            template <typename FieldView>
            static Kokkos::complex<T> computeType2Value(const FieldView& field,
                                                        const ippl::Vector<T, Dim>& testPos,
                                                        const ippl::NDIndex<Dim>& lDom,
                                                        const ippl::Vector<T, Dim>& hx,
                                                        const ippl::Vector<int, Dim>& nModes,
                                                        int nghost) {
                auto dftLocal =
                    computeType2ValueLocal(field, testPos, lDom, hx, nModes, nghost);

                // Global reduce
                Kokkos::complex<T> dftGlobal(0.0, 0.0);
                T sendBuf[2] = {dftLocal.real(), dftLocal.imag()};
                T recvBuf[2] = {0.0, 0.0};
                MPI_Allreduce(sendBuf, recvBuf, 2,
                              std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM,
                              ippl::Comm->getCommunicator());
                dftGlobal = Kokkos::complex<T>(recvBuf[0], recvBuf[1]);

                return dftGlobal;
            }
        };

        //=======================================================================
        // Parameter Builders
        //=======================================================================

        struct NUFFTParams {
            /**
             * @brief Create native NUFFT parameters
             */
            template <typename T>
            static ippl::ParameterList createNativeParams(T tolerance,
                                                          bool useUpsampling,
                                                          const std::string& spreadMethod = "tiled",
                                                          const std::string& gatherMethod = "atomic_sort") {
                ippl::ParameterList params;

                params.add("tolerance", tolerance);
                params.add("use_upsampled_inputs", useUpsampling);
                params.add("use_finufft", false);
                params.add("use_kokkos_nufft", false);
                params.add("spread_method", spreadMethod);
                params.add("gather_method", gatherMethod);
                params.add("sort", true);
                params.add("tile_size_3d", 6);
                params.add("z_tiles", 1);

#ifdef ENABLE_GPU_NUFFT
                params.add("gpu_method", 1);
                params.add("gpu_sort", 0);
                params.add("gpu_kerevalmeth", 1);
#else
                params.add("spread_kerevalmeth", 1);
                params.add("spread_sort", 2);
                params.add("nthreads", 0);
#endif

                return params;
            }

#ifdef ENABLE_FINUFFT
            /**
             * @brief Create FINUFFT backend parameters
             */
            template <typename T>
            static ippl::ParameterList createFinufftParams(T tolerance,
                                                           bool useUpsampling) {
                ippl::ParameterList params;

                params.add("tolerance", tolerance);
                params.add("use_upsampled_inputs", useUpsampling);
                params.add("use_finufft", true);
                params.add("use_finufft_defaults", false);
                params.add("use_kokkos_nufft", false);

#ifdef ENABLE_GPU_NUFFT
                params.add("gpu_method", 1);
                params.add("gpu_sort", 0);
                params.add("gpu_kerevalmeth", 1);
#else
                params.add("spread_kerevalmeth", 1);
                params.add("spread_sort", 2);
                params.add("nthreads", 0);
#endif

                return params;
            }
#endif
        };

        //=======================================================================
        // Error Computation Helpers
        //=======================================================================

        template <typename T>
        struct ErrorMetrics {
            T absError;
            T relError;

            static ErrorMetrics compute(const Kokkos::complex<T>& expected,
                                        const Kokkos::complex<T>& actual) {
                ErrorMetrics metrics;
                T expectedMag = Kokkos::abs(expected);
                T diff        = Kokkos::abs(expected - actual);

                metrics.absError = diff;
                metrics.relError = (expectedMag > 0) ? (diff / expectedMag) : diff;

                return metrics;
            }
        };

    }  // namespace test
}  // namespace ippl

#endif  // IPPL_NUFFT_TEST_UTILS_H
