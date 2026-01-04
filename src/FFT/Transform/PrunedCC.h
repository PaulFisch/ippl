#ifndef IPPL_FFT_TRANSFORM_PRUNEDCC_HPP
#define IPPL_FFT_TRANSFORM_PRUNEDCC_HPP

#include <array>
#include <mpi.h>

#include "Utility/IpplTimings.h"
#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

namespace ippl {

    namespace fft {
        //=========================================================================
        // Backend Traits - add to your Backend.h or Traits.h
        //=========================================================================

        template <typename Backend>
        struct backend_traits {
            static constexpr bool supports_native_batched = false;
        };

        // Specialize for HeFFTe (assuming version 2.3+)
        template <typename T, unsigned Dim, typename MemSpace>
        struct backend_traits<HeffteC2C<T, Dim, MemSpace>> {
            static constexpr bool supports_native_batched = true;
        };

        // cuFFTMp might also support batching
        #ifdef IPPL_ENABLE_CUFFTMP
        template <typename T, unsigned Dim, typename MemSpace>
        struct backend_traits<CuFFTMpC2C<T, Dim, MemSpace>> {
            static constexpr bool supports_native_batched = false; // adjust as needed
        };
        #endif

        template <typename Backend>
        inline constexpr bool supports_native_batched_v = backend_traits<Backend>::supports_native_batched;

    }  // namespace fft

    //=========================================================================
    // Pruned Complex-to-Complex Transform
    //=========================================================================

    template <typename ComplexField>
    class FFT<PrunedCCTransform, ComplexField> {
    public:
        static constexpr unsigned Dim   = ComplexField::dim;
        static constexpr int NumSubFFTs = 1 << Dim;

        using Complex_t = typename ComplexField::value_type;
        using T         = typename Complex_t::value_type;
        using MemSpace  = typename ComplexField::memory_space;
        using ExecSpace = typename ComplexField::execution_space;
        using Layout_t  = FieldLayout<Dim>;

#ifdef IPPL_ENABLE_CUFFTMP
        using Backend_t = fft::CuFFTMpC2C<T, Dim, MemSpace>;
#else
        using Backend_t = fft::HeffteC2C<T, Dim, MemSpace>;
#endif

        static constexpr bool UseBatched = fft::supports_native_batched_v<Backend_t>;

        using GPUOps     = fft::Stream<MemSpace>;
        using Stream_t   = typename GPUOps::stream_type;
        using DeviceExec = typename GPUOps::exec_space;
        using TempView_t = Kokkos::View<Complex_t***, Kokkos::LayoutLeft, MemSpace>;

        // Contiguous buffer for batched transforms
        using BatchBuffer_t = Kokkos::View<Complex_t*, Kokkos::LayoutLeft, MemSpace>;

        FFT(const Layout_t& layoutIn, const Layout_t& layoutOut, const PruningParams<Dim>& pruning,
            const ParameterList& params)
            : pruning_(pruning) {
            static_assert(Dim == 3, "Pruned FFT currently only supports 3D");

            auto& prunedLayout =
                (layoutOut.getLocalNDIndex().size() < layoutIn.getLocalNDIndex().size()) ? layoutOut
                                                                                         : layoutIn;

            std::array<long long, 3> low, high;
            fft::domainToBounds<Dim>(prunedLayout.getLocalNDIndex(), low, high);
            heffte::box3d<long long> box{low, high};

            if constexpr (UseBatched) {
                // Single batched backend
                MPI_Comm_dup(MPI_COMM_WORLD, &comms_[0]);
                backend_ = std::make_unique<Backend_t>(box, box, comms_[0], params, NumSubFFTs);

                // Compute batch stride
                batchSize_ = 1;
                for (unsigned d = 0; d < Dim; ++d) {
                    batchSize_ *= (high[d] - low[d] + 1);
                }
            } else {
                // Multiple independent backends with streams
                numConcurrent_ = std::clamp(params.get<int>("num_concurrent_ffts", 4), 1, NumSubFFTs);
                for (int s = 0; s < numConcurrent_; ++s) {
                    MPI_Comm_dup(MPI_COMM_WORLD, &comms_[s]);
                    GPUOps::create(streams_[s]);
                    backends_[s] = std::make_unique<Backend_t>(box, box, comms_[s], params);
                }
            }
        }

        ~FFT() {
            if constexpr (UseBatched) {
                MPI_Comm_free(&comms_[0]);
            } else {
                for (int s = 0; s < numConcurrent_; ++s) {
                    GPUOps::destroy(streams_[s]);
                    MPI_Comm_free(&comms_[s]);
                }
            }
        }

        void transform(TransformDirection direction, ComplexField& input, ComplexField& output,
                       int dir = 1) {
            if (direction == FORWARD) {
                forwardPruned(dir, input, output);
            } else {
                backwardPruned(dir, input, output);
            }
        }

        void forwardPruned(int dir, ComplexField& input, ComplexField& output);
        void backwardPruned(int dir, ComplexField& input, ComplexField& output);

    private:
        PruningParams<Dim> pruning_;

        // Batched mode members
        std::unique_ptr<Backend_t> backend_;
        BatchBuffer_t batchBuffer_;
        size_t batchSize_ = 0;

        // Stream-based mode members
        int numConcurrent_ = 1;
        std::array<std::unique_ptr<Backend_t>, NumSubFFTs> backends_;
        std::array<TempView_t, NumSubFFTs> temps_;
        std::array<MPI_Comm, NumSubFFTs> comms_{};
        std::array<Stream_t, NumSubFFTs> streams_{};

        // Helper to ensure batch buffer is allocated
        void ensureBatchBuffer(size_t singleSize) {
            const size_t totalSize = singleSize * NumSubFFTs;
            if (batchBuffer_.size() != totalSize) {
                batchBuffer_ = BatchBuffer_t("batch_buffer", totalSize);
            }
        }
    };

    //-------------------------------------------------------------------------
    // Forward Pruned C2C Implementation
    //-------------------------------------------------------------------------

    template <typename ComplexField>
    void FFT<PrunedCCTransform, ComplexField>::forwardPruned(int dir, ComplexField& input,
                                                             ComplexField& output) {
        static IpplTimings::TimerRef twiddleTimer = IpplTimings::getTimer("TwiddleAdd");
        static IpplTimings::TimerRef subFFTTimer  = IpplTimings::getTimer("subFFTs");

        auto inView     = input.getView();
        auto outView    = output.getView();
        const int ngIn  = input.getNghost();
        const int ngOut = output.getNghost();

        const auto& lDomPruned = output.getLayout().getLocalNDIndex();
        const auto& gDomFull   = input.getLayout().getDomain();
        const auto& modes      = pruning_.n_modes;

        Kokkos::deep_copy(outView, Complex_t(0, 0));

        double scale = 1.0;
        if (dir == 1) {
            for (unsigned d = 0; d < Dim; ++d) {
                scale *= double(modes[d]) / double(gDomFull[d].length());
            }
        }

        std::array<Vector<long, Dim>, NumSubFFTs> offsets;
        for (int k = 0; k < NumSubFFTs; ++k) {
            for (unsigned d = 0; d < Dim; ++d) {
                offsets[k][d] = (k >> d) & 1;
            }
        }

        Vector<int, Dim> localFirst;
        for (unsigned d = 0; d < Dim; ++d) {
            localFirst[d] = lDomPruned[d].first();
        }

        auto owned = output.getOwned();

        if constexpr (UseBatched) {
            //=================================================================
            // BATCHED PATH: Single batched FFT call
            //=================================================================

            ensureBatchBuffer(batchSize_);

            // Phase 1: Strided copy all sub-transforms into contiguous batch buffer
            IpplTimings::startTimer(subFFTTimer);

            auto batchBuffer = batchBuffer_;
            auto batchSize = batchSize_;
            Kokkos::parallel_for(
                "strided_copy_all_batches",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
                    {0, 0, 0, 0},
                    {NumSubFFTs, long(owned[0].length()), long(owned[1].length()),
                     long(owned[2].length())}),
                KOKKOS_LAMBDA(int k, int i0, int i1, int i2) {
                    int off0 = (k >> 0) & 1;
                    int off1 = (k >> 1) & 1;
                    int off2 = (k >> 2) & 1;

                    int si = i0 * 2 + off0 + ngIn;
                    int sj = i1 * 2 + off1 + ngIn;
                    int sk = i2 * 2 + off2 + ngIn;

                    // Linear index within this batch
                    size_t idx = i0 + owned[0].length() * (i1 + owned[1].length() * i2);
                    batchBuffer(k * batchSize + idx) = inView(si, sj, sk);
                });

            Kokkos::fence();

            // Single batched FFT call
            static IpplTimings::TimerRef heffteTimer = IpplTimings::getTimer("RunHeffte");
            IpplTimings::startTimer(heffteTimer);
            if (dir == 1) {
                backend_->forward(NumSubFFTs, batchBuffer_.data(), batchBuffer_.data());
            } else {
                backend_->backward(NumSubFFTs, batchBuffer_.data(), batchBuffer_.data());
            }
            IpplTimings::stopTimer(heffteTimer);

            IpplTimings::stopTimer(subFFTTimer);

            // Phase 2: Twiddle and accumulate
            IpplTimings::startTimer(twiddleTimer);

            auto batchBuf = batchBuffer_;
            auto bSize    = batchSize_;

            Kokkos::parallel_for(
                "twiddle_add_all_batches",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {ngOut, ngOut, ngOut},
                    {int(outView.extent(0)) - ngOut, int(outView.extent(1)) - ngOut,
                     int(outView.extent(2)) - ngOut}),
                KOKKOS_LAMBDA(int i, int j, int kk) {
                    int gi = i - ngOut + localFirst[0];
                    int gj = j - ngOut + localFirst[1];
                    int gk = kk - ngOut + localFirst[2];

                    int64_t f0 = (gi < int64_t(modes[0]) / 2)
                                     ? gi
                                     : int64_t(gDomFull[0].length()) - int64_t(modes[0]) + gi;
                    int64_t f1 = (gj < int64_t(modes[1]) / 2)
                                     ? gj
                                     : int64_t(gDomFull[1].length()) - int64_t(modes[1]) + gj;
                    int64_t f2 = (gk < int64_t(modes[2]) / 2)
                                     ? gk
                                     : int64_t(gDomFull[2].length()) - int64_t(modes[2]) + gk;

                    auto twiddle = [&](int64_t freq, int64_t N) {
                        double ang = -dir * 2.0 * M_PI * double(freq) / double(N);
                        return Complex_t(Kokkos::cos(ang), Kokkos::sin(ang));
                    };

                    Complex_t acc(0, 0);
                    for (int k = 0; k < NumSubFFTs; ++k) {
                        int off0 = (k >> 0) & 1;
                        int off1 = (k >> 1) & 1;
                        int off2 = (k >> 2) & 1;

                        Complex_t w(1.0, 0.0);
                        if (off0) w *= twiddle(f0, gDomFull[0].length());
                        if (off1) w *= twiddle(f1, gDomFull[1].length());
                        if (off2) w *= twiddle(f2, gDomFull[2].length());

                        size_t idx = (i - ngOut) + owned[0].length() *
                                    ((j - ngOut) + owned[1].length() * (kk - ngOut));
                        acc += w * batchBuf(k * bSize + idx);
                    }
                    outView(i, j, kk) = acc * scale;
                });

            IpplTimings::stopTimer(twiddleTimer);

        } else {
            //=================================================================
            // STREAM-BASED PATH: Original implementation
            //=================================================================

            // Ensure temps
            for (int s = 0; s < numConcurrent_; ++s) {
                static IpplTimings::TimerRef allocateTempTimer =
                    IpplTimings::getTimer("FFTAllocateTemp");
                IpplTimings::startTimer(allocateTempTimer);
                if (temps_[s].size() != output.getOwned().size()) {
                    temps_[s] = detail::shrinkView("pruned_temp_" + std::to_string(s), outView, ngOut);
                }
                IpplTimings::stopTimer(allocateTempTimer);
            }

            const int numBatches = (NumSubFFTs + numConcurrent_ - 1) / numConcurrent_;

            for (int batch = 0; batch < numBatches; ++batch) {
                const int start = batch * numConcurrent_;
                const int end   = std::min(start + numConcurrent_, NumSubFFTs);
                const int count = end - start;

                IpplTimings::startTimer(subFFTTimer);

                #pragma omp parallel for
                for (int local = 0; local < count; ++local) {
                    const int k = start + local;
                    auto offs   = offsets[k];
                    auto& temp  = temps_[local];
                    auto exec   = GPUOps::instance(streams_[local]);

                    Kokkos::parallel_for(
                        "strided_copy_forward",
                        Kokkos::MDRangePolicy<DeviceExec, Kokkos::Rank<3>>(
                            exec, {0, 0, 0},
                            {long(owned[0].length()), long(owned[1].length()),
                             long(owned[2].length())}),
                        KOKKOS_LAMBDA(int i0, int i1, int i2) {
                            int si           = i0 * 2 + offs[0] + ngIn;
                            int sj           = i1 * 2 + offs[1] + ngIn;
                            int sk           = i2 * 2 + offs[2] + ngIn;
                            temp(i0, i1, i2) = inView(si, sj, sk);
                        });
                    GPUOps::sync(streams_[local]);

                    static IpplTimings::TimerRef heffteTimer = IpplTimings::getTimer("RunHeffte");
                    IpplTimings::startTimer(heffteTimer);
                    if (dir == 1) {
                        backends_[local]->forward(temp.data(), temp.data());
                    } else {
                        backends_[local]->backward(temp.data(), temp.data());
                    }
                    IpplTimings::stopTimer(heffteTimer);
                }

                Kokkos::fence();
                for (int local = 0; local < count; ++local) {
                    GPUOps::sync(streams_[local]);
                }

                IpplTimings::stopTimer(subFFTTimer);

                IpplTimings::startTimer(twiddleTimer);

                for (int local = 0; local < count; ++local) {
                    const int k = start + local;
                    auto offs   = offsets[k];
                    auto& temp  = temps_[local];

                    Kokkos::parallel_for(
                        "twiddle_add_forward",
                        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                            {ngOut, ngOut, ngOut},
                            {int(outView.extent(0)) - ngOut, int(outView.extent(1)) - ngOut,
                             int(outView.extent(2)) - ngOut}),
                        KOKKOS_LAMBDA(int i, int j, int kk) {
                            int gi = i - ngOut + localFirst[0];
                            int gj = j - ngOut + localFirst[1];
                            int gk = kk - ngOut + localFirst[2];

                            int64_t f0 = (gi < int64_t(modes[0]) / 2)
                                             ? gi
                                             : int64_t(gDomFull[0].length()) - int64_t(modes[0]) + gi;
                            int64_t f1 = (gj < int64_t(modes[1]) / 2)
                                             ? gj
                                             : int64_t(gDomFull[1].length()) - int64_t(modes[1]) + gj;
                            int64_t f2 = (gk < int64_t(modes[2]) / 2)
                                             ? gk
                                             : int64_t(gDomFull[2].length()) - int64_t(modes[2]) + gk;

                            Complex_t w(1.0, 0.0);
                            auto twiddle = [&](int64_t freq, int64_t N) {
                                double ang = -dir * 2.0 * M_PI * double(freq) / double(N);
                                return Complex_t(Kokkos::cos(ang), Kokkos::sin(ang));
                            };

                            if (offs[0]) w *= twiddle(f0, gDomFull[0].length());
                            if (offs[1]) w *= twiddle(f1, gDomFull[1].length());
                            if (offs[2]) w *= twiddle(f2, gDomFull[2].length());

                            auto val = temp(i - ngOut, j - ngOut, kk - ngOut);
                            outView(i, j, kk) += w * val * scale;
                        });
                }

                IpplTimings::stopTimer(twiddleTimer);
            }
        }
    }

    //-------------------------------------------------------------------------
    // Backward Pruned C2C Implementation
    //-------------------------------------------------------------------------

    template <typename ComplexField>
    void FFT<PrunedCCTransform, ComplexField>::backwardPruned(int dir, ComplexField& input,
                                                              ComplexField& output) {
        static IpplTimings::TimerRef subIFFTTimer      = IpplTimings::getTimer("subIFFTs");
        static IpplTimings::TimerRef stridedWriteTimer = IpplTimings::getTimer("StridedWrite");

        auto inView     = input.getView();
        auto outView    = output.getView();
        const int ngIn  = input.getNghost();
        const int ngOut = output.getNghost();

        const auto& lDomPruned = input.getLayout().getLocalNDIndex();
        const auto& gDomFull   = output.getLayout().getDomain();
        const auto& modes      = pruning_.n_modes;

        Kokkos::deep_copy(outView, Complex_t(0, 0));

        std::array<Vector<long, Dim>, NumSubFFTs> offsets;
        for (int k = 0; k < NumSubFFTs; ++k) {
            for (unsigned d = 0; d < Dim; ++d) {
                offsets[k][d] = (k >> d) & 1;
            }
        }

        Vector<int, Dim> localFirst;
        for (unsigned d = 0; d < Dim; ++d) {
            localFirst[d] = lDomPruned[d].first();
        }

        auto owned = input.getOwned();

        if constexpr (UseBatched) {
            //=================================================================
            // BATCHED PATH
            //=================================================================

            ensureBatchBuffer(batchSize_);

            IpplTimings::startTimer(subIFFTTimer);

            auto batchBuf = batchBuffer_;
            auto bSize    = batchSize_;

            // Phase 1: Apply twiddles and pack into batch buffer
            Kokkos::parallel_for(
                "twiddle_pack_backward",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
                    {0, 0, 0, 0},
                    {NumSubFFTs, long(owned[0].length()), long(owned[1].length()),
                     long(owned[2].length())}),
                KOKKOS_LAMBDA(int k, int i0, int i1, int i2) {
                    int off0 = (k >> 0) & 1;
                    int off1 = (k >> 1) & 1;
                    int off2 = (k >> 2) & 1;

                    int gi = i0 + localFirst[0];
                    int gj = i1 + localFirst[1];
                    int gk = i2 + localFirst[2];

                    int64_t f0 = (gi < int64_t(modes[0]) / 2)
                                     ? gi
                                     : int64_t(gDomFull[0].length()) - int64_t(modes[0]) + gi;
                    int64_t f1 = (gj < int64_t(modes[1]) / 2)
                                     ? gj
                                     : int64_t(gDomFull[1].length()) - int64_t(modes[1]) + gj;
                    int64_t f2 = (gk < int64_t(modes[2]) / 2)
                                     ? gk
                                     : int64_t(gDomFull[2].length()) - int64_t(modes[2]) + gk;

                    Complex_t w(1.0, 0.0);
                    auto twiddle = [&](int64_t freq, int64_t N) {
                        double ang = dir * 2.0 * M_PI * double(freq) / double(N);
                        return Complex_t(Kokkos::cos(ang), Kokkos::sin(ang));
                    };

                    if (off0) w *= twiddle(f0, gDomFull[0].length());
                    if (off1) w *= twiddle(f1, gDomFull[1].length());
                    if (off2) w *= twiddle(f2, gDomFull[2].length());

                    auto input_val = inView(i0 + ngIn, i1 + ngIn, i2 + ngIn);
                    size_t idx = i0 + owned[0].length() * (i1 + owned[1].length() * i2);
                    batchBuf(k * bSize + idx) = w * input_val;
                });

            Kokkos::fence();

            // Single batched FFT
            if (dir == -1) {
                backend_->forward(NumSubFFTs, batchBuffer_.data(), batchBuffer_.data());
            } else {
                backend_->backward(NumSubFFTs, batchBuffer_.data(), batchBuffer_.data());
            }

            IpplTimings::stopTimer(subIFFTTimer);

            // Phase 2: Strided write from batch buffer
            IpplTimings::startTimer(stridedWriteTimer);

            Kokkos::parallel_for(
                "strided_write_all_batches",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
                    {0, 0, 0, 0},
                    {NumSubFFTs, long(owned[0].length()), long(owned[1].length()),
                     long(owned[2].length())}),
                KOKKOS_LAMBDA(int k, int i0, int i1, int i2) {
                    int off0 = (k >> 0) & 1;
                    int off1 = (k >> 1) & 1;
                    int off2 = (k >> 2) & 1;

                    int oi = i0 * 2 + off0 + ngOut;
                    int oj = i1 * 2 + off1 + ngOut;
                    int ok = i2 * 2 + off2 + ngOut;

                    size_t idx = i0 + owned[0].length() * (i1 + owned[1].length() * i2);
                    outView(oi, oj, ok) = batchBuf(k * bSize + idx);
                });

            IpplTimings::stopTimer(stridedWriteTimer);

        } else {
            //=================================================================
            // STREAM-BASED PATH: Original implementation
            //=================================================================

            for (int s = 0; s < numConcurrent_; ++s) {
                if (temps_[s].size() != input.getOwned().size()) {
                    temps_[s] = detail::shrinkView("pruned_ifft_temp_" + std::to_string(s), inView, ngIn);
                }
            }

            const int numBatches = (NumSubFFTs + numConcurrent_ - 1) / numConcurrent_;

            for (int batch = 0; batch < numBatches; ++batch) {
                const int start = batch * numConcurrent_;
                const int end   = std::min(start + numConcurrent_, NumSubFFTs);
                const int count = end - start;

                IpplTimings::startTimer(subIFFTTimer);

                #pragma omp parallel for
                for (int local = 0; local < count; ++local) {
                    const int k = start + local;
                    auto offs   = offsets[k];
                    auto& temp  = temps_[local];
                    auto exec   = GPUOps::instance(streams_[local]);

                    Kokkos::parallel_for(
                        "twiddle_multiply_backward",
                        Kokkos::MDRangePolicy<DeviceExec, Kokkos::Rank<3>>(
                            exec, {0, 0, 0},
                            {long(owned[0].length()), long(owned[1].length()),
                             long(owned[2].length())}),
                        KOKKOS_LAMBDA(int i0, int i1, int i2) {
                            int gi = i0 + localFirst[0];
                            int gj = i1 + localFirst[1];
                            int gk = i2 + localFirst[2];

                            int64_t f0 = (gi < int64_t(modes[0]) / 2)
                                             ? gi
                                             : int64_t(gDomFull[0].length()) - int64_t(modes[0]) + gi;
                            int64_t f1 = (gj < int64_t(modes[1]) / 2)
                                             ? gj
                                             : int64_t(gDomFull[1].length()) - int64_t(modes[1]) + gj;
                            int64_t f2 = (gk < int64_t(modes[2]) / 2)
                                             ? gk
                                             : int64_t(gDomFull[2].length()) - int64_t(modes[2]) + gk;

                            Complex_t w(1.0, 0.0);
                            auto twiddle = [&](int64_t freq, int64_t N) {
                                double ang = dir * 2.0 * M_PI * double(freq) / double(N);
                                return Complex_t(Kokkos::cos(ang), Kokkos::sin(ang));
                            };

                            if (offs[0]) w *= twiddle(f0, gDomFull[0].length());
                            if (offs[1]) w *= twiddle(f1, gDomFull[1].length());
                            if (offs[2]) w *= twiddle(f2, gDomFull[2].length());

                            auto input_val   = inView(i0 + ngIn, i1 + ngIn, i2 + ngIn);
                            temp(i0, i1, i2) = w * input_val;
                        });
                    GPUOps::sync(streams_[local]);

                    if (dir == -1) {
                        backends_[local]->forward(temp.data(), temp.data());
                    } else {
                        backends_[local]->backward(temp.data(), temp.data());
                    }
                }

                Kokkos::fence();
                for (int local = 0; local < count; ++local) {
                    GPUOps::sync(streams_[local]);
                }
                IpplTimings::stopTimer(subIFFTTimer);

                IpplTimings::startTimer(stridedWriteTimer);

                for (int local = 0; local < count; ++local) {
                    const int k = start + local;
                    auto offs   = offsets[k];
                    auto& temp  = temps_[local];

                    Kokkos::parallel_for(
                        "strided_write_backward",
                        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                            {0, 0, 0}, {long(owned[0].length()), long(owned[1].length()),
                                        long(owned[2].length())}),
                        KOKKOS_LAMBDA(int i0, int i1, int i2) {
                            int oi              = i0 * 2 + offs[0] + ngOut;
                            int oj              = i1 * 2 + offs[1] + ngOut;
                            int ok              = i2 * 2 + offs[2] + ngOut;
                            outView(oi, oj, ok) = temp(i0, i1, i2);
                        });
                }

                IpplTimings::stopTimer(stridedWriteTimer);
            }
        }
    }

}  // namespace ippl

#endif