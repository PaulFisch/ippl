#ifndef IPPL_FFT_BACKEND_HEFFTE_H
#define IPPL_FFT_BACKEND_HEFFTE_H

#include <heffte_fft3d.h>
#include <heffte_fft3d_r2c.h>
#include <memory>

#include "Utility/ParameterList.h"

#include "Field/BareField.h"

#include "FFT/Traits.h"
#include "FieldLayout/FieldLayout.h"

namespace ippl {
    namespace fft {

        template <typename T, typename MemSpace>
        inline void applyScale(Kokkos::complex<T>* data, T scale, size_t size) {
            Kokkos::View<Kokkos::complex<T>*, MemSpace> view(data, size);
            Kokkos::parallel_for(
                "HeffteC2C_scale", Kokkos::RangePolicy<typename MemSpace::execution_space>(0, size),
                KOKKOS_LAMBDA(const size_t i) { view(i) *= scale; });
            Kokkos::fence();
        }

        template <typename T, typename MemSpace>
        inline void applyScaleReal(T* data, T scale, size_t size) {
            Kokkos::View<T*, MemSpace> view(data, size);
            Kokkos::parallel_for(
                "HeffteC2C_scale", Kokkos::RangePolicy<typename MemSpace::execution_space>(0, size),
                KOKKOS_LAMBDA(const size_t i) { view(i) *= scale; });
            Kokkos::fence();
        }

        template <typename HeffteBackendT>
        heffte::plan_options makeHeffteOptions(const ParameterList& params) {
            auto opts = heffte::default_options<HeffteBackendT>();

            if (!params.get<bool>("use_heffte_defaults")) {
                opts.use_pencils = params.get<bool>("use_pencils");
                opts.use_reorder = params.get<bool>("use_reorder");

                if constexpr (is_available_v<HeffteGPU>) {
                    opts.use_gpu_aware = params.get<bool>("use_gpu_aware");
                }

                switch (params.get<int>("comm")) {
                    case a2a:
                        opts.algorithm = heffte::reshape_algorithm::alltoall;
                        break;
                    case a2av:
                        opts.algorithm = heffte::reshape_algorithm::alltoallv;
                        break;
                    case p2p:
                        opts.algorithm = heffte::reshape_algorithm::p2p;
                        break;
                    case p2p_pl:
                        opts.algorithm = heffte::reshape_algorithm::p2p_plined;
                        break;
                    default:
                        throw IpplException("FFT", "Unknown communication type");
                }
            }
            return opts;
        }

        //=============================================================================
        // heFFTe C2C
        //=============================================================================

        template <typename T, unsigned Dim, typename MemSpace>
        class HeffteC2C {
        public:
            using complex_t   = Kokkos::complex<T>;
            using backend_t   = typename HeffteBackend<MemSpace>::c2c;
            using heffte_t    = heffte::fft3d<backend_t, long long>;
            using workspace_t = typename heffte_t::template buffer_container<complex_t>;

            HeffteC2C(const heffte::box3d<long long>& inbox, const heffte::box3d<long long>& outbox,
                      MPI_Comm comm, const ParameterList& params) {
                auto opts  = makeHeffteOptions<backend_t>(params);
                heffte_    = std::make_shared<heffte_t>(inbox, outbox, comm, opts);
                workspace_ = workspace_t(heffte_->size_workspace());

                // Compute total size for manual scaling
                size_ = heffte_->size_outbox();
            }

            void forward(complex_t* in, complex_t* out) {
                heffte_->forward(in, out, workspace_.data(), heffte::scale::none);
                applyScale<T, MemSpace>(out, T(1) / static_cast<T>(size_), size_);
            }

            void backward(complex_t* in, complex_t* out) {
                heffte_->backward(in, out, workspace_.data(), heffte::scale::none);
            }

            std::size_t workspace_size() const { return heffte_->size_workspace(); }

        private:
            std::shared_ptr<heffte_t> heffte_;
            workspace_t workspace_;
            size_t size_;
        };

        //=============================================================================
        // heFFTe R2C
        //=============================================================================

        template <typename T, unsigned Dim, typename MemSpace>
        class HeffteR2C {
        public:
            using complex_t   = Kokkos::complex<T>;
            using backend_t   = typename HeffteBackend<MemSpace>::c2c;
            using heffte_t    = heffte::fft3d_r2c<backend_t, long long>;
            using workspace_t = typename heffte_t::template buffer_container<complex_t>;

            HeffteR2C(const heffte::box3d<long long>& inbox, const heffte::box3d<long long>& outbox,
                      int r2c_direction, MPI_Comm comm, const ParameterList& params) {
                auto opts  = makeHeffteOptions<backend_t>(params);
                heffte_    = std::make_shared<heffte_t>(inbox, outbox, r2c_direction, comm, opts);
                workspace_ = workspace_t(heffte_->size_workspace());

                // For R2C, size is the real input size (not the complex output size)
                real_size_    = heffte_->size_inbox();
                complex_size_ = heffte_->size_outbox();
            }

            void forward(T* in, complex_t* out) {
                heffte_->forward(in, out, workspace_.data(), heffte::scale::none);
                applyScale<T, MemSpace>(out, T(1) / static_cast<T>(real_size_), real_size_);
            }

            void backward(complex_t* in, T* out) {
                heffte_->backward(in, out, workspace_.data(), heffte::scale::none);
            }

        private:
            std::shared_ptr<heffte_t> heffte_;
            workspace_t workspace_;
            size_t real_size_;
            size_t complex_size_;
        };

        //=============================================================================
        // heFFTe Trigonometric (Sine, Cos, Cos1)
        //=============================================================================

        template <typename T, unsigned Dim, typename MemSpace, typename Tag>
        class HeffteTrig;

#define IPPL_FFT_DEFINE_HEFFTE_TRIG(TagType, member)                                              \
    template <typename T, unsigned Dim, typename MemSpace>                                        \
    class HeffteTrig<T, Dim, MemSpace, TagType> {                                                 \
    public:                                                                                       \
        using backend_t   = typename HeffteBackend<MemSpace>::member;                             \
        using heffte_t    = heffte::fft3d<backend_t, long long>;                                  \
        using workspace_t = typename heffte_t::template buffer_container<T>;                      \
                                                                                                  \
        HeffteTrig(const heffte::box3d<long long>& inbox, const heffte::box3d<long long>& outbox, \
                   MPI_Comm comm, const ParameterList& params) {                                  \
            auto opts  = makeHeffteOptions<backend_t>(params);                                    \
            heffte_    = std::make_shared<heffte_t>(inbox, outbox, comm, opts);                   \
            workspace_ = workspace_t(heffte_->size_workspace());                                  \
            size_      = heffte_->size_outbox();                                                  \
        }                                                                                         \
                                                                                                  \
        void forward(T* in, T* out) {                                                             \
            heffte_->forward(in, out, workspace_.data(), heffte::scale::none);                    \
            applyScaleReal<T, MemSpace>(out, T(1) / static_cast<T>(size_), size_);                \
        }                                                                                         \
                                                                                                  \
        void backward(T* in, T* out) {                                                            \
            heffte_->backward(in, out, workspace_.data(), heffte::scale::none);                   \
        }                                                                                         \
                                                                                                  \
    private:                                                                                      \
        std::shared_ptr<heffte_t> heffte_;                                                        \
        workspace_t workspace_;                                                                   \
        size_t size_;                                                                             \
    };

        IPPL_FFT_DEFINE_HEFFTE_TRIG(SineTransform, sin)
        IPPL_FFT_DEFINE_HEFFTE_TRIG(CosTransform, cos)
        IPPL_FFT_DEFINE_HEFFTE_TRIG(Cos1Transform, cos1)

#undef IPPL_FFT_DEFINE_HEFFTE_TRIG

    }  // namespace fft
}  // namespace ippl

#endif