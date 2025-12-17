#ifndef IPPL_FFT_TRANSFORM_NUFFT_HPP
#define IPPL_FFT_TRANSFORM_NUFFT_HPP

#include <cmath>
#include <memory>

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"
#include "FFT/NUFFT/NativeNUFFT.h"

#ifdef ENABLE_FINUFFT
    #include <finufft.h>
    #ifdef ENABLE_GPU_NUFFT
        #include <cufinufft.h>
    #endif
#endif

namespace ippl {

    template <typename T, class... Properties>
    class ParticleAttrib;

    template <typename RealField>
    class FFT<NUFFTransform, RealField> {
    public:
        static constexpr unsigned Dim = RealField::dim;

        using T          = typename RealField::value_type;
        using Complex_t  = Kokkos::complex<T>;
        using MemSpace   = typename RealField::memory_space;
        using ExecSpace  = typename RealField::execution_space;
        using Layout_t   = FieldLayout<Dim>;

        using ComplexField = typename Field<Complex_t, Dim,
                                            typename RealField::Mesh_t,
                                            typename RealField::Centering_t,
                                            ExecSpace>::uniform_type;

        using NativeNUFFT_t = NUFFT::NativeNUFFT<Dim, T, ExecSpace>;

#ifdef ENABLE_FINUFFT
        using FinufftTypes      = fft::FinufftTypes<T>;
        using finufft_complex_t = typename FinufftTypes::complex_type;
        using finufft_plan_t    = typename FinufftTypes::plan_type;
#endif

        // View types for temporary storage
        using view_field_type = Kokkos::View<
#ifdef ENABLE_FINUFFT
            finufft_complex_t***,
#else
            Complex_t***,
#endif
            Kokkos::LayoutLeft, MemSpace>;

        using view_particle_real_type = Kokkos::View<T*, Kokkos::LayoutLeft, MemSpace>;

        using view_particle_complex_type = Kokkos::View<
#ifdef ENABLE_FINUFFT
            finufft_complex_t*,
#else
            Complex_t*,
#endif
            Kokkos::LayoutLeft, MemSpace>;

        FFT(const Layout_t& layout,
            std::size_t localNp,
            int type,
            const ParameterList& params)
            : type_(type)
            , tol_(params.get<T>("tolerance", T(1e-6)))
            , useFinufft_(params.get<bool>("use_finufft", false))
            , useUpsampledInputs_(params.get<bool>("use_upsampled_inputs", false))
        {
            // Store grid dimensions
            const auto& domain = layout.getDomain();
            for (unsigned d = 0; d < Dim; ++d) {
                nModes_[d] = domain[d].length();
            }

            // Allocate temporary buffers
            const auto& lDom = layout.getLocalNDIndex();
            Kokkos::realloc(tempField_, lDom[0].length(), lDom[1].length(), lDom[2].length());

            for (unsigned d = 0; d < Dim; ++d) {
                Kokkos::realloc(tempR_[d], localNp);
            }
            Kokkos::realloc(tempQ_, localNp);

            // Initialize the selected backend
            initBackend(layout, params);
        }

        ~FFT() {
            cleanupBackend();
        }

        template <class... Properties>
        void transform(const ParticleAttrib<Vector<T, Dim>, Properties...>& R,
                       ParticleAttrib<T, Properties...>& Q,
                       ComplexField& f)
        {
            if constexpr (fft::is_available_v<fft::Finufft>) {
                if (useFinufft_) {
                    transformFinufft(R, Q, f);
                    return;
                }
            }

            transformNative(R, Q, f);
        }

    // private:
        int type_;
        T tol_;
        bool useFinufft_;
        bool useUpsampledInputs_;

        std::array<int64_t, 3> nModes_{1, 1, 1};

        // Temporary buffers
        view_field_type tempField_;
        std::array<view_particle_real_type, 3> tempR_;
        view_particle_complex_type tempQ_;

        // Native NUFFT implementation
        std::unique_ptr<NativeNUFFT_t> nativeNufft_;

#ifdef ENABLE_FINUFFT
        finufft_plan_t finufftPlan_{};
        int finufftError_ = 0;
#endif

        //=====================================================================
        // Backend Initialization
        //=====================================================================

        void initBackend(const Layout_t& layout, const ParameterList& params) {
            if constexpr (fft::is_available_v<fft::Finufft>) {
                if (useFinufft_) {
                    initFinufft(params);
                    return;
                }
            }

            // Default: Native NUFFT
            initNative(layout, params);
        }

        void initNative(const Layout_t& layout, const ParameterList& params) {
            Vector<std::size_t, Dim> nModesVec;
            for (unsigned d = 0; d < Dim; ++d) {
                nModesVec[d] = nModes_[d];
            }

            typename NativeNUFFT_t::Config cfg;
            cfg.tol   = tol_;
            cfg.sigma = params.get<T>("sigma", T(2.0));

            // Get default scatter/gather configs for execution space
            cfg.scatter_config = Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.gather_config  = Interpolation::GatherConfig::get_default<ExecSpace>();

            // Configure spread method
            std::string spreadMethod = params.get<std::string>("spread_method", "none");
            if (spreadMethod == "atomic") {
                cfg.scatter_config.method = Interpolation::ScatterMethod::Atomic;
            } else if (spreadMethod == "output_focused") {
                cfg.scatter_config.method = Interpolation::ScatterMethod::OutputFocused;
            } else if (spreadMethod == "tiled") {
                cfg.scatter_config.method = Interpolation::ScatterMethod::Tiled;
            }

            // Configure gather method
            std::string gatherMethod = params.get<std::string>("gather_method", "none");
            if (gatherMethod == "tiled") {
                cfg.gather_config.method = Interpolation::GatherMethod::Tiled;
            } else if (gatherMethod == "native") {
                cfg.gather_config.method = Interpolation::GatherMethod::Native;
            } else if (gatherMethod == "atomic") {
                cfg.gather_config.method = Interpolation::GatherMethod::Atomic;
            } else if (gatherMethod == "atomic_sort") {
                cfg.gather_config.method = Interpolation::GatherMethod::AtomicSort;
            }

            // Optional tuning parameters
            if (params.contains("tile_size_3d")) {
                cfg.scatter_config.tile_size_3d = params.get<int>("tile_size_3d");
            }
            if (params.contains("z_tiles")) {
                cfg.scatter_config.z_tiles = params.get<int>("z_tiles");
            }
            if (params.contains("team_size")) {
                cfg.scatter_config.team_size = params.get<int>("team_size");
                cfg.gather_config.team_size  = params.get<int>("team_size");
            }

            nativeNufft_ = std::make_unique<NativeNUFFT_t>(nModesVec, useUpsampledInputs_, cfg);
            nativeNufft_->initialize(layout, MPI_COMM_WORLD);
        }

        void initFinufft([[maybe_unused]] const ParameterList& params) {
#ifdef ENABLE_FINUFFT
    #ifdef ENABLE_GPU_NUFFT
            cufinufft_opts opts;
            cufinufft_default_opts(&opts);

            opts.gpu_method         = params.get<int>("gpu_method", opts.gpu_method);
            opts.gpu_sort           = params.get<int>("gpu_sort", opts.gpu_sort);
            opts.gpu_kerevalmeth    = params.get<int>("gpu_kerevalmeth", opts.gpu_kerevalmeth);
            opts.gpu_binsizex       = params.get<int>("gpu_binsizex", opts.gpu_binsizex);
            opts.gpu_binsizey       = params.get<int>("gpu_binsizey", opts.gpu_binsizey);
            opts.gpu_binsizez       = params.get<int>("gpu_binsizez", opts.gpu_binsizez);
            opts.gpu_maxsubprobsize = params.get<int>("gpu_maxsubprobsize", opts.gpu_maxsubprobsize);
            opts.gpu_maxbatchsize   = 0;  // Default, ignored for ntransf=1
    #else
            finufft_opts opts;
            finufft_default_opts(&opts);

            opts.spread_sort        = params.get<int>("spread_sort", opts.spread_sort);
            opts.spread_kerevalmeth = params.get<int>("spread_kerevalmeth", opts.spread_kerevalmeth);
            opts.nthreads           = params.get<int>("nthreads", opts.nthreads);
    #endif

            int iflag = (type_ == 1) ? 1 : -1;
            int dim   = static_cast<int>(Dim);

            finufftError_ = FinufftTypes::makeplan()(
                type_, dim, nModes_.data(), iflag, 1, tol_, &finufftPlan_, &opts);

            if (finufftError_ != 0) {
                throw IpplException("FFT<NUFFTransform>", "FINUFFT makeplan failed");
            }
#endif
        }

        void cleanupBackend() {
#ifdef ENABLE_FINUFFT
            if (useFinufft_ && finufftPlan_) {
                FinufftTypes::destroy()(finufftPlan_);
                finufftPlan_ = nullptr;
            }
#endif
            // nativeNufft_ cleaned up automatically by unique_ptr
        }

        //=====================================================================
        // Native NUFFT Transform
        //=====================================================================

        template <class... Properties>
        void transformNative(const ParticleAttrib<Vector<T, Dim>, Properties...>& R,
                             ParticleAttrib<T, Properties...>& Q,
                             ComplexField& f)
        {
            const auto localNp = R.getParticleCount();
            const auto& layout = f.getLayout();
            const auto& mesh   = f.get_mesh();
            const auto& dx     = mesh.getMeshSpacing();
            const auto& domain = layout.getDomain();

            Vector<T, Dim> Len;
            for (unsigned d = 0; d < Dim; ++d) {
                Len[d] = dx[d] * domain[d].length();
            }

            constexpr T twoPi = 2.0 * M_PI;
            auto Rview = R.getView();

            if (type_ == 1) {
                // Type 1: Particles -> Grid (spreading/adjoint)

                // Scale positions to [0, 2π)
                Kokkos::parallel_for(
                    "NUFFT_native_scale_to_2pi_type1",
                    localNp,
                    KOKKOS_LAMBDA(std::size_t i) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            Rview(i)[d] *= (twoPi / Len[d]);
                        }
                    });

                // Execute native type 1 transform
                nativeNufft_->type1(R, Q, f, useUpsampledInputs_);

                // Scale positions back
                Kokkos::parallel_for(
                    "NUFFT_native_scale_back_type1",
                    localNp,
                    KOKKOS_LAMBDA(std::size_t i) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            Rview(i)[d] *= (Len[d] / twoPi);
                        }
                    });

            } else if (type_ == 2) {
                // Type 2: Grid -> Particles (interpolation)

                // Scale positions to [0, 2π)
                Kokkos::parallel_for(
                    "NUFFT_native_scale_to_2pi_type2",
                    localNp,
                    KOKKOS_LAMBDA(std::size_t i) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            Rview(i)[d] *= (twoPi / Len[d]);
                        }
                    });

                // Execute native type 2 transform
                nativeNufft_->type2(f, R, Q, useUpsampledInputs_);

                // Scale positions back
                Kokkos::parallel_for(
                    "NUFFT_native_scale_back_type2",
                    localNp,
                    KOKKOS_LAMBDA(std::size_t i) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            Rview(i)[d] *= (Len[d] / twoPi);
                        }
                    });

            } else {
                throw IpplException("FFT<NUFFTransform>", "Only type 1 and type 2 NUFFT supported");
            }
        }

        //=====================================================================
        // FINUFFT Transform
        //=====================================================================

        template <class... Properties>
        void transformFinufft(
            [[maybe_unused]] const ParticleAttrib<Vector<T, Dim>, Properties...>& R,
            [[maybe_unused]] ParticleAttrib<T, Properties...>& Q,
            [[maybe_unused]] ComplexField& f)
        {
#ifdef ENABLE_FINUFFT
            const auto localNp = R.getParticleCount();
            const auto& layout = f.getLayout();
            const auto& mesh   = f.get_mesh();
            const auto& dx     = mesh.getMeshSpacing();
            const auto& domain = layout.getDomain();
            const int nghost   = f.getNghost();

            Vector<T, Dim> Len;
            for (unsigned d = 0; d < Dim; ++d) {
                Len[d] = dx[d] * domain[d].length();
            }

            constexpr T twoPi = 2.0 * M_PI;

            auto fview = f.getView();
            auto Rview = R.getView();
            auto Qview = Q.getView();

            // Ensure temp buffers are allocated
            const auto& lDom = layout.getLocalNDIndex();
            std::size_t fieldSize = lDom[0].length() * lDom[1].length() * lDom[2].length();

            if (tempField_.size() != fieldSize) {
                Kokkos::realloc(tempField_, lDom[0].length(), lDom[1].length(), lDom[2].length());
            }

            if (tempQ_.extent(0) < localNp) {
                Kokkos::realloc(tempQ_, localNp);
            }

            for (unsigned d = 0; d < Dim; ++d) {
                if (tempR_[d].extent(0) < localNp) {
                    Kokkos::realloc(tempR_[d], localNp);
                }
            }

            auto tempField = tempField_;
            auto tempQ     = tempQ_;
            auto tempRx    = tempR_[0];
            auto tempRy    = tempR_[1];
            auto tempRz    = tempR_[2];

            // Copy field data to FINUFFT temp buffer
            using mdrange_type = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "FINUFFT_copy_field_to_temp",
                mdrange_type(
                    {nghost, nghost, nghost},
                    {int(fview.extent(0)) - nghost,
                     int(fview.extent(1)) - nghost,
                     int(fview.extent(2)) - nghost}),
                KOKKOS_LAMBDA(int i, int j, int k) {
#ifdef ENABLE_GPU_NUFFT
                    tempField(i - nghost, j - nghost, k - nghost).x = fview(i, j, k).real();
                    tempField(i - nghost, j - nghost, k - nghost).y = fview(i, j, k).imag();
#else
                    tempField(i - nghost, j - nghost, k - nghost).real(fview(i, j, k).real());
                    tempField(i - nghost, j - nghost, k - nghost).imag(fview(i, j, k).imag());
#endif
                });

            // Copy particle data to FINUFFT temp buffers
            Kokkos::parallel_for(
                "FINUFFT_copy_particles_to_temp",
                localNp,
                KOKKOS_LAMBDA(std::size_t i) {
                    // Scale positions to [0, 2π)
                    tempRx(i) = Rview(i)[0] * (twoPi / Len[0]);
                    tempRy(i) = Rview(i)[1] * (twoPi / Len[1]);
                    tempRz(i) = Rview(i)[2] * (twoPi / Len[2]);

#ifdef ENABLE_GPU_NUFFT
                    tempQ(i).x = Qview(i);
                    tempQ(i).y = 0.0;
#else
                    tempQ(i).real(Qview(i));
                    tempQ(i).imag(0.0);
#endif
                });

            Kokkos::fence();

            // Set points
            finufftError_ = FinufftTypes::setpts()(
                finufftPlan_, localNp,
                tempRx.data(), tempRy.data(), tempRz.data(),
                0, nullptr, nullptr, nullptr);

            if (finufftError_ != 0) {
                throw IpplException("FFT<NUFFTransform>", "FINUFFT setpts failed");
            }

            // Execute transform
            finufftError_ = FinufftTypes::execute()(
                finufftPlan_, tempQ.data(), tempField.data());

            if (finufftError_ != 0) {
                throw IpplException("FFT<NUFFTransform>", "FINUFFT execute failed");
            }

            Kokkos::fence();

            // Copy results back
            if (type_ == 1) {
                // Type 1: Copy field data back
                Kokkos::parallel_for(
                    "FINUFFT_copy_field_from_temp",
                    mdrange_type(
                        {nghost, nghost, nghost},
                        {int(fview.extent(0)) - nghost,
                         int(fview.extent(1)) - nghost,
                         int(fview.extent(2)) - nghost}),
                    KOKKOS_LAMBDA(int i, int j, int k) {
#ifdef ENABLE_GPU_NUFFT
                        fview(i, j, k).real() = tempField(i - nghost, j - nghost, k - nghost).x;
                        fview(i, j, k).imag() = tempField(i - nghost, j - nghost, k - nghost).y;
#else
                        fview(i, j, k).real() = tempField(i - nghost, j - nghost, k - nghost).real();
                        fview(i, j, k).imag() = tempField(i - nghost, j - nghost, k - nghost).imag();
#endif
                    });

            } else if (type_ == 2) {
                // Type 2: Copy particle data back
                Kokkos::parallel_for(
                    "FINUFFT_copy_particles_from_temp",
                    localNp,
                    KOKKOS_LAMBDA(std::size_t i) {
#ifdef ENABLE_GPU_NUFFT
                        Qview(i) = tempQ(i).x;
#else
                        Qview(i) = tempQ(i).real();
#endif
                    });
            }
#else
            throw IpplException("FFT<NUFFTransform>",
                                "FINUFFT requested but not available (ENABLE_FINUFFT not set)");
#endif
        }
    };

}  // namespace ippl

#endif