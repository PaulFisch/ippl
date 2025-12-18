#ifndef IPPL_FFT_BACKEND_CUFFTMP_HPP
#define IPPL_FFT_BACKEND_CUFFTMP_HPP

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"
#include "FFT/Traits.h"

#include <cufftMp.h>
#include <mpi.h>
#include <array>
#include <type_traits>

namespace ippl {
namespace fft {

namespace detail {
    inline void checkCufftResult(cufftResult result, const char* context) {
        if (result != CUFFT_SUCCESS) {
            std::string msg = std::string(context) + " (error code: " + std::to_string(result) + ")";
            throw IpplException("cuFFTMp", msg.c_str());
        }
    }

    inline void checkCudaError(cudaError_t err, const char* context) {
        if (err != cudaSuccess) {
            std::string msg = std::string(context) + ": " + cudaGetErrorString(err);
            throw IpplException("cuFFTMp", msg.c_str());
        }
    }
}  // namespace detail

//=============================================================================
// cuFFTMp C2C Backend
//=============================================================================

    // CUDA scaling kernels (defined outside class or as static device functions)
    static __global__ void scaleKernelFloat(cufftComplex* data, size_t n, float scale) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx].x *= scale;
            data[idx].y *= scale;
        }
    }

    static __global__ void scaleKernelDouble(cufftDoubleComplex* data, size_t n, double scale) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx].x *= scale;
            data[idx].y *= scale;
        }
    }

template <typename T, unsigned Dim, typename MemSpace>
class CuFFTMpC2C {
public:
    using complex_t = Kokkos::complex<T>;
    using cuda_complex_t = std::conditional_t<std::is_same_v<T, float>,
                                               cufftComplex, cufftDoubleComplex>;

    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "cuFFTMp only supports float and double precision");
    static_assert(Dim == 3, "cuFFTMp backend currently only supports 3D transforms");
    static_assert(is_available_v<CuFFTMp>, "cuFFTMp not available");

    CuFFTMpC2C(const heffte::box3d<long long>& inbox,
               const heffte::box3d<long long>& outbox,
               MPI_Comm comm,
               const ParameterList& params)
        : comm_(comm)
    {
        using detail::checkCufftResult;
        using detail::checkCudaError;

        // Create cuFFT handle
        checkCufftResult(cufftCreate(&handle_), "Failed to create cuFFT handle");

        // Determine transform type based on precision
        cufftType type = std::is_same_v<T, float> ? CUFFT_C2C : CUFFT_Z2Z;

        // Convert heffte box3d to cuFFTMp decomposition format
        // heffte boxes are inclusive on both ends; cuFFTMp uses exclusive upper bound
        std::array<long long, 3> lower_in, upper_in, strides_in;
        std::array<long long, 3> lower_out, upper_out, strides_out;

        for (int d = 0; d < 3; ++d) {
            lower_in[d]  = inbox.low[d];
            upper_in[d]  = inbox.high[d] + 1;   // exclusive
            lower_out[d] = outbox.low[d];
            upper_out[d] = outbox.high[d] + 1;  // exclusive
        }

        // Compute local sizes
        std::array<long long, 3> local_in_size, local_out_size;
        for (int d = 0; d < 3; ++d) {
            local_in_size[d]  = upper_in[d] - lower_in[d];
            local_out_size[d] = upper_out[d] - lower_out[d];
        }

        // Row-major strides: dim 0 is slowest (largest stride), dim 2 is fastest (stride=1)
        // cuFFTMp requires strides to be decreasing and positive
        strides_in[2] = 1;
        strides_in[1] = local_in_size[2];
        strides_in[0] = local_in_size[2] * local_in_size[1];

        strides_out[2] = 1;
        strides_out[1] = local_out_size[2];
        strides_out[0] = local_out_size[2] * local_out_size[1];

        // Compute global dimensions via MPI reduction
        std::array<long long, 3> local_max, global_size;
        for (int d = 0; d < 3; ++d) {
            // The global size is the maximum upper bound across all processes
            local_max[d] = std::max(upper_in[d], upper_out[d]);
        }
        MPI_Allreduce(local_max.data(), global_size.data(), 3,
                      MPI_LONG_LONG, MPI_MAX, comm);

        int n[3] = {static_cast<int>(global_size[0]),
                    static_cast<int>(global_size[1]),
                    static_cast<int>(global_size[2])};

        // Store total elements for scaling
        total_elements_ = static_cast<size_t>(n[0]) * n[1] * n[2];
        local_in_elements_  = local_in_size[0] * local_in_size[1] * local_in_size[2];
        local_out_elements_ = local_out_size[0] * local_out_size[1] * local_out_size[2];

        // Create plan with custom decomposition
        checkCufftResult(
            cufftMpMakePlanDecomposition(
                handle_, 3, n,
                lower_in.data(), upper_in.data(), strides_in.data(),
                lower_out.data(), upper_out.data(), strides_out.data(),
                type, &comm_, CUFFT_COMM_MPI, &worksize_
            ),
            "Failed to create cuFFTMp decomposition plan"
        );

        // Allocate internal descriptors for data transfer
        checkCufftResult(
            cufftXtMalloc(handle_, &desc_in_, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT),
            "Failed to allocate input descriptor"
        );
        checkCufftResult(
            cufftXtMalloc(handle_, &desc_out_, CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT),
            "Failed to allocate output descriptor"
        );
    }

    ~CuFFTMpC2C() {
        if (desc_in_)  cufftXtFree(desc_in_);
        if (desc_out_) cufftXtFree(desc_out_);
        if (handle_)   cufftDestroy(handle_);
    }

    // Disable copy operations
    CuFFTMpC2C(const CuFFTMpC2C&) = delete;
    CuFFTMpC2C& operator=(const CuFFTMpC2C&) = delete;

    // Move operations
    CuFFTMpC2C(CuFFTMpC2C&& other) noexcept
        : handle_(other.handle_)
        , comm_(other.comm_)
        , desc_in_(other.desc_in_)
        , desc_out_(other.desc_out_)
        , worksize_(other.worksize_)
        , total_elements_(other.total_elements_)
        , local_in_elements_(other.local_in_elements_)
        , local_out_elements_(other.local_out_elements_)
    {
        other.handle_   = 0;
        other.desc_in_  = nullptr;
        other.desc_out_ = nullptr;
    }

    CuFFTMpC2C& operator=(CuFFTMpC2C&& other) noexcept {
        if (this != &other) {
            if (desc_in_)  cufftXtFree(desc_in_);
            if (desc_out_) cufftXtFree(desc_out_);
            if (handle_)   cufftDestroy(handle_);

            handle_            = other.handle_;
            comm_              = other.comm_;
            desc_in_           = other.desc_in_;
            desc_out_          = other.desc_out_;
            worksize_          = other.worksize_;
            total_elements_    = other.total_elements_;
            local_in_elements_ = other.local_in_elements_;
            local_out_elements_= other.local_out_elements_;

            other.handle_   = 0;
            other.desc_in_  = nullptr;
            other.desc_out_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Execute forward FFT with full scaling (1/N normalization)
     *
     * Matches heFFTe convention: forward uses heffte::scale::full
     */
    void forward(complex_t* in, complex_t* out) {
        using detail::checkCufftResult;

        // Copy input data to internal descriptor
        checkCufftResult(
            cufftXtMemcpy(handle_, desc_in_, reinterpret_cast<void*>(in),
                          CUFFT_COPY_DEVICE_TO_DEVICE),
            "Failed to copy input data for forward FFT"
        );

        // Execute forward transform
        checkCufftResult(
            cufftXtExecDescriptor(handle_, desc_in_, desc_out_, CUFFT_FORWARD),
            "Forward FFT execution failed"
        );

        // Copy result to output buffer
        checkCufftResult(
            cufftXtMemcpy(handle_, reinterpret_cast<void*>(out), desc_out_,
                          CUFFT_COPY_DEVICE_TO_DEVICE),
            "Failed to copy output data from forward FFT"
        );

        // Apply full scaling (1/N) to match heFFTe::scale::full
        T scale = T(1) / static_cast<T>(total_elements_);
        applyScaling(out, local_out_elements_, scale);
    }

    /**
     * @brief Execute backward FFT without scaling
     *
     * Matches heFFTe convention: backward uses heffte::scale::none
     */
    void backward(complex_t* in, complex_t* out) {
        using detail::checkCufftResult;

        // Copy input data to output descriptor (backward starts from output space)
        checkCufftResult(
            cufftXtMemcpy(handle_, desc_out_, reinterpret_cast<void*>(in),
                          CUFFT_COPY_DEVICE_TO_DEVICE),
            "Failed to copy input data for backward FFT"
        );

        // Execute backward (inverse) transform
        checkCufftResult(
            cufftXtExecDescriptor(handle_, desc_out_, desc_in_, CUFFT_INVERSE),
            "Backward FFT execution failed"
        );

        // Copy result to output buffer
        checkCufftResult(
            cufftXtMemcpy(handle_, reinterpret_cast<void*>(out), desc_in_,
                          CUFFT_COPY_DEVICE_TO_DEVICE),
            "Failed to copy output data from backward FFT"
        );

        // No scaling for backward (matches heffte::scale::none)
    }

    std::size_t workspace_size() const { return worksize_; }

private:
    /**
     * @brief Apply scaling factor to complex array on GPU
     */
    void applyScaling(complex_t* data, size_t count, T scale) {
        // Use Kokkos or CUDA kernel for scaling
        // Simple CUDA implementation:
        cuda_complex_t* cuda_data = reinterpret_cast<cuda_complex_t*>(data);

        // Launch scaling kernel
        int blockSize = 256;
        int numBlocks = (count + blockSize - 1) / blockSize;

        if constexpr (std::is_same_v<T, float>) {
            scaleKernelFloat<<<numBlocks, blockSize>>>(cuda_data, count, scale);
        } else {
            scaleKernelDouble<<<numBlocks, blockSize>>>(cuda_data, count, scale);
        }
        cudaDeviceSynchronize();
    }

    cufftHandle handle_ = 0;
    MPI_Comm comm_;
    cudaLibXtDesc* desc_in_  = nullptr;
    cudaLibXtDesc* desc_out_ = nullptr;
    size_t worksize_ = 0;
    size_t total_elements_ = 0;
    size_t local_in_elements_ = 0;
    size_t local_out_elements_ = 0;
};

//=============================================================================
// cuFFTMp R2C Backend
//=============================================================================

template <typename T, unsigned Dim, typename MemSpace>
class CuFFTMpR2C {
public:
    using complex_t = Kokkos::complex<T>;
    using cuda_complex_t = std::conditional_t<std::is_same_v<T, float>,
                                               cufftComplex, cufftDoubleComplex>;

    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "cuFFTMp only supports float and double precision");
    static_assert(Dim == 3, "cuFFTMp backend currently only supports 3D transforms");
    static_assert(is_available_v<CuFFTMp>, "cuFFTMp not available");

    CuFFTMpR2C(const heffte::box3d<long long>& inbox,
               const heffte::box3d<long long>& outbox,
               int r2c_direction,
               MPI_Comm comm,
               const ParameterList& params)
        : comm_(comm), r2c_direction_(r2c_direction)
    {
        using detail::checkCufftResult;

        checkCufftResult(cufftCreate(&handle_), "Failed to create cuFFT handle");

        // R2C/D2Z type based on precision
        cufftType fwd_type = std::is_same_v<T, float> ? CUFFT_R2C : CUFFT_D2Z;

        // Convert boxes to decomposition format
        std::array<long long, 3> lower_in, upper_in, strides_in;
        std::array<long long, 3> lower_out, upper_out, strides_out;

        for (int d = 0; d < 3; ++d) {
            lower_in[d]  = inbox.low[d];
            upper_in[d]  = inbox.high[d] + 1;
            lower_out[d] = outbox.low[d];
            upper_out[d] = outbox.high[d] + 1;
        }

        std::array<long long, 3> local_in_size, local_out_size;
        for (int d = 0; d < 3; ++d) {
            local_in_size[d]  = upper_in[d] - lower_in[d];
            local_out_size[d] = upper_out[d] - lower_out[d];
        }

        // Row-major strides
        strides_in[2] = 1;
        strides_in[1] = local_in_size[2];
        strides_in[0] = local_in_size[2] * local_in_size[1];

        strides_out[2] = 1;
        strides_out[1] = local_out_size[2];
        strides_out[0] = local_out_size[2] * local_out_size[1];

        // Compute global dimensions
        std::array<long long, 3> local_max, global_size;
        for (int d = 0; d < 3; ++d) {
            local_max[d] = std::max(upper_in[d], upper_out[d]);
        }
        MPI_Allreduce(local_max.data(), global_size.data(), 3,
                      MPI_LONG_LONG, MPI_MAX, comm);

        int n[3] = {static_cast<int>(global_size[0]),
                    static_cast<int>(global_size[1]),
                    static_cast<int>(global_size[2])};

        total_elements_ = static_cast<size_t>(n[0]) * n[1] * n[2];
        local_real_elements_    = local_in_size[0] * local_in_size[1] * local_in_size[2];
        local_complex_elements_ = local_out_size[0] * local_out_size[1] * local_out_size[2];

        // Create R2C plan with decomposition
        checkCufftResult(
            cufftMpMakePlanDecomposition(
                handle_, 3, n,
                lower_in.data(), upper_in.data(), strides_in.data(),
                lower_out.data(), upper_out.data(), strides_out.data(),
                fwd_type, &comm_, CUFFT_COMM_MPI, &worksize_
            ),
            "Failed to create cuFFTMp R2C decomposition plan"
        );

        // Allocate descriptors
        checkCufftResult(
            cufftXtMalloc(handle_, &desc_real_, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT),
            "Failed to allocate real descriptor"
        );
        checkCufftResult(
            cufftXtMalloc(handle_, &desc_complex_, CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT),
            "Failed to allocate complex descriptor"
        );
    }

    ~CuFFTMpR2C() {
        if (desc_real_)    cufftXtFree(desc_real_);
        if (desc_complex_) cufftXtFree(desc_complex_);
        if (handle_)       cufftDestroy(handle_);
    }

    CuFFTMpR2C(const CuFFTMpR2C&) = delete;
    CuFFTMpR2C& operator=(const CuFFTMpR2C&) = delete;

    CuFFTMpR2C(CuFFTMpR2C&& other) noexcept
        : handle_(other.handle_)
        , comm_(other.comm_)
        , desc_real_(other.desc_real_)
        , desc_complex_(other.desc_complex_)
        , worksize_(other.worksize_)
        , r2c_direction_(other.r2c_direction_)
        , total_elements_(other.total_elements_)
        , local_real_elements_(other.local_real_elements_)
        , local_complex_elements_(other.local_complex_elements_)
    {
        other.handle_       = 0;
        other.desc_real_    = nullptr;
        other.desc_complex_ = nullptr;
    }

    /**
     * @brief Execute forward R2C FFT with full scaling
     */
    void forward(T* in, complex_t* out) {
        using detail::checkCufftResult;

        checkCufftResult(
            cufftXtMemcpy(handle_, desc_real_, reinterpret_cast<void*>(in),
                          CUFFT_COPY_DEVICE_TO_DEVICE),
            "Failed to copy real input for R2C"
        );

        checkCufftResult(
            cufftXtExecDescriptor(handle_, desc_real_, desc_complex_, CUFFT_FORWARD),
            "R2C forward FFT execution failed"
        );

        checkCufftResult(
            cufftXtMemcpy(handle_, reinterpret_cast<void*>(out), desc_complex_,
                          CUFFT_COPY_DEVICE_TO_DEVICE),
            "Failed to copy complex output from R2C"
        );

        // Apply full scaling
        T scale = T(1) / static_cast<T>(total_elements_);
        applyScaling(out, local_complex_elements_, scale);
    }

    /**
     * @brief Execute backward C2R FFT without scaling
     */
    void backward(complex_t* in, T* out) {
        using detail::checkCufftResult;

        checkCufftResult(
            cufftXtMemcpy(handle_, desc_complex_, reinterpret_cast<void*>(in),
                          CUFFT_COPY_DEVICE_TO_DEVICE),
            "Failed to copy complex input for C2R"
        );

        checkCufftResult(
            cufftXtExecDescriptor(handle_, desc_complex_, desc_real_, CUFFT_INVERSE),
            "C2R backward FFT execution failed"
        );

        checkCufftResult(
            cufftXtMemcpy(handle_, reinterpret_cast<void*>(out), desc_real_,
                          CUFFT_COPY_DEVICE_TO_DEVICE),
            "Failed to copy real output from C2R"
        );
    }

    std::size_t workspace_size() const { return worksize_; }

private:
    void applyScaling(complex_t* data, size_t count, T scale) {
        cuda_complex_t* cuda_data = reinterpret_cast<cuda_complex_t*>(data);
        int blockSize = 256;
        int numBlocks = (count + blockSize - 1) / blockSize;

        if constexpr (std::is_same_v<T, float>) {
            CuFFTMpC2C<T, Dim, MemSpace>::scaleKernelFloat<<<numBlocks, blockSize>>>(
                cuda_data, count, scale);
        } else {
            CuFFTMpC2C<T, Dim, MemSpace>::scaleKernelDouble<<<numBlocks, blockSize>>>(
                cuda_data, count, scale);
        }
        cudaDeviceSynchronize();
    }

    cufftHandle handle_ = 0;
    MPI_Comm comm_;
    cudaLibXtDesc* desc_real_    = nullptr;
    cudaLibXtDesc* desc_complex_ = nullptr;
    size_t worksize_ = 0;
    int r2c_direction_;
    size_t total_elements_ = 0;
    size_t local_real_elements_ = 0;
    size_t local_complex_elements_ = 0;
};

}  // namespace fft
}  // namespace ippl

#endif  // IPPL_FFT_BACKEND_CUFFTMP_HPP