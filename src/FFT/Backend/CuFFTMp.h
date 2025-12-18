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

// CUDA scaling kernel
template <typename T>
__global__ void scaleKernel(T* data, size_t n, double scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

//=============================================================================
// cuFFTMp C2C Backend
//=============================================================================

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

        // Create CUDA stream
        checkCudaError(cudaStreamCreate(&stream_), "Failed to create CUDA stream");

        // Create cuFFT handle
        checkCufftResult(cufftCreate(&handle_), "Failed to create cuFFT handle");
        checkCufftResult(cufftSetStream(handle_, stream_), "Failed to set stream");

        // Determine transform type
        cufftType type = std::is_same_v<T, float> ? CUFFT_C2C : CUFFT_Z2Z;

        // Convert heffte box3d to cuFFTMp format (exclusive upper bounds)
        for (int d = 0; d < 3; ++d) {
            lower_in_[d]  = inbox.low[d];
            upper_in_[d]  = inbox.high[d] + 1;
            lower_out_[d] = outbox.low[d];
            upper_out_[d] = outbox.high[d] + 1;
        }

        // Compute local sizes
        std::array<long long, 3> local_in_size, local_out_size;
        for (int d = 0; d < 3; ++d) {
            local_in_size[d]  = upper_in_[d] - lower_in_[d];
            local_out_size[d] = upper_out_[d] - lower_out_[d];
        }

        // Row-major strides (C order)
        std::array<long long, 3> strides_in, strides_out;
        strides_in[2] = 1;
        strides_in[1] = local_in_size[2];
        strides_in[0] = local_in_size[2] * local_in_size[1];

        strides_out[2] = 1;
        strides_out[1] = local_out_size[2];
        strides_out[0] = local_out_size[2] * local_out_size[1];

        // Compute global dimensions
        std::array<long long, 3> local_max;
        for (int d = 0; d < 3; ++d) {
            local_max[d] = std::max(upper_in_[d], upper_out_[d]);
        }
        MPI_Allreduce(local_max.data(), global_size_.data(), 3, MPI_LONG_LONG, MPI_MAX, comm);

        int n[3] = {static_cast<int>(global_size_[0]),
                    static_cast<int>(global_size_[1]),
                    static_cast<int>(global_size_[2])};

        total_elements_ = static_cast<size_t>(n[0]) * n[1] * n[2];
        local_elements_ = local_in_size[0] * local_in_size[1] * local_in_size[2];

        // Create plan with custom decomposition
        checkCufftResult(
            cufftMpMakePlanDecomposition(
                handle_, 3, n,
                lower_in_.data(), upper_in_.data(), strides_in.data(),
                lower_out_.data(), upper_out_.data(), strides_out.data(),
                type, &comm_, CUFFT_COMM_MPI, &worksize_
            ),
            "Failed to create cuFFTMp decomposition plan"
        );

        // Allocate descriptor for input distribution
        checkCufftResult(
            cufftXtMalloc(handle_, &desc_, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT),
            "Failed to allocate descriptor"
        );
    }

    ~CuFFTMpC2C() {
        if (desc_)   cufftXtFree(desc_);
        if (handle_) cufftDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    CuFFTMpC2C(const CuFFTMpC2C&) = delete;
    CuFFTMpC2C& operator=(const CuFFTMpC2C&) = delete;

    CuFFTMpC2C(CuFFTMpC2C&& other) noexcept
        : handle_(other.handle_)
        , comm_(other.comm_)
        , stream_(other.stream_)
        , desc_(other.desc_)
        , worksize_(other.worksize_)
        , total_elements_(other.total_elements_)
        , local_elements_(other.local_elements_)
        , global_size_(other.global_size_)
        , lower_in_(other.lower_in_)
        , upper_in_(other.upper_in_)
        , lower_out_(other.lower_out_)
        , upper_out_(other.upper_out_)
    {
        other.handle_ = 0;
        other.stream_ = nullptr;
        other.desc_ = nullptr;
    }

    /**
     * @brief Execute forward FFT with full scaling (1/N normalization)
     */
    void forward(complex_t* in, complex_t* out) {
        using detail::checkCufftResult;
        using detail::checkCudaError;

        cuda_complex_t* desc_data = static_cast<cuda_complex_t*>(desc_->descriptor->data[0]);

        // Copy input data to descriptor's internal buffer
        checkCudaError(
            cudaMemcpyAsync(desc_data, in,
                           local_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy input to descriptor"
        );

        // Execute forward transform in-place on descriptor
        checkCufftResult(
            cufftXtExecDescriptor(handle_, desc_, desc_, CUFFT_FORWARD),
            "Forward FFT execution failed"
        );

        // Copy result from descriptor to output
        // After forward, data is in DISTRIBUTED_OUTPUT format
        checkCudaError(
            cudaMemcpyAsync(out, desc_data,
                           local_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy output from descriptor"
        );

        // Apply full scaling (1/N) to match heFFTe::scale::full
        T scale = T(1) / static_cast<T>(total_elements_);
        applyScaling(reinterpret_cast<cuda_complex_t*>(out), local_elements_, scale);

        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

    /**
     * @brief Execute backward FFT without scaling
     */
    void backward(complex_t* in, complex_t* out) {
        using detail::checkCufftResult;
        using detail::checkCudaError;

        cuda_complex_t* desc_data = static_cast<cuda_complex_t*>(desc_->descriptor->data[0]);

        // Copy input to descriptor
        checkCudaError(
            cudaMemcpyAsync(desc_data, in,
                           local_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy input to descriptor"
        );

        // Execute backward transform
        checkCufftResult(
            cufftXtExecDescriptor(handle_, desc_, desc_, CUFFT_INVERSE),
            "Backward FFT execution failed"
        );

        // Copy result to output
        checkCudaError(
            cudaMemcpyAsync(out, desc_data,
                           local_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy output from descriptor"
        );

        // No scaling for backward (matches heffte::scale::none)
        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

    std::size_t workspace_size() const { return worksize_; }

private:
    void applyScaling(cuda_complex_t* data, size_t count, T scale) {
        int blockSize = 256;
        int numBlocks = (count + blockSize - 1) / blockSize;
        scaleKernel<<<numBlocks, blockSize, 0, stream_>>>(data, count, scale);
    }

    cufftHandle handle_ = 0;
    MPI_Comm comm_;
    cudaStream_t stream_ = nullptr;
    cudaLibXtDesc* desc_ = nullptr;

    size_t worksize_ = 0;
    size_t total_elements_ = 0;
    size_t local_elements_ = 0;
    std::array<long long, 3> global_size_;
    std::array<long long, 3> lower_in_, upper_in_;
    std::array<long long, 3> lower_out_, upper_out_;
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
        using detail::checkCudaError;

        checkCudaError(cudaStreamCreate(&stream_), "Failed to create CUDA stream");
        checkCufftResult(cufftCreate(&handle_r2c_), "Failed to create R2C handle");
        checkCufftResult(cufftCreate(&handle_c2r_), "Failed to create C2R handle");
        checkCufftResult(cufftSetStream(handle_r2c_, stream_), "Failed to set stream");
        checkCufftResult(cufftSetStream(handle_c2r_, stream_), "Failed to set stream");

        // Convert boxes (inbox = real, outbox = complex for R2C)
        std::array<long long, 3> lower_real, upper_real, strides_real;
        std::array<long long, 3> lower_complex, upper_complex, strides_complex;

        for (int d = 0; d < 3; ++d) {
            lower_real[d]    = inbox.low[d];
            upper_real[d]    = inbox.high[d] + 1;
            lower_complex[d] = outbox.low[d];
            upper_complex[d] = outbox.high[d] + 1;
        }

        std::array<long long, 3> local_real_size, local_complex_size;
        for (int d = 0; d < 3; ++d) {
            local_real_size[d]    = upper_real[d] - lower_real[d];
            local_complex_size[d] = upper_complex[d] - lower_complex[d];
        }

        // For in-place R2C, real data needs padding: last dimension stride = 2*(nz/2+1)
        long long nz_complex = local_complex_size[2];
        long long nz_real_padded = 2 * nz_complex;

        strides_real[2] = 1;
        strides_real[1] = nz_real_padded;  // Padded for in-place
        strides_real[0] = local_real_size[1] * nz_real_padded;

        strides_complex[2] = 1;
        strides_complex[1] = local_complex_size[2];
        strides_complex[0] = local_complex_size[1] * local_complex_size[2];

        // Global dimensions
        std::array<long long, 3> local_max;
        for (int d = 0; d < 3; ++d) {
            local_max[d] = std::max(upper_real[d], upper_complex[d]);
        }
        MPI_Allreduce(local_max.data(), global_size_.data(), 3, MPI_LONG_LONG, MPI_MAX, comm);

        int n[3] = {static_cast<int>(global_size_[0]),
                    static_cast<int>(global_size_[1]),
                    static_cast<int>(global_size_[2])};

        total_elements_ = static_cast<size_t>(n[0]) * n[1] * n[2];
        local_real_elements_ = local_real_size[0] * local_real_size[1] * local_real_size[2];
        local_complex_elements_ = local_complex_size[0] * local_complex_size[1] * local_complex_size[2];

        // R2C plan: input=real box, output=complex box
        checkCufftResult(
            cufftMpMakePlanDecomposition(
                handle_r2c_, 3, n,
                lower_real.data(), upper_real.data(), strides_real.data(),
                lower_complex.data(), upper_complex.data(), strides_complex.data(),
                CUFFT_R2C, &comm_, CUFFT_COMM_MPI, &worksize_
            ),
            "Failed to create R2C plan"
        );

        // C2R plan: same boxes (input=real, output=complex in cuFFTMp convention)
        checkCufftResult(
            cufftMpMakePlanDecomposition(
                handle_c2r_, 3, n,
                lower_real.data(), upper_real.data(), strides_real.data(),
                lower_complex.data(), upper_complex.data(), strides_complex.data(),
                CUFFT_C2R, &comm_, CUFFT_COMM_MPI, &worksize_
            ),
            "Failed to create C2R plan"
        );

        // Allocate descriptor
        checkCufftResult(
            cufftXtMalloc(handle_r2c_, &desc_, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT),
            "Failed to allocate R2C descriptor"
        );
    }

    ~CuFFTMpR2C() {
        if (desc_)       cufftXtFree(desc_);
        if (handle_r2c_) cufftDestroy(handle_r2c_);
        if (handle_c2r_) cufftDestroy(handle_c2r_);
        if (stream_)     cudaStreamDestroy(stream_);
    }

    CuFFTMpR2C(const CuFFTMpR2C&) = delete;
    CuFFTMpR2C& operator=(const CuFFTMpR2C&) = delete;

    void forward(T* in, complex_t* out) {
        using detail::checkCufftResult;
        using detail::checkCudaError;

        T* desc_data = static_cast<T*>(desc_->descriptor->data[0]);

        // Copy real input to descriptor
        checkCudaError(
            cudaMemcpyAsync(desc_data, in,
                           local_real_elements_ * sizeof(T),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy real input"
        );

        // Execute R2C (always CUFFT_FORWARD)
        checkCufftResult(
            cufftXtExecDescriptor(handle_r2c_, desc_, desc_, CUFFT_FORWARD),
            "R2C execution failed"
        );

        // Copy complex output
        checkCudaError(
            cudaMemcpyAsync(out, desc_->descriptor->data[0],
                           local_complex_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy complex output"
        );

        // Apply scaling
        T scale = T(1) / static_cast<T>(total_elements_);
        applyScaling(reinterpret_cast<cuda_complex_t*>(out), local_complex_elements_, scale);

        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

    void backward(complex_t* in, T* out) {
        using detail::checkCufftResult;
        using detail::checkCudaError;

        // Copy complex input to descriptor
        checkCudaError(
            cudaMemcpyAsync(desc_->descriptor->data[0], in,
                           local_complex_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy complex input"
        );

        // Execute C2R (always CUFFT_INVERSE)
        checkCufftResult(
            cufftXtExecDescriptor(handle_c2r_, desc_, desc_, CUFFT_INVERSE),
            "C2R execution failed"
        );

        // Copy real output
        checkCudaError(
            cudaMemcpyAsync(out, desc_->descriptor->data[0],
                           local_real_elements_ * sizeof(T),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy real output"
        );

        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

    std::size_t workspace_size() const { return worksize_; }

private:
    void applyScaling(cuda_complex_t* data, size_t count, T scale) {
        int blockSize = 256;
        int numBlocks = (count + blockSize - 1) / blockSize;
        scaleKernel<<<numBlocks, blockSize, 0, stream_>>>(data, count, scale);
    }

    cufftHandle handle_r2c_ = 0;
    cufftHandle handle_c2r_ = 0;
    MPI_Comm comm_;
    cudaStream_t stream_ = nullptr;
    cudaLibXtDesc* desc_ = nullptr;

    size_t worksize_ = 0;
    int r2c_direction_;
    size_t total_elements_ = 0;
    size_t local_real_elements_ = 0;
    size_t local_complex_elements_ = 0;
    std::array<long long, 3> global_size_;
};

}  // namespace fft
}  // namespace ippl

#endif