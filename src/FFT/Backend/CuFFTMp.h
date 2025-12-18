#ifndef IPPL_FFT_BACKEND_CUFFTMP_HPP
#define IPPL_FFT_BACKEND_CUFFTMP_HPP

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"
#include "FFT/Traits.h"

#include <cufftMp.h>
#include <nvshmem.h>
#include <nvshmemx.h>
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

    // NVSHMEM initialization helper (call once at application startup)
    inline void ensureNvshmemInitialized(MPI_Comm comm) {
        static bool initialized = false;
        if (!initialized) {
            nvshmemx_init_attr_t attr;
            attr.mpi_comm = &comm;
            nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
            initialized = true;
        }
    }
}  // namespace detail

// CUDA scaling kernels
__global__ void scaleKernelFloat(cufftComplex* data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

__global__ void scaleKernelDouble(cufftDoubleComplex* data, size_t n, double scale) {
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

        // Ensure NVSHMEM is initialized
        detail::ensureNvshmemInitialized(comm);

        // Create CUDA stream
        checkCudaError(cudaStreamCreate(&stream_), "Failed to create CUDA stream");

        // Create cuFFT handle
        checkCufftResult(cufftCreate(&handle_), "Failed to create cuFFT handle");

        // Attach MPI communicator
        checkCufftResult(
            cufftMpAttachComm(handle_, CUFFT_COMM_MPI, &comm_),
            "Failed to attach MPI communicator"
        );

        // Set stream
        checkCufftResult(cufftSetStream(handle_, stream_), "Failed to set stream");

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
        for (int d = 0; d < 3; ++d) {
            local_in_size_[d]  = upper_in[d] - lower_in[d];
            local_out_size_[d] = upper_out[d] - lower_out[d];
        }

        // Row-major strides (C order): last dimension is contiguous
        strides_in[2] = 1;
        strides_in[1] = local_in_size_[2];
        strides_in[0] = local_in_size_[2] * local_in_size_[1];

        strides_out[2] = 1;
        strides_out[1] = local_out_size_[2];
        strides_out[0] = local_out_size_[2] * local_out_size_[1];

        // Compute global dimensions via MPI reduction
        std::array<long long, 3> local_max, global_size;
        for (int d = 0; d < 3; ++d) {
            local_max[d] = std::max(upper_in[d], upper_out[d]);
        }
        MPI_Allreduce(local_max.data(), global_size.data(), 3,
                      MPI_LONG_LONG, MPI_MAX, comm);

        int n[3] = {static_cast<int>(global_size[0]),
                    static_cast<int>(global_size[1]),
                    static_cast<int>(global_size[2])};

        // Store total elements for scaling
        total_elements_ = static_cast<size_t>(n[0]) * n[1] * n[2];
        local_in_elements_  = local_in_size_[0] * local_in_size_[1] * local_in_size_[2];
        local_out_elements_ = local_out_size_[0] * local_out_size_[1] * local_out_size_[2];

        // Create plan with custom decomposition
        // This implicitly sets CUFFT_XT_FORMAT_DISTRIBUTED_INPUT/OUTPUT
        checkCufftResult(
            cufftMpMakePlanDecomposition(
                handle_, 3, n,
                lower_in.data(), upper_in.data(), strides_in.data(),
                lower_out.data(), upper_out.data(), strides_out.data(),
                type, &comm_, CUFFT_COMM_MPI, &worksize_
            ),
            "Failed to create cuFFTMp decomposition plan"
        );

        // Allocate NVSHMEM buffers for internal use
        // cuFFTMp requires NVSHMEM-allocated memory for cufftExecC2C
        size_t in_bytes  = local_in_elements_ * sizeof(cuda_complex_t);
        size_t out_bytes = local_out_elements_ * sizeof(cuda_complex_t);

        nvshmem_buffer_in_  = static_cast<cuda_complex_t*>(nvshmem_malloc(in_bytes));
        nvshmem_buffer_out_ = static_cast<cuda_complex_t*>(nvshmem_malloc(out_bytes));

        if (!nvshmem_buffer_in_ || !nvshmem_buffer_out_) {
            throw IpplException("cuFFTMp", "Failed to allocate NVSHMEM memory");
        }
    }

    ~CuFFTMpC2C() {
        if (nvshmem_buffer_in_)  nvshmem_free(nvshmem_buffer_in_);
        if (nvshmem_buffer_out_) nvshmem_free(nvshmem_buffer_out_);
        if (handle_) cufftDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    // Disable copy operations
    CuFFTMpC2C(const CuFFTMpC2C&) = delete;
    CuFFTMpC2C& operator=(const CuFFTMpC2C&) = delete;

    // Move operations
    CuFFTMpC2C(CuFFTMpC2C&& other) noexcept
        : handle_(other.handle_)
        , comm_(other.comm_)
        , stream_(other.stream_)
        , nvshmem_buffer_in_(other.nvshmem_buffer_in_)
        , nvshmem_buffer_out_(other.nvshmem_buffer_out_)
        , worksize_(other.worksize_)
        , total_elements_(other.total_elements_)
        , local_in_elements_(other.local_in_elements_)
        , local_out_elements_(other.local_out_elements_)
        , local_in_size_(other.local_in_size_)
        , local_out_size_(other.local_out_size_)
    {
        other.handle_ = 0;
        other.stream_ = nullptr;
        other.nvshmem_buffer_in_ = nullptr;
        other.nvshmem_buffer_out_ = nullptr;
    }

    CuFFTMpC2C& operator=(CuFFTMpC2C&& other) noexcept {
        if (this != &other) {
            if (nvshmem_buffer_in_)  nvshmem_free(nvshmem_buffer_in_);
            if (nvshmem_buffer_out_) nvshmem_free(nvshmem_buffer_out_);
            if (handle_) cufftDestroy(handle_);
            if (stream_) cudaStreamDestroy(stream_);

            handle_ = other.handle_;
            comm_ = other.comm_;
            stream_ = other.stream_;
            nvshmem_buffer_in_ = other.nvshmem_buffer_in_;
            nvshmem_buffer_out_ = other.nvshmem_buffer_out_;
            worksize_ = other.worksize_;
            total_elements_ = other.total_elements_;
            local_in_elements_ = other.local_in_elements_;
            local_out_elements_ = other.local_out_elements_;
            local_in_size_ = other.local_in_size_;
            local_out_size_ = other.local_out_size_;

            other.handle_ = 0;
            other.stream_ = nullptr;
            other.nvshmem_buffer_in_ = nullptr;
            other.nvshmem_buffer_out_ = nullptr;
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
        using detail::checkCudaError;

        cuda_complex_t* cuda_in = reinterpret_cast<cuda_complex_t*>(in);
        cuda_complex_t* cuda_out = reinterpret_cast<cuda_complex_t*>(out);

        // Copy user data to NVSHMEM buffer
        checkCudaError(
            cudaMemcpyAsync(nvshmem_buffer_in_, cuda_in,
                           local_in_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy input to NVSHMEM buffer"
        );

        // Execute forward transform directly on NVSHMEM buffers
        // For custom decomposition plans, use cufftExecC2C with NVSHMEM pointers
        if constexpr (std::is_same_v<T, float>) {
            checkCufftResult(
                cufftExecC2C(handle_, nvshmem_buffer_in_, nvshmem_buffer_out_, CUFFT_FORWARD),
                "Forward FFT execution failed"
            );
        } else {
            checkCufftResult(
                cufftExecZ2Z(handle_,
                            reinterpret_cast<cufftDoubleComplex*>(nvshmem_buffer_in_),
                            reinterpret_cast<cufftDoubleComplex*>(nvshmem_buffer_out_),
                            CUFFT_FORWARD),
                "Forward FFT execution failed"
            );
        }

        // Copy result from NVSHMEM buffer to user output
        checkCudaError(
            cudaMemcpyAsync(cuda_out, nvshmem_buffer_out_,
                           local_out_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy output from NVSHMEM buffer"
        );

        // Apply full scaling (1/N) to match heFFTe::scale::full
        T scale = T(1) / static_cast<T>(total_elements_);
        applyScaling(cuda_out, local_out_elements_, scale);

        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

    /**
     * @brief Execute backward FFT without scaling
     *
     * Matches heFFTe convention: backward uses heffte::scale::none
     */
    void backward(complex_t* in, complex_t* out) {
        using detail::checkCufftResult;
        using detail::checkCudaError;

        cuda_complex_t* cuda_in = reinterpret_cast<cuda_complex_t*>(in);
        cuda_complex_t* cuda_out = reinterpret_cast<cuda_complex_t*>(out);

        // Copy user data to NVSHMEM buffer (use output buffer as input for inverse)
        checkCudaError(
            cudaMemcpyAsync(nvshmem_buffer_out_, cuda_in,
                           local_out_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy input to NVSHMEM buffer"
        );

        // Execute backward (inverse) transform
        if constexpr (std::is_same_v<T, float>) {
            checkCufftResult(
                cufftExecC2C(handle_, nvshmem_buffer_out_, nvshmem_buffer_in_, CUFFT_INVERSE),
                "Backward FFT execution failed"
            );
        } else {
            checkCufftResult(
                cufftExecZ2Z(handle_,
                            reinterpret_cast<cufftDoubleComplex*>(nvshmem_buffer_out_),
                            reinterpret_cast<cufftDoubleComplex*>(nvshmem_buffer_in_),
                            CUFFT_INVERSE),
                "Backward FFT execution failed"
            );
        }

        // Copy result from NVSHMEM buffer to user output
        checkCudaError(
            cudaMemcpyAsync(cuda_out, nvshmem_buffer_in_,
                           local_in_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy output from NVSHMEM buffer"
        );

        // No scaling for backward (matches heffte::scale::none)

        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

    std::size_t workspace_size() const { return worksize_; }

private:
    void applyScaling(cuda_complex_t* data, size_t count, T scale) {
        int blockSize = 256;
        int numBlocks = (count + blockSize - 1) / blockSize;

        if constexpr (std::is_same_v<T, float>) {
            scaleKernelFloat<<<numBlocks, blockSize, 0, stream_>>>(
                reinterpret_cast<cufftComplex*>(data), count, scale);
        } else {
            scaleKernelDouble<<<numBlocks, blockSize, 0, stream_>>>(
                reinterpret_cast<cufftDoubleComplex*>(data), count, scale);
        }
    }

    cufftHandle handle_ = 0;
    MPI_Comm comm_;
    cudaStream_t stream_ = nullptr;

    // NVSHMEM-allocated buffers (required by cuFFTMp)
    cuda_complex_t* nvshmem_buffer_in_ = nullptr;
    cuda_complex_t* nvshmem_buffer_out_ = nullptr;

    size_t worksize_ = 0;
    size_t total_elements_ = 0;
    size_t local_in_elements_ = 0;
    size_t local_out_elements_ = 0;
    std::array<long long, 3> local_in_size_;
    std::array<long long, 3> local_out_size_;
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

        // Ensure NVSHMEM is initialized
        detail::ensureNvshmemInitialized(comm);

        // Create CUDA stream
        checkCudaError(cudaStreamCreate(&stream_), "Failed to create CUDA stream");

        // Create cuFFT handle
        checkCufftResult(cufftCreate(&handle_), "Failed to create cuFFT handle");

        // Attach MPI communicator
        checkCufftResult(
            cufftMpAttachComm(handle_, CUFFT_COMM_MPI, &comm_),
            "Failed to attach MPI communicator"
        );

        checkCufftResult(cufftSetStream(handle_, stream_), "Failed to set stream");

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

        for (int d = 0; d < 3; ++d) {
            local_real_size_[d]    = upper_in[d] - lower_in[d];
            local_complex_size_[d] = upper_out[d] - lower_out[d];
        }

        // Row-major strides
        strides_in[2] = 1;
        strides_in[1] = local_real_size_[2];
        strides_in[0] = local_real_size_[2] * local_real_size_[1];

        strides_out[2] = 1;
        strides_out[1] = local_complex_size_[2];
        strides_out[0] = local_complex_size_[2] * local_complex_size_[1];

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
        local_real_elements_    = local_real_size_[0] * local_real_size_[1] * local_real_size_[2];
        local_complex_elements_ = local_complex_size_[0] * local_complex_size_[1] * local_complex_size_[2];

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

        // Allocate NVSHMEM buffers
        size_t real_bytes    = local_real_elements_ * sizeof(T);
        size_t complex_bytes = local_complex_elements_ * sizeof(cuda_complex_t);

        nvshmem_buffer_real_    = static_cast<T*>(nvshmem_malloc(real_bytes));
        nvshmem_buffer_complex_ = static_cast<cuda_complex_t*>(nvshmem_malloc(complex_bytes));

        if (!nvshmem_buffer_real_ || !nvshmem_buffer_complex_) {
            throw IpplException("cuFFTMp", "Failed to allocate NVSHMEM memory for R2C");
        }
    }

    ~CuFFTMpR2C() {
        if (nvshmem_buffer_real_)    nvshmem_free(nvshmem_buffer_real_);
        if (nvshmem_buffer_complex_) nvshmem_free(nvshmem_buffer_complex_);
        if (handle_) cufftDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    CuFFTMpR2C(const CuFFTMpR2C&) = delete;
    CuFFTMpR2C& operator=(const CuFFTMpR2C&) = delete;

    CuFFTMpR2C(CuFFTMpR2C&& other) noexcept
        : handle_(other.handle_)
        , comm_(other.comm_)
        , stream_(other.stream_)
        , nvshmem_buffer_real_(other.nvshmem_buffer_real_)
        , nvshmem_buffer_complex_(other.nvshmem_buffer_complex_)
        , worksize_(other.worksize_)
        , r2c_direction_(other.r2c_direction_)
        , total_elements_(other.total_elements_)
        , local_real_elements_(other.local_real_elements_)
        , local_complex_elements_(other.local_complex_elements_)
        , local_real_size_(other.local_real_size_)
        , local_complex_size_(other.local_complex_size_)
    {
        other.handle_ = 0;
        other.stream_ = nullptr;
        other.nvshmem_buffer_real_ = nullptr;
        other.nvshmem_buffer_complex_ = nullptr;
    }

    void forward(T* in, complex_t* out) {
        using detail::checkCufftResult;
        using detail::checkCudaError;

        cuda_complex_t* cuda_out = reinterpret_cast<cuda_complex_t*>(out);

        // Copy input to NVSHMEM buffer
        checkCudaError(
            cudaMemcpyAsync(nvshmem_buffer_real_, in,
                           local_real_elements_ * sizeof(T),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy real input to NVSHMEM buffer"
        );

        // Execute R2C transform
        if constexpr (std::is_same_v<T, float>) {
            checkCufftResult(
                cufftExecR2C(handle_, nvshmem_buffer_real_, nvshmem_buffer_complex_),
                "R2C forward FFT execution failed"
            );
        } else {
            checkCufftResult(
                cufftExecD2Z(handle_,
                            nvshmem_buffer_real_,
                            reinterpret_cast<cufftDoubleComplex*>(nvshmem_buffer_complex_)),
                "D2Z forward FFT execution failed"
            );
        }

        // Copy result to user output
        checkCudaError(
            cudaMemcpyAsync(cuda_out, nvshmem_buffer_complex_,
                           local_complex_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy complex output from NVSHMEM buffer"
        );

        // Apply full scaling
        T scale = T(1) / static_cast<T>(total_elements_);
        applyScaling(cuda_out, local_complex_elements_, scale);

        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

    void backward(complex_t* in, T* out) {
        using detail::checkCufftResult;
        using detail::checkCudaError;

        cuda_complex_t* cuda_in = reinterpret_cast<cuda_complex_t*>(in);

        // Copy input to NVSHMEM buffer
        checkCudaError(
            cudaMemcpyAsync(nvshmem_buffer_complex_, cuda_in,
                           local_complex_elements_ * sizeof(cuda_complex_t),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy complex input to NVSHMEM buffer"
        );

        // Execute C2R transform
        if constexpr (std::is_same_v<T, float>) {
            checkCufftResult(
                cufftExecC2R(handle_, nvshmem_buffer_complex_, nvshmem_buffer_real_),
                "C2R backward FFT execution failed"
            );
        } else {
            checkCufftResult(
                cufftExecZ2D(handle_,
                            reinterpret_cast<cufftDoubleComplex*>(nvshmem_buffer_complex_),
                            nvshmem_buffer_real_),
                "Z2D backward FFT execution failed"
            );
        }

        // Copy result to user output
        checkCudaError(
            cudaMemcpyAsync(out, nvshmem_buffer_real_,
                           local_real_elements_ * sizeof(T),
                           cudaMemcpyDeviceToDevice, stream_),
            "Failed to copy real output from NVSHMEM buffer"
        );

        // No scaling for backward

        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

    std::size_t workspace_size() const { return worksize_; }

private:
    void applyScaling(cuda_complex_t* data, size_t count, T scale) {
        int blockSize = 256;
        int numBlocks = (count + blockSize - 1) / blockSize;

        if constexpr (std::is_same_v<T, float>) {
            scaleKernelFloat<<<numBlocks, blockSize, 0, stream_>>>(
                reinterpret_cast<cufftComplex*>(data), count, scale);
        } else {
            scaleKernelDouble<<<numBlocks, blockSize, 0, stream_>>>(
                reinterpret_cast<cufftDoubleComplex*>(data), count, scale);
        }
    }

    cufftHandle handle_ = 0;
    MPI_Comm comm_;
    cudaStream_t stream_ = nullptr;

    T* nvshmem_buffer_real_ = nullptr;
    cuda_complex_t* nvshmem_buffer_complex_ = nullptr;

    size_t worksize_ = 0;
    int r2c_direction_;
    size_t total_elements_ = 0;
    size_t local_real_elements_ = 0;
    size_t local_complex_elements_ = 0;
    std::array<long long, 3> local_real_size_;
    std::array<long long, 3> local_complex_size_;
};

}  // namespace fft
}  // namespace ippl

#endif  // IPPL_FFT_BACKEND_CUFFTMP_HPP