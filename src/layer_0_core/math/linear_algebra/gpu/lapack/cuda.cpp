/**
 * CudaGPUEigensolver — GPUEigensolver implementation using NVIDIA cuSOLVER.
 *
 * Provides Hermitian eigenvalue solvers via cusolverDnZheevd (standard)
 * and cusolverDnZhegvd (generalized). cuSOLVER operates on device memory;
 * with unified memory (cudaMallocManaged) the same pointers work seamlessly.
 *
 * A cuSOLVER handle is created on construction and destroyed on destruction.
 * Workspace is allocated per-call using cudaMalloc (device-only scratch space).
 */

#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace q3 {
namespace execution {
namespace gpu {
namespace lapack {

namespace {

/**
 * Check a cuSOLVER return code and throw on error.
 */
void check_cusolver_error(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("CudaGPUEigensolver: ") + operation +
                                 " failed with status " + std::to_string(static_cast<int>(status)));
    }
}

/**
 * Check a CUDA runtime return code and throw on error.
 */
void check_cuda_error(cudaError_t error_code, const char* operation) {
    if (error_code != cudaSuccess) {
        throw std::runtime_error(std::string("CudaGPUEigensolver: ") + operation +
                                 " failed: " + cudaGetErrorString(error_code));
    }
}

} // anonymous namespace

class CudaGPUEigensolver : public GPUEigensolver {
  public:
    /**
     * Construct a cuSOLVER eigensolver, creating a cuSOLVER dense handle.
     */
    explicit CudaGPUEigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : GPUEigensolver(std::move(device_backend)) {
        check_cusolver_error(cusolverDnCreate(&cusolver_handle_), "cusolverDnCreate");
    }

    ~CudaGPUEigensolver() override {
        if (cusolver_handle_ != nullptr) {
            cusolverDnDestroy(cusolver_handle_);
        }
    }

    /**
     * Solve A · x = λ · x via cusolverDnZheevd.
     *
     * Uses the divide-and-conquer algorithm for better performance on
     * large matrices. On entry, matrix_data holds the Hermitian matrix
     * (upper triangle). On exit, it holds the eigenvectors.
     */
    void hermitian_eigenvalue_solve(int matrix_dimension,
                                    Complex* matrix_data,
                                    int leading_dimension,
                                    Real* eigenvalues) override {
        cusolverEigMode_t job_mode = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

        auto* device_matrix = reinterpret_cast<cuDoubleComplex*>(matrix_data);

        // Workspace query.
        int workspace_size = 0;
        check_cusolver_error(cusolverDnZheevd_bufferSize(cusolver_handle_,
                                                         job_mode,
                                                         uplo,
                                                         matrix_dimension,
                                                         device_matrix,
                                                         leading_dimension,
                                                         eigenvalues,
                                                         &workspace_size),
                             "cusolverDnZheevd_bufferSize");

        // Allocate workspace and info on device.
        cuDoubleComplex* device_workspace = nullptr;
        int* device_info = nullptr;
        check_cuda_error(cudaMalloc(&device_workspace, sizeof(cuDoubleComplex) * workspace_size),
                         "cudaMalloc workspace");
        check_cuda_error(cudaMalloc(&device_info, sizeof(int)), "cudaMalloc info");

        // Solve.
        check_cusolver_error(cusolverDnZheevd(cusolver_handle_,
                                              job_mode,
                                              uplo,
                                              matrix_dimension,
                                              device_matrix,
                                              leading_dimension,
                                              eigenvalues,
                                              device_workspace,
                                              workspace_size,
                                              device_info),
                             "cusolverDnZheevd");

        // Check for convergence.
        int host_info = 0;
        check_cuda_error(cudaMemcpy(&host_info, device_info, sizeof(int), cudaMemcpyDeviceToHost),
                         "cudaMemcpy info");

        cudaFree(device_workspace);
        cudaFree(device_info);

        if (host_info != 0) {
            throw std::runtime_error("CudaGPUEigensolver: cusolverDnZheevd failed, info = " +
                                     std::to_string(host_info));
        }
    }

    /**
     * Solve A · x = λ · B · x via cusolverDnZhegvd.
     *
     * B must be positive definite. On exit, matrix_a_data holds the
     * eigenvectors and eigenvalues contains eigenvalues in ascending order.
     */
    void generalized_hermitian_eigenvalue_solve(int matrix_dimension,
                                                Complex* matrix_a_data,
                                                int leading_dimension_a,
                                                Complex* matrix_b_data,
                                                int leading_dimension_b,
                                                Real* eigenvalues) override {
        cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;
        cusolverEigMode_t job_mode = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

        auto* device_matrix_a = reinterpret_cast<cuDoubleComplex*>(matrix_a_data);
        auto* device_matrix_b = reinterpret_cast<cuDoubleComplex*>(matrix_b_data);

        // Workspace query.
        int workspace_size = 0;
        check_cusolver_error(cusolverDnZhegvd_bufferSize(cusolver_handle_,
                                                         itype,
                                                         job_mode,
                                                         uplo,
                                                         matrix_dimension,
                                                         device_matrix_a,
                                                         leading_dimension_a,
                                                         device_matrix_b,
                                                         leading_dimension_b,
                                                         eigenvalues,
                                                         &workspace_size),
                             "cusolverDnZhegvd_bufferSize");

        // Allocate workspace and info on device.
        cuDoubleComplex* device_workspace = nullptr;
        int* device_info = nullptr;
        check_cuda_error(cudaMalloc(&device_workspace, sizeof(cuDoubleComplex) * workspace_size),
                         "cudaMalloc workspace");
        check_cuda_error(cudaMalloc(&device_info, sizeof(int)), "cudaMalloc info");

        // Solve.
        check_cusolver_error(cusolverDnZhegvd(cusolver_handle_,
                                              itype,
                                              job_mode,
                                              uplo,
                                              matrix_dimension,
                                              device_matrix_a,
                                              leading_dimension_a,
                                              device_matrix_b,
                                              leading_dimension_b,
                                              eigenvalues,
                                              device_workspace,
                                              workspace_size,
                                              device_info),
                             "cusolverDnZhegvd");

        // Check for convergence.
        int host_info = 0;
        check_cuda_error(cudaMemcpy(&host_info, device_info, sizeof(int), cudaMemcpyDeviceToHost),
                         "cudaMemcpy info");

        cudaFree(device_workspace);
        cudaFree(device_info);

        if (host_info != 0) {
            throw std::runtime_error("CudaGPUEigensolver: cusolverDnZhegvd failed, info = " +
                                     std::to_string(host_info));
        }
    }

  private:
    cusolverDnHandle_t cusolver_handle_ = nullptr;
};

std::unique_ptr<GPUEigensolver>
create_cuda_gpu_eigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<CudaGPUEigensolver>(std::move(device_backend));
}

} // namespace lapack
} // namespace gpu
} // namespace execution
} // namespace q3
