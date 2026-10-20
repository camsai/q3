/**
 * HIPGPUEigensolver — GPUEigensolver implementation using AMD hipSOLVER.
 *
 * Provides Hermitian eigenvalue solvers via hipsolverDnZheevd (standard)
 * and hipsolverDnZhegvd (generalized). hipSOLVER operates on device memory;
 * with unified memory (hipMallocManaged) the same pointers work seamlessly.
 *
 * A hipSOLVER handle is created on construction and destroyed on destruction.
 * Workspace is allocated per-call using hipMalloc (device-only scratch space).
 */

#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"

#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>
#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace lapack {

namespace {

/**
 * Check a hipSOLVER return code and throw on error.
 */
void check_hipsolver_error(hipsolverStatus_t status, const char* operation) {
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("HIPGPUEigensolver: ") + operation +
                                 " failed with status " + std::to_string(static_cast<int>(status)));
    }
}

/**
 * Check a HIP runtime return code and throw on error.
 */
void check_hip_error(hipError_t error_code, const char* operation) {
    if (error_code != hipSuccess) {
        throw std::runtime_error(std::string("HIPGPUEigensolver: ") + operation +
                                 " failed: " + hipGetErrorString(error_code));
    }
}

} // anonymous namespace

class HIPGPUEigensolver : public GPUEigensolver {
  public:
    /**
     * Construct a hipSOLVER eigensolver, creating a hipSOLVER handle.
     */
    explicit HIPGPUEigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : GPUEigensolver(std::move(device_backend)) {
        check_hipsolver_error(hipsolverCreate(&hipsolver_handle_), "hipsolverCreate");
    }

    ~HIPGPUEigensolver() override {
        if (hipsolver_handle_ != nullptr) {
            hipsolverDestroy(hipsolver_handle_);
        }
    }

    /**
     * Solve A · x = λ · x via hipsolverDnZheevd.
     *
     * Uses the divide-and-conquer algorithm for better performance on
     * large matrices. On entry, matrix_data holds the Hermitian matrix
     * (upper triangle). On exit, it holds the eigenvectors.
     */
    void hermitian_eigenvalue_solve(int matrix_dimension,
                                    Complex* matrix_data,
                                    int leading_dimension,
                                    Real* eigenvalues) override {
        hipsolverEigMode_t job_mode = HIPSOLVER_EIG_MODE_VECTOR;
        hipsolverFillMode_t uplo = HIPSOLVER_FILL_MODE_UPPER;

        auto* device_matrix = reinterpret_cast<hipDoubleComplex*>(matrix_data);

        // Workspace query.
        int workspace_size = 0;
        check_hipsolver_error(hipsolverDnZheevd_bufferSize(hipsolver_handle_,
                                                           job_mode,
                                                           uplo,
                                                           matrix_dimension,
                                                           device_matrix,
                                                           leading_dimension,
                                                           eigenvalues,
                                                           &workspace_size),
                              "hipsolverDnZheevd_bufferSize");

        // Allocate workspace and info on device.
        hipDoubleComplex* device_workspace = nullptr;
        int* device_info = nullptr;
        check_hip_error(hipMalloc(&device_workspace, sizeof(hipDoubleComplex) * workspace_size),
                        "hipMalloc workspace");
        check_hip_error(hipMalloc(&device_info, sizeof(int)), "hipMalloc info");

        // Solve.
        check_hipsolver_error(hipsolverDnZheevd(hipsolver_handle_,
                                                job_mode,
                                                uplo,
                                                matrix_dimension,
                                                device_matrix,
                                                leading_dimension,
                                                eigenvalues,
                                                device_workspace,
                                                workspace_size,
                                                device_info),
                              "hipsolverDnZheevd");

        // Check for convergence.
        int host_info = 0;
        check_hip_error(hipMemcpy(&host_info, device_info, sizeof(int), hipMemcpyDeviceToHost),
                        "hipMemcpy info");

        hipFree(device_workspace);
        hipFree(device_info);

        if (host_info != 0) {
            throw std::runtime_error("HIPGPUEigensolver: hipsolverDnZheevd failed, info = " +
                                     std::to_string(host_info));
        }
    }

    /**
     * Solve A · x = λ · B · x via hipsolverDnZhegvd.
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
        hipsolverEigType_t itype = HIPSOLVER_EIG_TYPE_1;
        hipsolverEigMode_t job_mode = HIPSOLVER_EIG_MODE_VECTOR;
        hipsolverFillMode_t uplo = HIPSOLVER_FILL_MODE_UPPER;

        auto* device_matrix_a = reinterpret_cast<hipDoubleComplex*>(matrix_a_data);
        auto* device_matrix_b = reinterpret_cast<hipDoubleComplex*>(matrix_b_data);

        // Workspace query.
        int workspace_size = 0;
        check_hipsolver_error(hipsolverDnZhegvd_bufferSize(hipsolver_handle_,
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
                              "hipsolverDnZhegvd_bufferSize");

        // Allocate workspace and info on device.
        hipDoubleComplex* device_workspace = nullptr;
        int* device_info = nullptr;
        check_hip_error(hipMalloc(&device_workspace, sizeof(hipDoubleComplex) * workspace_size),
                        "hipMalloc workspace");
        check_hip_error(hipMalloc(&device_info, sizeof(int)), "hipMalloc info");

        // Solve.
        check_hipsolver_error(hipsolverDnZhegvd(hipsolver_handle_,
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
                              "hipsolverDnZhegvd");

        // Check for convergence.
        int host_info = 0;
        check_hip_error(hipMemcpy(&host_info, device_info, sizeof(int), hipMemcpyDeviceToHost),
                        "hipMemcpy info");

        hipFree(device_workspace);
        hipFree(device_info);

        if (host_info != 0) {
            throw std::runtime_error("HIPGPUEigensolver: hipsolverDnZhegvd failed, info = " +
                                     std::to_string(host_info));
        }
    }

  private:
    hipsolverHandle_t hipsolver_handle_ = nullptr;
};

std::unique_ptr<GPUEigensolver>
create_hip_gpu_eigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<HIPGPUEigensolver>(std::move(device_backend));
}

} // namespace lapack
} // namespace gpu
} // namespace execution
} // namespace q3
