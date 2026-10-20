/**
 * AccelerateGPUEigensolver — GPUEigensolver implementation using Apple's
 * Accelerate framework.
 *
 * Provides Hermitian eigenvalue solvers via the LAPACK routines zheev_
 * and zhegv_ included in Accelerate.framework. These always run on CPU,
 * which is appropriate since the subspace matrices are small (N = number
 * of bands, typically 8–100).
 *
 * Uses the standard workspace-query pattern: call with lwork=-1 to
 * determine optimal workspace size, then allocate and solve.
 */

#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"

#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace q3 {
namespace execution {
namespace gpu {
namespace lapack {

class AccelerateGPUEigensolver : public GPUEigensolver {
  public:
    /**
     * Construct an Accelerate eigensolver backed by the given device.
     *
     * No vendor handle is needed — Accelerate LAPACK routines are stateless.
     */
    explicit AccelerateGPUEigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : GPUEigensolver(std::move(device_backend)) {}

    ~AccelerateGPUEigensolver() override = default;

    /**
     * Solve A · x = λ · x via LAPACK zheev_ (Accelerate).
     *
     * On entry, matrix_data holds the Hermitian matrix (upper triangle).
     * On exit, it holds the orthonormal eigenvectors, and eigenvalues
     * contains the eigenvalues in ascending order.
     */
    void hermitian_eigenvalue_solve(int matrix_dimension,
                                    Complex* matrix_data,
                                    int leading_dimension,
                                    Real* eigenvalues) override {
        char job = 'V';
        char uplo = 'U';
        auto* matrix = reinterpret_cast<__CLPK_doublecomplex*>(matrix_data);
        __CLPK_integer dimension = static_cast<__CLPK_integer>(matrix_dimension);
        __CLPK_integer lead_dim = static_cast<__CLPK_integer>(leading_dimension);
        __CLPK_integer info = 0;

        // Workspace query
        __CLPK_integer workspace_query_size = -1;
        __CLPK_doublecomplex optimal_workspace_size;
        std::vector<double> real_workspace(
            static_cast<size_t>(std::max(__CLPK_integer{1}, 3 * dimension - 2)));

        zheev_(&job,
               &uplo,
               &dimension,
               matrix,
               &lead_dim,
               eigenvalues,
               &optimal_workspace_size,
               &workspace_query_size,
               real_workspace.data(),
               &info);

        if (info != 0) {
            throw std::runtime_error("AccelerateGPUEigensolver: zheev_ workspace query failed, "
                                     "info = " +
                                     std::to_string(info));
        }

        // Allocate optimal workspace and solve
        __CLPK_integer optimal_size = static_cast<__CLPK_integer>(optimal_workspace_size.r);
        std::vector<__CLPK_doublecomplex> workspace(static_cast<size_t>(optimal_size));

        zheev_(&job,
               &uplo,
               &dimension,
               matrix,
               &lead_dim,
               eigenvalues,
               workspace.data(),
               &optimal_size,
               real_workspace.data(),
               &info);

        if (info != 0) {
            throw std::runtime_error("AccelerateGPUEigensolver: zheev_ failed, info = " +
                                     std::to_string(info));
        }
    }

    /**
     * Solve A · x = λ · B · x via LAPACK zhegv_ (Accelerate).
     *
     * B must be positive definite. On exit, matrix_a_data holds the
     * eigenvectors and eigenvalues contains the eigenvalues in ascending
     * order. matrix_b_data is overwritten with the Cholesky factor of B.
     */
    void generalized_hermitian_eigenvalue_solve(int matrix_dimension,
                                                Complex* matrix_a_data,
                                                int leading_dimension_a,
                                                Complex* matrix_b_data,
                                                int leading_dimension_b,
                                                Real* eigenvalues) override {
        __CLPK_integer itype = 1;
        char job = 'V';
        char uplo = 'U';
        auto* matrix_a = reinterpret_cast<__CLPK_doublecomplex*>(matrix_a_data);
        auto* matrix_b = reinterpret_cast<__CLPK_doublecomplex*>(matrix_b_data);
        __CLPK_integer dimension = static_cast<__CLPK_integer>(matrix_dimension);
        __CLPK_integer lead_dim_a = static_cast<__CLPK_integer>(leading_dimension_a);
        __CLPK_integer lead_dim_b = static_cast<__CLPK_integer>(leading_dimension_b);
        __CLPK_integer info = 0;

        // Workspace query
        __CLPK_integer workspace_query_size = -1;
        __CLPK_doublecomplex optimal_workspace_size;
        std::vector<double> real_workspace(
            static_cast<size_t>(std::max(__CLPK_integer{1}, 3 * dimension - 2)));

        zhegv_(&itype,
               &job,
               &uplo,
               &dimension,
               matrix_a,
               &lead_dim_a,
               matrix_b,
               &lead_dim_b,
               eigenvalues,
               &optimal_workspace_size,
               &workspace_query_size,
               real_workspace.data(),
               &info);

        if (info != 0) {
            throw std::runtime_error("AccelerateGPUEigensolver: zhegv_ workspace query failed, "
                                     "info = " +
                                     std::to_string(info));
        }

        // Allocate optimal workspace and solve
        __CLPK_integer optimal_size = static_cast<__CLPK_integer>(optimal_workspace_size.r);
        std::vector<__CLPK_doublecomplex> workspace(static_cast<size_t>(optimal_size));

        zhegv_(&itype,
               &job,
               &uplo,
               &dimension,
               matrix_a,
               &lead_dim_a,
               matrix_b,
               &lead_dim_b,
               eigenvalues,
               workspace.data(),
               &optimal_size,
               real_workspace.data(),
               &info);

        if (info != 0) {
            throw std::runtime_error("AccelerateGPUEigensolver: zhegv_ failed, info = " +
                                     std::to_string(info));
        }
    }
};

std::unique_ptr<GPUEigensolver>
create_accelerate_gpu_eigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<AccelerateGPUEigensolver>(std::move(device_backend));
}

} // namespace lapack
} // namespace gpu
} // namespace execution
} // namespace q3
