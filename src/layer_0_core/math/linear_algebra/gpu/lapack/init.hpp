#pragma once

/**
 * GPUEigensolver — vendor-neutral GPU eigenvalue solver interface.
 *
 * Provides Hermitian eigenvalue decomposition for the subspace
 * eigenproblem in the Davidson diagonalizer. Subspace matrices are
 * always small (N×N where N = number of bands, typically 8–100), so
 * the eigensolver always runs on CPU even in GPU builds.
 *
 * Each concrete implementation owns its own vendor handle (e.g.,
 * cuSOLVER handle or Accelerate LAPACK session).
 *
 * No CUDA headers. No Metal headers. Vendor-neutral interface only.
 * Concrete implementations live in:
 *   - cuda.hpp / cuda.cpp   (CudaGPUEigensolver — cuSOLVER / LAPACK)
 *   - metal.hpp / metal.cpp (MetalGPUEigensolver — Accelerate LAPACK)
 */

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/types/type_aliases.hpp"

#include <memory>

namespace q3 {
namespace execution {
namespace gpu {
namespace lapack {

class GPUEigensolver {
  public:
    /**
     * Construct a GPU eigensolver backed by the given device.
     *
     * @param device_backend  Shared pointer to the GPU device backend.
     *                        Must not be null.
     */
    explicit GPUEigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : device_backend_(std::move(device_backend)) {}

    virtual ~GPUEigensolver() = default;

    /**
     * Solve the standard Hermitian eigenvalue problem: A · x = λ · x.
     *
     * Computes all eigenvalues and eigenvectors of a complex Hermitian
     * matrix. On return, matrix_data contains the eigenvectors and
     * eigenvalues contains the eigenvalues in ascending order.
     *
     * @param matrix_dimension   Order of the matrix (N).
     * @param matrix_data        On entry: the Hermitian matrix (upper triangle).
     *                           On exit: the orthonormal eigenvectors.
     * @param leading_dimension  Leading dimension of matrix_data (>= N).
     * @param eigenvalues        Output array of N real eigenvalues (ascending).
     */
    virtual void hermitian_eigenvalue_solve(int matrix_dimension,
                                            Complex* matrix_data,
                                            int leading_dimension,
                                            Real* eigenvalues) = 0;

    /**
     * Solve the generalized Hermitian eigenvalue problem: A · x = λ · B · x.
     *
     * B must be positive definite (e.g., the overlap matrix S for US-PP).
     * On return, matrix_a_data contains the eigenvectors and eigenvalues
     * contains the eigenvalues in ascending order.
     *
     * @param matrix_dimension    Order of both matrices (N).
     * @param matrix_a_data       On entry: Hermitian matrix A (upper triangle).
     *                            On exit: the eigenvectors.
     * @param leading_dimension_a Leading dimension of matrix_a_data (>= N).
     * @param matrix_b_data       On entry: positive-definite Hermitian B.
     *                            On exit: the Cholesky factor of B.
     * @param leading_dimension_b Leading dimension of matrix_b_data (>= N).
     * @param eigenvalues         Output array of N real eigenvalues (ascending).
     */
    virtual void generalized_hermitian_eigenvalue_solve(int matrix_dimension,
                                                        Complex* matrix_a_data,
                                                        int leading_dimension_a,
                                                        Complex* matrix_b_data,
                                                        int leading_dimension_b,
                                                        Real* eigenvalues) = 0;

    /**
     * Access the underlying GPU device backend.
     */
    device::GPUDeviceBackend& device_backend() {
        return *device_backend_;
    }

    /**
     * Access the underlying GPU device backend (const).
     */
    const device::GPUDeviceBackend& device_backend() const {
        return *device_backend_;
    }

  protected:
    std::shared_ptr<device::GPUDeviceBackend> device_backend_;
};

} // namespace lapack
} // namespace gpu
} // namespace execution
} // namespace q3
