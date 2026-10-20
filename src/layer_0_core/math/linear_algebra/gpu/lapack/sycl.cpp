/**
 * SYCLGPUEigensolver — GPUEigensolver implementation using Intel oneMKL LAPACK.
 *
 * Provides Hermitian eigenvalue solvers via oneapi::mkl::lapack::heevd (standard)
 * and oneapi::mkl::lapack::hegvd (generalized). oneMKL LAPACK operates on USM
 * pointers; with sycl::malloc_shared, the same pointers work seamlessly.
 *
 * Scratchpad memory is queried via heevd_scratchpad_size / hegvd_scratchpad_size
 * and allocated as device memory for each call.
 */

#include "layer_0_core/execution/gpu/device/sycl.hpp"
#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"

#include <oneapi/mkl/lapack.hpp>
#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace lapack {

class SYCLGPUEigensolver : public GPUEigensolver {
  public:
    /**
     * Construct a oneMKL LAPACK eigensolver.
     *
     * Obtains the SYCL queue from the device backend via dynamic_cast.
     * Throws if the device backend is not a SYCLGPUDeviceBackend.
     */
    explicit SYCLGPUEigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : GPUEigensolver(std::move(device_backend)) {
        auto* sycl_backend = dynamic_cast<device::SYCLGPUDeviceBackend*>(device_backend_.get());
        if (sycl_backend == nullptr) {
            throw std::runtime_error(
                "SYCLGPUEigensolver: device_backend is not a SYCLGPUDeviceBackend");
        }
        queue_ = &sycl_backend->queue();
    }

    ~SYCLGPUEigensolver() override = default;

    /**
     * Solve A · x = λ · x via oneMKL heevd.
     *
     * Uses the divide-and-conquer algorithm for better performance on
     * larger matrices. On entry, matrix_data holds the Hermitian matrix
     * (upper triangle). On exit, it holds the eigenvectors.
     */
    void hermitian_eigenvalue_solve(int matrix_dimension,
                                    Complex* matrix_data,
                                    int leading_dimension,
                                    Real* eigenvalues) override {
        auto job = oneapi::mkl::job::vec;
        auto uplo = oneapi::mkl::uplo::upper;
        std::int64_t dimension = static_cast<std::int64_t>(matrix_dimension);
        std::int64_t lead_dim = static_cast<std::int64_t>(leading_dimension);

        // Query scratchpad size.
        std::int64_t scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<Complex>(
            *queue_, job, uplo, dimension, lead_dim);

        // Allocate scratchpad on device.
        Complex* scratchpad =
            sycl::malloc_device<Complex>(static_cast<size_t>(scratchpad_size), *queue_);
        if (scratchpad == nullptr) {
            throw std::runtime_error(
                "SYCLGPUEigensolver: sycl::malloc_device for heevd scratchpad failed");
        }

        try {
            oneapi::mkl::lapack::heevd(*queue_,
                                       job,
                                       uplo,
                                       dimension,
                                       matrix_data,
                                       lead_dim,
                                       eigenvalues,
                                       scratchpad,
                                       scratchpad_size)
                .wait();
        } catch (const oneapi::mkl::lapack::computation_error& error) {
            sycl::free(scratchpad, *queue_);
            throw std::runtime_error("SYCLGPUEigensolver: heevd failed, info = " +
                                     std::to_string(error.info()));
        } catch (...) {
            sycl::free(scratchpad, *queue_);
            throw;
        }

        sycl::free(scratchpad, *queue_);
    }

    /**
     * Solve A · x = λ · B · x via oneMKL hegvd.
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
        std::int64_t itype = 1;
        auto job = oneapi::mkl::job::vec;
        auto uplo = oneapi::mkl::uplo::upper;
        std::int64_t dimension = static_cast<std::int64_t>(matrix_dimension);
        std::int64_t lead_dim_a = static_cast<std::int64_t>(leading_dimension_a);
        std::int64_t lead_dim_b = static_cast<std::int64_t>(leading_dimension_b);

        // Query scratchpad size.
        std::int64_t scratchpad_size = oneapi::mkl::lapack::hegvd_scratchpad_size<Complex>(
            *queue_, itype, job, uplo, dimension, lead_dim_a, lead_dim_b);

        // Allocate scratchpad on device.
        Complex* scratchpad =
            sycl::malloc_device<Complex>(static_cast<size_t>(scratchpad_size), *queue_);
        if (scratchpad == nullptr) {
            throw std::runtime_error(
                "SYCLGPUEigensolver: sycl::malloc_device for hegvd scratchpad failed");
        }

        try {
            oneapi::mkl::lapack::hegvd(*queue_,
                                       itype,
                                       job,
                                       uplo,
                                       dimension,
                                       matrix_a_data,
                                       lead_dim_a,
                                       matrix_b_data,
                                       lead_dim_b,
                                       eigenvalues,
                                       scratchpad,
                                       scratchpad_size)
                .wait();
        } catch (const oneapi::mkl::lapack::computation_error& error) {
            sycl::free(scratchpad, *queue_);
            throw std::runtime_error("SYCLGPUEigensolver: hegvd failed, info = " +
                                     std::to_string(error.info()));
        } catch (...) {
            sycl::free(scratchpad, *queue_);
            throw;
        }

        sycl::free(scratchpad, *queue_);
    }

  private:
    /**
     * Non-owning pointer to the SYCL queue from the device backend.
     * Lifetime is guaranteed by device_backend_ (shared_ptr) in the base class.
     */
    sycl::queue* queue_ = nullptr;
};

std::unique_ptr<GPUEigensolver>
create_sycl_gpu_eigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<SYCLGPUEigensolver>(std::move(device_backend));
}

} // namespace lapack
} // namespace gpu
} // namespace execution
} // namespace q3
