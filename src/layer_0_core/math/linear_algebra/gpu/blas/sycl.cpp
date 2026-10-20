/**
 * SYCLGPUBLAS — GPUBLAS implementation using Intel oneMKL BLAS.
 *
 * All BLAS operations delegate to oneMKL column-major USM routines.
 * oneMKL operates on USM pointers; with sycl::malloc_shared, the
 * same pointers work on both host and device.
 *
 * Column-major layout is default for oneMKL, matching Fortran/LAPACK conventions.
 *
 * The SYCL queue is obtained via dynamic_cast from GPUDeviceBackend to
 * SYCLGPUDeviceBackend (defined in device/sycl.hpp), which is the standard
 * pattern for vendor-specific backends that need access to vendor-specific handles.
 *
 * Scalar result buffers (for dot product and norm) are pre-allocated in shared
 * memory during construction to avoid per-call malloc/free overhead in the
 * Davidson hot loop.
 */

#include "layer_0_core/execution/gpu/device/sycl.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"

#include <oneapi/mkl/blas.hpp>
#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace blas {

namespace {

/**
 * Translate a single-character transpose flag to the oneMKL enum.
 */
oneapi::mkl::transpose translate_transpose(char transpose_flag) {
    switch (transpose_flag) {
        case 'N':
        case 'n':
            return oneapi::mkl::transpose::nontrans;
        case 'T':
        case 't':
            return oneapi::mkl::transpose::trans;
        case 'C':
        case 'c':
            return oneapi::mkl::transpose::conjtrans;
        default:
            return oneapi::mkl::transpose::nontrans;
    }
}

} // anonymous namespace

class SYCLGPUBLAS : public GPUBLAS {
  public:
    /**
     * Construct a oneMKL BLAS wrapper.
     *
     * Obtains the SYCL queue from the device backend via dynamic_cast.
     * Pre-allocates persistent shared-memory buffers for scalar results
     * to avoid per-call allocation overhead.
     * Throws if the device backend is not a SYCLGPUDeviceBackend.
     */
    explicit SYCLGPUBLAS(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : GPUBLAS(std::move(device_backend)) {
        auto* sycl_backend = dynamic_cast<device::SYCLGPUDeviceBackend*>(device_backend_.get());
        if (sycl_backend == nullptr) {
            throw std::runtime_error("SYCLGPUBLAS: device_backend is not a SYCLGPUDeviceBackend");
        }
        queue_ = &sycl_backend->queue();

        // Pre-allocate persistent scalar result buffers in shared memory.
        result_complex_ = sycl::malloc_shared<Complex>(1, *queue_);
        result_double_ = sycl::malloc_shared<double>(1, *queue_);
        result_complex_single_ = sycl::malloc_shared<ComplexSingle>(1, *queue_);
        result_float_ = sycl::malloc_shared<float>(1, *queue_);
        if (result_complex_ == nullptr || result_double_ == nullptr ||
            result_complex_single_ == nullptr || result_float_ == nullptr) {
            throw std::runtime_error("SYCLGPUBLAS: failed to allocate scalar result buffers");
        }
    }

    ~SYCLGPUBLAS() override {
        if (result_complex_ != nullptr) {
            sycl::free(result_complex_, *queue_);
        }
        if (result_double_ != nullptr) {
            sycl::free(result_double_, *queue_);
        }
        if (result_complex_single_ != nullptr) {
            sycl::free(result_complex_single_, *queue_);
        }
        if (result_float_ != nullptr) {
            sycl::free(result_float_, *queue_);
        }
    }

    // ── Complex double-precision (FP64) ──────────────────────────────────

    /**
     * Complex double-precision ZGEMM via oneMKL gemm.
     */
    void gemm(char transpose_a,
              char transpose_b,
              int m,
              int n,
              int k,
              Complex alpha,
              const Complex* matrix_a,
              int leading_dimension_a,
              const Complex* matrix_b,
              int leading_dimension_b,
              Complex beta,
              Complex* matrix_c,
              int leading_dimension_c) const override {
        oneapi::mkl::blas::column_major::gemm(*queue_,
                                              translate_transpose(transpose_a),
                                              translate_transpose(transpose_b),
                                              static_cast<std::int64_t>(m),
                                              static_cast<std::int64_t>(n),
                                              static_cast<std::int64_t>(k),
                                              alpha,
                                              matrix_a,
                                              static_cast<std::int64_t>(leading_dimension_a),
                                              matrix_b,
                                              static_cast<std::int64_t>(leading_dimension_b),
                                              beta,
                                              matrix_c,
                                              static_cast<std::int64_t>(leading_dimension_c))
            .wait();
    }

    /**
     * Complex double-precision conjugated dot product via oneMKL dotc.
     *
     * Uses a pre-allocated shared-memory scalar buffer to avoid per-call
     * allocation overhead in the Davidson hot loop.
     */
    Complex dot_product(int number_of_elements,
                        const Complex* vector_x,
                        const Complex* vector_y) const override {
        oneapi::mkl::blas::column_major::dotc(*queue_,
                                              static_cast<std::int64_t>(number_of_elements),
                                              vector_x,
                                              static_cast<std::int64_t>(1),
                                              vector_y,
                                              static_cast<std::int64_t>(1),
                                              result_complex_)
            .wait();

        return *result_complex_;
    }

    /**
     * Euclidean norm of a complex double-precision vector via oneMKL nrm2.
     *
     * Uses a pre-allocated shared-memory scalar buffer.
     */
    double norm(int number_of_elements, const Complex* vector_x) const override {
        oneapi::mkl::blas::column_major::nrm2(*queue_,
                                              static_cast<std::int64_t>(number_of_elements),
                                              vector_x,
                                              static_cast<std::int64_t>(1),
                                              result_double_)
            .wait();

        return *result_double_;
    }

    /**
     * Scale a complex double-precision vector in-place via oneMKL scal.
     */
    void scale(int number_of_elements, Complex scaling_factor, Complex* vector_x) const override {
        oneapi::mkl::blas::column_major::scal(*queue_,
                                              static_cast<std::int64_t>(number_of_elements),
                                              scaling_factor,
                                              vector_x,
                                              static_cast<std::int64_t>(1))
            .wait();
    }

    /**
     * Scaled vector addition (ZAXPY) via oneMKL axpy.
     */
    void axpy(int number_of_elements,
              Complex scaling_factor,
              const Complex* vector_x,
              Complex* vector_y) const override {
        oneapi::mkl::blas::column_major::axpy(*queue_,
                                              static_cast<std::int64_t>(number_of_elements),
                                              scaling_factor,
                                              vector_x,
                                              static_cast<std::int64_t>(1),
                                              vector_y,
                                              static_cast<std::int64_t>(1))
            .wait();
    }

    // ── Complex single-precision (FP32) ──────────────────────────────────

    /**
     * Complex single-precision CGEMM via oneMKL gemm.
     */
    void gemm_single(char transpose_a,
                     char transpose_b,
                     int m,
                     int n,
                     int k,
                     ComplexSingle alpha,
                     const ComplexSingle* matrix_a,
                     int leading_dimension_a,
                     const ComplexSingle* matrix_b,
                     int leading_dimension_b,
                     ComplexSingle beta,
                     ComplexSingle* matrix_c,
                     int leading_dimension_c) const override {
        oneapi::mkl::blas::column_major::gemm(*queue_,
                                              translate_transpose(transpose_a),
                                              translate_transpose(transpose_b),
                                              static_cast<std::int64_t>(m),
                                              static_cast<std::int64_t>(n),
                                              static_cast<std::int64_t>(k),
                                              alpha,
                                              matrix_a,
                                              static_cast<std::int64_t>(leading_dimension_a),
                                              matrix_b,
                                              static_cast<std::int64_t>(leading_dimension_b),
                                              beta,
                                              matrix_c,
                                              static_cast<std::int64_t>(leading_dimension_c))
            .wait();
    }

    /**
     * Complex single-precision conjugated dot product via oneMKL dotc.
     *
     * Uses a pre-allocated shared-memory scalar buffer.
     */
    ComplexSingle dot_product_single(int number_of_elements,
                                     const ComplexSingle* vector_x,
                                     const ComplexSingle* vector_y) const override {
        oneapi::mkl::blas::column_major::dotc(*queue_,
                                              static_cast<std::int64_t>(number_of_elements),
                                              vector_x,
                                              static_cast<std::int64_t>(1),
                                              vector_y,
                                              static_cast<std::int64_t>(1),
                                              result_complex_single_)
            .wait();

        return *result_complex_single_;
    }

    /**
     * Euclidean norm of a complex single-precision vector via oneMKL nrm2.
     *
     * Uses a pre-allocated shared-memory scalar buffer.
     */
    float norm_single(int number_of_elements, const ComplexSingle* vector_x) const override {
        oneapi::mkl::blas::column_major::nrm2(*queue_,
                                              static_cast<std::int64_t>(number_of_elements),
                                              vector_x,
                                              static_cast<std::int64_t>(1),
                                              result_float_)
            .wait();

        return *result_float_;
    }

    // ── Real double-precision (FP64) — for gamma-only ────────────────────

    /**
     * Real double-precision DGEMM via oneMKL gemm.
     */
    void dgemm(char transpose_a,
               char transpose_b,
               int m,
               int n,
               int k,
               Real alpha,
               const Real* matrix_a,
               int leading_dimension_a,
               const Real* matrix_b,
               int leading_dimension_b,
               Real beta,
               Real* matrix_c,
               int leading_dimension_c) const override {
        oneapi::mkl::blas::column_major::gemm(*queue_,
                                              translate_transpose(transpose_a),
                                              translate_transpose(transpose_b),
                                              static_cast<std::int64_t>(m),
                                              static_cast<std::int64_t>(n),
                                              static_cast<std::int64_t>(k),
                                              alpha,
                                              matrix_a,
                                              static_cast<std::int64_t>(leading_dimension_a),
                                              matrix_b,
                                              static_cast<std::int64_t>(leading_dimension_b),
                                              beta,
                                              matrix_c,
                                              static_cast<std::int64_t>(leading_dimension_c))
            .wait();
    }

  private:
    /**
     * Non-owning pointer to the SYCL queue from the device backend.
     * Lifetime is guaranteed by device_backend_ (shared_ptr) in the base class.
     */
    sycl::queue* queue_ = nullptr;

    /**
     * Pre-allocated scalar result buffers in shared memory.
     * Avoids per-call malloc_shared/free overhead in the Davidson hot loop.
     */
    Complex* result_complex_ = nullptr;
    double* result_double_ = nullptr;
    ComplexSingle* result_complex_single_ = nullptr;
    float* result_float_ = nullptr;
};

std::unique_ptr<GPUBLAS>
create_sycl_gpu_blas(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<SYCLGPUBLAS>(std::move(device_backend));
}

} // namespace blas
} // namespace gpu
} // namespace execution
} // namespace q3
