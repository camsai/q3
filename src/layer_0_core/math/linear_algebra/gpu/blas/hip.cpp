/**
 * HIPGPUBLAS — GPUBLAS implementation using AMD hipBLAS.
 *
 * All BLAS operations delegate to hipBLAS routines. hipBLAS operates on
 * device pointers, but with unified memory (hipMallocManaged), the
 * same pointers work on both host and device.
 *
 * Column-major layout is default for hipBLAS, matching Fortran/LAPACK conventions.
 *
 * A hipBLAS handle is created on construction and destroyed on destruction.
 */

#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace blas {

namespace {

/**
 * Check a hipBLAS return code and throw on error.
 */
void check_hipblas_error(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("HIPGPUBLAS: ") + operation + " failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

/**
 * Translate a single-character transpose flag to the hipBLAS enum.
 */
hipblasOperation_t translate_transpose(char transpose_flag) {
    switch (transpose_flag) {
        case 'N':
        case 'n':
            return HIPBLAS_OP_N;
        case 'T':
        case 't':
            return HIPBLAS_OP_T;
        case 'C':
        case 'c':
            return HIPBLAS_OP_C;
        default:
            return HIPBLAS_OP_N;
    }
}

} // anonymous namespace

class HIPGPUBLAS : public GPUBLAS {
  public:
    /**
     * Construct a hipBLAS wrapper, creating a hipBLAS handle.
     */
    explicit HIPGPUBLAS(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : GPUBLAS(std::move(device_backend)) {
        check_hipblas_error(hipblasCreate(&hipblas_handle_), "hipblasCreate");
    }

    ~HIPGPUBLAS() override {
        if (hipblas_handle_ != nullptr) {
            hipblasDestroy(hipblas_handle_);
        }
    }

    // ── Complex double-precision (FP64) ──────────────────────────────────

    /**
     * Complex double-precision ZGEMM via hipblasZgemm.
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
        auto hip_alpha = make_hip_double_complex(alpha);
        auto hip_beta = make_hip_double_complex(beta);
        check_hipblas_error(hipblasZgemm(hipblas_handle_,
                                         translate_transpose(transpose_a),
                                         translate_transpose(transpose_b),
                                         m,
                                         n,
                                         k,
                                         &hip_alpha,
                                         reinterpret_cast<const hipblasDoubleComplex*>(matrix_a),
                                         leading_dimension_a,
                                         reinterpret_cast<const hipblasDoubleComplex*>(matrix_b),
                                         leading_dimension_b,
                                         &hip_beta,
                                         reinterpret_cast<hipblasDoubleComplex*>(matrix_c),
                                         leading_dimension_c),
                            "hipblasZgemm");
    }

    /**
     * Complex double-precision conjugated dot product via hipblasZdotc.
     */
    Complex dot_product(int number_of_elements,
                        const Complex* vector_x,
                        const Complex* vector_y) const override {
        hipblasDoubleComplex result;
        check_hipblas_error(hipblasZdotc(hipblas_handle_,
                                         number_of_elements,
                                         reinterpret_cast<const hipblasDoubleComplex*>(vector_x),
                                         1,
                                         reinterpret_cast<const hipblasDoubleComplex*>(vector_y),
                                         1,
                                         &result),
                            "hipblasZdotc");
        /**
         * Use reinterpret_cast for portable access — hipblasDoubleComplex field
         * names (.real/.imag vs .x/.y) vary across ROCm versions.
         */
        return *reinterpret_cast<Complex*>(&result);
    }

    /**
     * Euclidean norm of a complex double-precision vector via hipblasDznrm2.
     */
    double norm(int number_of_elements, const Complex* vector_x) const override {
        double result = 0.0;
        check_hipblas_error(hipblasDznrm2(hipblas_handle_,
                                          number_of_elements,
                                          reinterpret_cast<const hipblasDoubleComplex*>(vector_x),
                                          1,
                                          &result),
                            "hipblasDznrm2");
        return result;
    }

    /**
     * Scale a complex double-precision vector in-place via hipblasZscal.
     */
    void scale(int number_of_elements, Complex scaling_factor, Complex* vector_x) const override {
        auto hip_alpha = make_hip_double_complex(scaling_factor);
        check_hipblas_error(hipblasZscal(hipblas_handle_,
                                         number_of_elements,
                                         &hip_alpha,
                                         reinterpret_cast<hipblasDoubleComplex*>(vector_x),
                                         1),
                            "hipblasZscal");
    }

    /**
     * Scaled vector addition (ZAXPY) via hipblasZaxpy.
     */
    void axpy(int number_of_elements,
              Complex scaling_factor,
              const Complex* vector_x,
              Complex* vector_y) const override {
        auto hip_alpha = make_hip_double_complex(scaling_factor);
        check_hipblas_error(hipblasZaxpy(hipblas_handle_,
                                         number_of_elements,
                                         &hip_alpha,
                                         reinterpret_cast<const hipblasDoubleComplex*>(vector_x),
                                         1,
                                         reinterpret_cast<hipblasDoubleComplex*>(vector_y),
                                         1),
                            "hipblasZaxpy");
    }

    // ── Complex single-precision (FP32) ──────────────────────────────────

    /**
     * Complex single-precision CGEMM via hipblasCgemm.
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
        auto hip_alpha = make_hip_float_complex(alpha);
        auto hip_beta = make_hip_float_complex(beta);
        check_hipblas_error(hipblasCgemm(hipblas_handle_,
                                         translate_transpose(transpose_a),
                                         translate_transpose(transpose_b),
                                         m,
                                         n,
                                         k,
                                         &hip_alpha,
                                         reinterpret_cast<const hipblasComplex*>(matrix_a),
                                         leading_dimension_a,
                                         reinterpret_cast<const hipblasComplex*>(matrix_b),
                                         leading_dimension_b,
                                         &hip_beta,
                                         reinterpret_cast<hipblasComplex*>(matrix_c),
                                         leading_dimension_c),
                            "hipblasCgemm");
    }

    /**
     * Complex single-precision conjugated dot product via hipblasCdotc.
     */
    ComplexSingle dot_product_single(int number_of_elements,
                                     const ComplexSingle* vector_x,
                                     const ComplexSingle* vector_y) const override {
        hipblasComplex result;
        check_hipblas_error(hipblasCdotc(hipblas_handle_,
                                         number_of_elements,
                                         reinterpret_cast<const hipblasComplex*>(vector_x),
                                         1,
                                         reinterpret_cast<const hipblasComplex*>(vector_y),
                                         1,
                                         &result),
                            "hipblasCdotc");
        /**
         * Use reinterpret_cast for portable access — hipblasComplex field
         * names (.real/.imag vs .x/.y) vary across ROCm versions.
         */
        return *reinterpret_cast<ComplexSingle*>(&result);
    }

    /**
     * Euclidean norm of a complex single-precision vector via hipblasScnrm2.
     */
    float norm_single(int number_of_elements, const ComplexSingle* vector_x) const override {
        float result = 0.0f;
        check_hipblas_error(hipblasScnrm2(hipblas_handle_,
                                          number_of_elements,
                                          reinterpret_cast<const hipblasComplex*>(vector_x),
                                          1,
                                          &result),
                            "hipblasScnrm2");
        return result;
    }

    // ── Real double-precision (FP64) — for gamma-only ────────────────────

    /**
     * Real double-precision DGEMM via hipblasDgemm.
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
        check_hipblas_error(hipblasDgemm(hipblas_handle_,
                                         translate_transpose(transpose_a),
                                         translate_transpose(transpose_b),
                                         m,
                                         n,
                                         k,
                                         &alpha,
                                         matrix_a,
                                         leading_dimension_a,
                                         matrix_b,
                                         leading_dimension_b,
                                         &beta,
                                         matrix_c,
                                         leading_dimension_c),
                            "hipblasDgemm");
    }

  private:
    hipblasHandle_t hipblas_handle_ = nullptr;

    /**
     * Convert std::complex<double> to hipblasDoubleComplex.
     */
    static hipblasDoubleComplex make_hip_double_complex(Complex value) {
        return {value.real(), value.imag()};
    }

    /**
     * Convert std::complex<float> to hipblasComplex.
     */
    static hipblasComplex make_hip_float_complex(ComplexSingle value) {
        return {value.real(), value.imag()};
    }
};

std::unique_ptr<GPUBLAS>
create_hip_gpu_blas(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<HIPGPUBLAS>(std::move(device_backend));
}

} // namespace blas
} // namespace gpu
} // namespace execution
} // namespace q3
