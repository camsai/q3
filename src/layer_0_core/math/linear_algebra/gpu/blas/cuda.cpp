/**
 * CudaGPUBLAS — GPUBLAS implementation using NVIDIA cuBLAS.
 *
 * All BLAS operations delegate to cuBLAS routines. cuBLAS operates on
 * device pointers, but with unified memory (cudaMallocManaged), the
 * same pointers work on both host and device.
 *
 * Column-major layout is default for cuBLAS, matching Fortran/LAPACK conventions.
 *
 * A cuBLAS handle is created on construction and destroyed on destruction.
 */

#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace blas {

namespace {

/**
 * Check a cuBLAS return code and throw on error.
 */
void check_cublas_error(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("CudaGPUBLAS: ") + operation + " failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

/**
 * Translate a single-character transpose flag to the cuBLAS enum.
 */
cublasOperation_t translate_transpose(char transpose_flag) {
    switch (transpose_flag) {
        case 'N':
        case 'n':
            return CUBLAS_OP_N;
        case 'T':
        case 't':
            return CUBLAS_OP_T;
        case 'C':
        case 'c':
            return CUBLAS_OP_C;
        default:
            return CUBLAS_OP_N;
    }
}

} // anonymous namespace

class CudaGPUBLAS : public GPUBLAS {
  public:
    /**
     * Construct a cuBLAS wrapper, creating a cuBLAS handle.
     */
    explicit CudaGPUBLAS(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : GPUBLAS(std::move(device_backend)) {
        check_cublas_error(cublasCreate(&cublas_handle_), "cublasCreate");
    }

    ~CudaGPUBLAS() override {
        if (cublas_handle_ != nullptr) {
            cublasDestroy(cublas_handle_);
        }
    }

    // ── Complex double-precision (FP64) ──────────────────────────────────

    /**
     * Complex double-precision ZGEMM via cublasZgemm.
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
        auto cuda_alpha = make_cuda_double_complex(alpha);
        auto cuda_beta = make_cuda_double_complex(beta);
        check_cublas_error(cublasZgemm(cublas_handle_,
                                       translate_transpose(transpose_a),
                                       translate_transpose(transpose_b),
                                       m,
                                       n,
                                       k,
                                       &cuda_alpha,
                                       reinterpret_cast<const cuDoubleComplex*>(matrix_a),
                                       leading_dimension_a,
                                       reinterpret_cast<const cuDoubleComplex*>(matrix_b),
                                       leading_dimension_b,
                                       &cuda_beta,
                                       reinterpret_cast<cuDoubleComplex*>(matrix_c),
                                       leading_dimension_c),
                           "cublasZgemm");
    }

    /**
     * Complex double-precision conjugated dot product via cublasZdotc.
     */
    Complex dot_product(int number_of_elements,
                        const Complex* vector_x,
                        const Complex* vector_y) const override {
        cuDoubleComplex result;
        check_cublas_error(cublasZdotc(cublas_handle_,
                                       number_of_elements,
                                       reinterpret_cast<const cuDoubleComplex*>(vector_x),
                                       1,
                                       reinterpret_cast<const cuDoubleComplex*>(vector_y),
                                       1,
                                       &result),
                           "cublasZdotc");
        return Complex(result.x, result.y);
    }

    /**
     * Euclidean norm of a complex double-precision vector via cublasDznrm2.
     */
    double norm(int number_of_elements, const Complex* vector_x) const override {
        double result = 0.0;
        check_cublas_error(cublasDznrm2(cublas_handle_,
                                        number_of_elements,
                                        reinterpret_cast<const cuDoubleComplex*>(vector_x),
                                        1,
                                        &result),
                           "cublasDznrm2");
        return result;
    }

    /**
     * Scale a complex double-precision vector in-place via cublasZscal.
     */
    void scale(int number_of_elements, Complex scaling_factor, Complex* vector_x) const override {
        auto cuda_alpha = make_cuda_double_complex(scaling_factor);
        check_cublas_error(cublasZscal(cublas_handle_,
                                       number_of_elements,
                                       &cuda_alpha,
                                       reinterpret_cast<cuDoubleComplex*>(vector_x),
                                       1),
                           "cublasZscal");
    }

    /**
     * Scaled vector addition (ZAXPY) via cublasZaxpy.
     */
    void axpy(int number_of_elements,
              Complex scaling_factor,
              const Complex* vector_x,
              Complex* vector_y) const override {
        auto cuda_alpha = make_cuda_double_complex(scaling_factor);
        check_cublas_error(cublasZaxpy(cublas_handle_,
                                       number_of_elements,
                                       &cuda_alpha,
                                       reinterpret_cast<const cuDoubleComplex*>(vector_x),
                                       1,
                                       reinterpret_cast<cuDoubleComplex*>(vector_y),
                                       1),
                           "cublasZaxpy");
    }

    // ── Complex single-precision (FP32) ──────────────────────────────────

    /**
     * Complex single-precision CGEMM via cublasCgemm.
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
        auto cuda_alpha = make_cuda_float_complex(alpha);
        auto cuda_beta = make_cuda_float_complex(beta);
        check_cublas_error(cublasCgemm(cublas_handle_,
                                       translate_transpose(transpose_a),
                                       translate_transpose(transpose_b),
                                       m,
                                       n,
                                       k,
                                       &cuda_alpha,
                                       reinterpret_cast<const cuFloatComplex*>(matrix_a),
                                       leading_dimension_a,
                                       reinterpret_cast<const cuFloatComplex*>(matrix_b),
                                       leading_dimension_b,
                                       &cuda_beta,
                                       reinterpret_cast<cuFloatComplex*>(matrix_c),
                                       leading_dimension_c),
                           "cublasCgemm");
    }

    /**
     * Complex single-precision conjugated dot product via cublasCdotc.
     */
    ComplexSingle dot_product_single(int number_of_elements,
                                     const ComplexSingle* vector_x,
                                     const ComplexSingle* vector_y) const override {
        cuFloatComplex result;
        check_cublas_error(cublasCdotc(cublas_handle_,
                                       number_of_elements,
                                       reinterpret_cast<const cuFloatComplex*>(vector_x),
                                       1,
                                       reinterpret_cast<const cuFloatComplex*>(vector_y),
                                       1,
                                       &result),
                           "cublasCdotc");
        return ComplexSingle(result.x, result.y);
    }

    /**
     * Euclidean norm of a complex single-precision vector via cublasScnrm2.
     */
    float norm_single(int number_of_elements, const ComplexSingle* vector_x) const override {
        float result = 0.0f;
        check_cublas_error(cublasScnrm2(cublas_handle_,
                                        number_of_elements,
                                        reinterpret_cast<const cuFloatComplex*>(vector_x),
                                        1,
                                        &result),
                           "cublasScnrm2");
        return result;
    }

    // ── Real double-precision (FP64) — for gamma-only ────────────────────

    /**
     * Real double-precision DGEMM via cublasDgemm.
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
        check_cublas_error(cublasDgemm(cublas_handle_,
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
                           "cublasDgemm");
    }

  private:
    cublasHandle_t cublas_handle_ = nullptr;

    /**
     * Convert std::complex<double> to cuDoubleComplex.
     */
    static cuDoubleComplex make_cuda_double_complex(Complex value) {
        return make_cuDoubleComplex(value.real(), value.imag());
    }

    /**
     * Convert std::complex<float> to cuFloatComplex.
     */
    static cuFloatComplex make_cuda_float_complex(ComplexSingle value) {
        return make_cuFloatComplex(value.real(), value.imag());
    }
};

std::unique_ptr<GPUBLAS>
create_cuda_gpu_blas(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<CudaGPUBLAS>(std::move(device_backend));
}

} // namespace blas
} // namespace gpu
} // namespace execution
} // namespace q3
