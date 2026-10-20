#pragma once

/**
 * GPUBLAS — vendor-neutral GPU BLAS wrapper interface.
 *
 * Provides dual-precision (FP32 + FP64) complex and real BLAS operations
 * for GPU-accelerated computation. Each concrete implementation owns its
 * own vendor handle (e.g., cuBLAS handle or Accelerate session).
 *
 * The device backend is injected at construction and used only for
 * synchronization after scalar-returning operations. BLAS handles
 * are created/destroyed in vendor-specific .cpp files.
 *
 * No CUDA headers. No Metal headers. Vendor-neutral interface only.
 * Concrete implementations live in:
 *   - cuda.hpp / cuda.cpp   (CudaGPUBLAS — cuBLAS)
 *   - metal.hpp / metal.cpp (MetalGPUBLAS — Accelerate)
 */

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/types/single_precision.hpp"
#include "layer_0_core/types/type_aliases.hpp"

#include <complex>
#include <memory>

namespace q3 {

namespace execution {
namespace gpu {
namespace blas {

class GPUBLAS {
  public:
    /**
     * Construct a GPU BLAS wrapper backed by the given device.
     *
     * The device backend is used for synchronization. The concrete
     * vendor BLAS handle (cuBLAS, Accelerate) is created internally
     * by the vendor-specific subclass constructor.
     *
     * @param device_backend  Shared pointer to the GPU device backend.
     *                        Must not be null.
     */
    explicit GPUBLAS(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : device_backend_(std::move(device_backend)) {}

    virtual ~GPUBLAS() = default;

    // ── Complex double-precision (FP64) ──────────────────────────────────

    /**
     * Complex double-precision general matrix-matrix multiply (ZGEMM).
     * matrix_c := alpha * op(matrix_a) * op(matrix_b) + beta * matrix_c
     *
     * @param transpose_a         'N', 'T', or 'C' for op(matrix_a).
     * @param transpose_b         'N', 'T', or 'C' for op(matrix_b).
     * @param m                   Number of rows of op(matrix_a) and matrix_c.
     * @param n                   Number of columns of op(matrix_b) and matrix_c.
     * @param k                   Number of columns of op(matrix_a) / rows of op(matrix_b).
     * @param alpha               Scalar multiplier for the product.
     * @param matrix_a            Pointer to matrix A data.
     * @param leading_dimension_a Leading dimension of matrix_a.
     * @param matrix_b            Pointer to matrix B data.
     * @param leading_dimension_b Leading dimension of matrix_b.
     * @param beta                Scalar multiplier for matrix_c.
     * @param matrix_c            Pointer to matrix C data (in-out).
     * @param leading_dimension_c Leading dimension of matrix_c.
     */
    virtual void gemm(char transpose_a,
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
                      int leading_dimension_c) const = 0;

    /**
     * Complex double-precision conjugated dot product: result = x† · y.
     *
     * Synchronizes before returning the scalar result.
     *
     * @param number_of_elements  Length of both vectors.
     * @param vector_x            Left operand (conjugated).
     * @param vector_y            Right operand.
     * @return                    The complex dot product.
     */
    virtual Complex
    dot_product(int number_of_elements, const Complex* vector_x, const Complex* vector_y) const = 0;

    /**
     * Euclidean norm of a complex double-precision vector.
     *
     * Synchronizes before returning the scalar result.
     *
     * @param number_of_elements  Length of the vector.
     * @param vector_x            Input vector.
     * @return                    The real-valued norm.
     */
    virtual double norm(int number_of_elements, const Complex* vector_x) const = 0;

    /**
     * Scale a complex double-precision vector: vector_x := scaling_factor * vector_x.
     *
     * @param number_of_elements  Length of the vector.
     * @param scaling_factor      Complex scaling coefficient.
     * @param vector_x            Vector to scale (in-place).
     */
    virtual void scale(int number_of_elements, Complex scaling_factor, Complex* vector_x) const = 0;

    /**
     * Scaled vector addition: vector_y += scaling_factor * vector_x (ZAXPY).
     *
     * @param number_of_elements  Length of both vectors.
     * @param scaling_factor      Complex scaling coefficient.
     * @param vector_x            Source vector.
     * @param vector_y            Destination vector (modified in-place).
     */
    virtual void axpy(int number_of_elements,
                      Complex scaling_factor,
                      const Complex* vector_x,
                      Complex* vector_y) const = 0;

    // ── Complex single-precision (FP32) ──────────────────────────────────

    /**
     * Complex single-precision general matrix-matrix multiply (CGEMM).
     * matrix_c := alpha * op(matrix_a) * op(matrix_b) + beta * matrix_c
     */
    virtual void gemm_single(char transpose_a,
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
                             int leading_dimension_c) const = 0;

    /**
     * Complex single-precision conjugated dot product: result = x† · y.
     *
     * Synchronizes before returning the scalar result.
     *
     * @param number_of_elements  Length of both vectors.
     * @param vector_x            Left operand (conjugated).
     * @param vector_y            Right operand.
     * @return                    The complex single-precision dot product.
     */
    virtual ComplexSingle dot_product_single(int number_of_elements,
                                             const ComplexSingle* vector_x,
                                             const ComplexSingle* vector_y) const = 0;

    /**
     * Euclidean norm of a complex single-precision vector.
     *
     * Synchronizes before returning the scalar result.
     *
     * @param number_of_elements  Length of the vector.
     * @param vector_x            Input vector.
     * @return                    The real-valued norm (float).
     */
    virtual float norm_single(int number_of_elements, const ComplexSingle* vector_x) const = 0;

    // ── Real double-precision (FP64) — for gamma-only ────────────────────

    /**
     * Real double-precision general matrix-matrix multiply (DGEMM).
     * matrix_c := alpha * op(matrix_a) * op(matrix_b) + beta * matrix_c
     *
     * Used in gamma-only calculations where wavefunctions are real-valued.
     */
    virtual void dgemm(char transpose_a,
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
                       int leading_dimension_c) const = 0;

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

} // namespace blas
} // namespace gpu
} // namespace execution
} // namespace q3
