/**
 * MetalGPUBLAS — GPUBLAS implementation using Metal Performance Shaders + Accelerate.
 *
 * FP32 GEMM (gemm_single) runs on the Apple Silicon GPU via MPSMatrixMultiplication
 * for matrices above a size threshold. Complex GEMM is decomposed into 4 real GEMMs
 * using the split-complex approach:
 *   C_r = alpha_r * (A_r * B_r - A_i * B_i) + beta_r * C_r
 *   C_i = alpha_r * (A_r * B_i + A_i * B_r) + beta_r * C_i
 *
 * All 4 real GEMMs are encoded into a single Metal command buffer for minimal
 * GPU dispatch overhead.
 *
 * FP64 operations and FP32 vector operations stay on CPU via Accelerate's
 * cblas_* routines. The dispatch overhead would exceed compute time for
 * vector operations.
 *
 * Column-major ↔ row-major conversion for MPS:
 *   C = op(A) * op(B) [col-major]
 *   ≡ C^T = op(B)^T * op(A)^T [row-major]
 * So in MPS: leftMatrix = B data, rightMatrix = A data (swapped).
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "layer_0_core/execution/gpu/device/metal.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace q3 {
namespace execution {
namespace gpu {
namespace blas {

namespace {

/**
 * Translate a single-character transpose flag to the Accelerate CBLAS enum.
 */
CBLAS_TRANSPOSE translate_transpose(char transpose_flag) {
    switch (transpose_flag) {
        case 'N':
        case 'n':
            return CblasNoTrans;
        case 'T':
        case 't':
            return CblasTrans;
        case 'C':
        case 'c':
            return CblasConjTrans;
        default:
            return CblasNoTrans;
    }
}

/**
 * Deinterleave a complex single-precision matrix into split real/imag arrays.
 *
 * Source is column-major with leading dimension source_leading_dimension.
 * Destination arrays are packed (leading dimension = number_of_rows).
 */
void deinterleave_complex_to_split(const ComplexSingle* source,
                                   int number_of_rows,
                                   int number_of_columns,
                                   int source_leading_dimension,
                                   float* destination_real,
                                   float* destination_imag) {
    if (source_leading_dimension == number_of_rows) {
        // Packed — single vDSP call for entire matrix
        DSPSplitComplex split;
        split.realp = destination_real;
        split.imagp = destination_imag;
        vDSP_ctoz(reinterpret_cast<const DSPComplex*>(source), 2,
                  &split, 1,
                  static_cast<vDSP_Length>(number_of_rows * number_of_columns));
    } else {
        // Non-packed — process column by column
        for (int column_index = 0; column_index < number_of_columns; ++column_index) {
            DSPSplitComplex column_split;
            column_split.realp = destination_real + column_index * number_of_rows;
            column_split.imagp = destination_imag + column_index * number_of_rows;
            vDSP_ctoz(
                reinterpret_cast<const DSPComplex*>(source + column_index * source_leading_dimension),
                2, &column_split, 1,
                static_cast<vDSP_Length>(number_of_rows));
        }
    }
}

/**
 * Interleave split real/imag arrays back into a complex single-precision matrix.
 *
 * Source arrays are packed (leading dimension = number_of_rows).
 * Destination is column-major with leading dimension destination_leading_dimension.
 */
void interleave_split_to_complex(const float* source_real,
                                 const float* source_imag,
                                 int number_of_rows,
                                 int number_of_columns,
                                 int destination_leading_dimension,
                                 ComplexSingle* destination) {
    if (destination_leading_dimension == number_of_rows) {
        // Packed — single vDSP call
        DSPSplitComplex split;
        split.realp = const_cast<float*>(source_real);
        split.imagp = const_cast<float*>(source_imag);
        vDSP_ztoc(&split, 1,
                  reinterpret_cast<DSPComplex*>(destination), 2,
                  static_cast<vDSP_Length>(number_of_rows * number_of_columns));
    } else {
        // Non-packed — process column by column
        for (int column_index = 0; column_index < number_of_columns; ++column_index) {
            DSPSplitComplex column_split;
            column_split.realp = const_cast<float*>(source_real + column_index * number_of_rows);
            column_split.imagp = const_cast<float*>(source_imag + column_index * number_of_rows);
            vDSP_ztoc(
                &column_split, 1,
                reinterpret_cast<DSPComplex*>(destination + column_index * destination_leading_dimension),
                2,
                static_cast<vDSP_Length>(number_of_rows));
        }
    }
}

} // anonymous namespace

class MetalGPUBLAS : public GPUBLAS {
  public:
    explicit MetalGPUBLAS(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : GPUBLAS(device_backend) {
        auto* metal_backend =
            dynamic_cast<device::MetalGPUDeviceBackend*>(device_backend.get());
        if (metal_backend == nullptr) {
            throw std::runtime_error(
                "MetalGPUBLAS: device backend is not a MetalGPUDeviceBackend");
        }
        metal_device_ = metal_backend->metal_device();
        metal_command_queue_ = metal_backend->metal_command_queue();
    }

    ~MetalGPUBLAS() override = default;

    // ── Complex double-precision (FP64) — Accelerate CPU ─────────────────

    /**
     * Complex double-precision ZGEMM via cblas_zgemm (CPU).
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
        cblas_zgemm(CblasColMajor,
                    translate_transpose(transpose_a),
                    translate_transpose(transpose_b),
                    m, n, k,
                    &alpha, matrix_a, leading_dimension_a,
                    matrix_b, leading_dimension_b,
                    &beta, matrix_c, leading_dimension_c);
    }

    /**
     * Complex double-precision conjugated dot product via cblas_zdotc_sub.
     */
    Complex dot_product(int number_of_elements,
                        const Complex* vector_x,
                        const Complex* vector_y) const override {
        Complex result;
        cblas_zdotc_sub(number_of_elements, vector_x, 1, vector_y, 1, &result);
        return result;
    }

    /**
     * Euclidean norm of a complex double-precision vector via cblas_dznrm2.
     */
    double norm(int number_of_elements, const Complex* vector_x) const override {
        return cblas_dznrm2(number_of_elements, vector_x, 1);
    }

    /**
     * Scale a complex double-precision vector in-place via cblas_zscal.
     */
    void scale(int number_of_elements, Complex scaling_factor, Complex* vector_x) const override {
        cblas_zscal(number_of_elements, &scaling_factor, vector_x, 1);
    }

    /**
     * Scaled vector addition (ZAXPY) via cblas_zaxpy.
     */
    void axpy(int number_of_elements,
              Complex scaling_factor,
              const Complex* vector_x,
              Complex* vector_y) const override {
        cblas_zaxpy(number_of_elements, &scaling_factor, vector_x, 1, vector_y, 1);
    }

    // ── Complex single-precision (FP32) — MPS GPU for large, Accelerate for small ──

    /**
     * Complex single-precision CGEMM.
     *
     * For large matrices (M*N*K above threshold): decomposes into 4 real
     * FP32 GEMMs via MPSMatrixMultiplication on the Apple Silicon GPU.
     *
     * For small matrices or complex alpha/beta: falls back to cblas_cgemm
     * (Accelerate AMX on CPU).
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

        const long long total_work = static_cast<long long>(m) * n * k;
        const bool has_complex_scalars =
            (std::abs(alpha.imag()) > 1e-12f || std::abs(beta.imag()) > 1e-12f);

        if (total_work < gpu_dispatch_threshold_ || has_complex_scalars) {
            // Small matrix or complex alpha/beta — use CPU
            cblas_cgemm(CblasColMajor,
                        translate_transpose(transpose_a),
                        translate_transpose(transpose_b),
                        m, n, k,
                        &alpha, matrix_a, leading_dimension_a,
                        matrix_b, leading_dimension_b,
                        &beta, matrix_c, leading_dimension_c);
            return;
        }

        // ── MPS GPU path: decompose complex GEMM into 4 real GEMMs ──────

        // Determine stored matrix dimensions (before any transpose)
        const bool transpose_a_flag = (transpose_a != 'N' && transpose_a != 'n');
        const bool transpose_b_flag = (transpose_b != 'N' && transpose_b != 'n');
        const bool conjugate_a = (transpose_a == 'C' || transpose_a == 'c');
        const bool conjugate_b = (transpose_b == 'C' || transpose_b == 'c');

        const int stored_rows_a = transpose_a_flag ? k : m;
        const int stored_columns_a = transpose_a_flag ? m : k;
        const int stored_rows_b = transpose_b_flag ? n : k;
        const int stored_columns_b = transpose_b_flag ? k : n;

        // Compute workspace sizes (all packed, ld = stored_rows)
        const size_t size_a = static_cast<size_t>(stored_rows_a) * stored_columns_a;
        const size_t size_b = static_cast<size_t>(stored_rows_b) * stored_columns_b;
        const size_t size_c = static_cast<size_t>(m) * n;
        const size_t total_floats = 2 * (size_a + size_b + size_c);

        ensure_workspace(total_floats);

        // Compute offsets into workspace buffer
        float* workspace = static_cast<float*>(workspace_buffer_.contents);
        float* workspace_a_real = workspace;
        float* workspace_a_imag = workspace_a_real + size_a;
        float* workspace_b_real = workspace_a_imag + size_a;
        float* workspace_b_imag = workspace_b_real + size_b;
        float* workspace_c_real = workspace_b_imag + size_b;
        float* workspace_c_imag = workspace_c_real + size_c;

        // Byte offsets for MPSMatrix creation
        const size_t offset_a_real = 0;
        const size_t offset_a_imag = size_a * sizeof(float);
        const size_t offset_b_real = 2 * size_a * sizeof(float);
        const size_t offset_b_imag = (2 * size_a + size_b) * sizeof(float);
        const size_t offset_c_real = (2 * size_a + 2 * size_b) * sizeof(float);
        const size_t offset_c_imag = (2 * size_a + 2 * size_b + size_c) * sizeof(float);

        // ── Deinterleave complex matrices into split real/imag ───────────

        deinterleave_complex_to_split(
            matrix_a, stored_rows_a, stored_columns_a, leading_dimension_a,
            workspace_a_real, workspace_a_imag);

        deinterleave_complex_to_split(
            matrix_b, stored_rows_b, stored_columns_b, leading_dimension_b,
            workspace_b_real, workspace_b_imag);

        const float beta_r = beta.real();
        if (std::abs(beta_r) > 1e-12f) {
            deinterleave_complex_to_split(
                matrix_c, m, n, leading_dimension_c,
                workspace_c_real, workspace_c_imag);
        } else {
            // beta = 0 → C will be overwritten, no need to deinterleave
            std::memset(workspace_c_real, 0, size_c * sizeof(float));
            std::memset(workspace_c_imag, 0, size_c * sizeof(float));
        }

        // ── Create MPSMatrix descriptors ─────────────────────────────────

        /**
         * Column-major split arrays viewed as row-major:
         * A_r is stored_rows_a × stored_columns_a col-major
         *   → stored_columns_a × stored_rows_a row-major
         *   → MPS: rows = stored_columns_a, columns = stored_rows_a
         *          rowBytes = stored_rows_a * sizeof(float)
         */

        MPSMatrixDescriptor* descriptor_a = [MPSMatrixDescriptor
            matrixDescriptorWithDimensions:stored_columns_a
                                  columns:stored_rows_a
                                 rowBytes:stored_rows_a * sizeof(float)
                                 dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descriptor_b = [MPSMatrixDescriptor
            matrixDescriptorWithDimensions:stored_columns_b
                                  columns:stored_rows_b
                                 rowBytes:stored_rows_b * sizeof(float)
                                 dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descriptor_c = [MPSMatrixDescriptor
            matrixDescriptorWithDimensions:n
                                  columns:m
                                 rowBytes:m * sizeof(float)
                                 dataType:MPSDataTypeFloat32];

        // Create MPSMatrix objects from workspace buffer at computed offsets
        MPSMatrix* mps_a_real = [[MPSMatrix alloc] initWithBuffer:workspace_buffer_
                                                           offset:offset_a_real
                                                       descriptor:descriptor_a];
        MPSMatrix* mps_a_imag = [[MPSMatrix alloc] initWithBuffer:workspace_buffer_
                                                           offset:offset_a_imag
                                                       descriptor:descriptor_a];
        MPSMatrix* mps_b_real = [[MPSMatrix alloc] initWithBuffer:workspace_buffer_
                                                           offset:offset_b_real
                                                       descriptor:descriptor_b];
        MPSMatrix* mps_b_imag = [[MPSMatrix alloc] initWithBuffer:workspace_buffer_
                                                           offset:offset_b_imag
                                                       descriptor:descriptor_b];
        MPSMatrix* mps_c_real = [[MPSMatrix alloc] initWithBuffer:workspace_buffer_
                                                           offset:offset_c_real
                                                       descriptor:descriptor_c];
        MPSMatrix* mps_c_imag = [[MPSMatrix alloc] initWithBuffer:workspace_buffer_
                                                           offset:offset_c_imag
                                                       descriptor:descriptor_c];

        /**
         * ── Compute transpose and conjugation flags ──────────────────────
         *
         * In col-major: C = alpha * op(A) * op(B) + beta * C
         * In row-major (MPS): C^T = alpha * op(B)^T * op(A)^T + beta * C^T
         *
         * Left matrix in MPS = B data, Right matrix = A data
         *
         * For the split real arrays (already transposed to row-major by the
         * column-major → row-major reinterpretation):
         *   If op = 'N': the col-major data viewed as row-major is already
         *     the transpose → in MPS, no additional transpose needed
         *   If op = 'T' or 'C': we need MPS to transpose the row-major view
         *     (which un-transposes it back to the original orientation)
         *
         * So: MPS transposeLeft  = (transpose_b != 'N')
         *     MPS transposeRight = (transpose_a != 'N')
         */

        BOOL mps_transpose_left = transpose_b_flag ? YES : NO;
        BOOL mps_transpose_right = transpose_a_flag ? YES : NO;

        const float alpha_r = alpha.real();
        const float conjugate_sign_a = conjugate_a ? -1.0f : 1.0f;
        const float conjugate_sign_b = conjugate_b ? -1.0f : 1.0f;

        /**
         * ── Encode 4 real GEMMs into a single command buffer ─────────────
         *
         * GEMM 1: C_r = alpha_r * A_r(op) * B_r(op) + beta_r * C_r
         * GEMM 2: C_r += (-alpha_r * conj_a * conj_b) * A_i(op) * B_i(op)
         * GEMM 3: C_i = (alpha_r * conj_b) * A_r(op) * B_i(op) + beta_r * C_i
         * GEMM 4: C_i += (alpha_r * conj_a) * A_i(op) * B_r(op)
         */

        id<MTLCommandBuffer> command_buffer = [metal_command_queue_ commandBuffer];

        // GEMM 1: C_r = alpha_r * B_r^T * A_r^T + beta_r * C_r [row-major]
        {
            MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc]
                initWithDevice:metal_device_
                 transposeLeft:mps_transpose_left
                transposeRight:mps_transpose_right
                    resultRows:n
                 resultColumns:m
               interiorColumns:k
                         alpha:alpha_r
                          beta:beta_r];
            [kernel encodeToCommandBuffer:command_buffer
                              leftMatrix:mps_b_real
                             rightMatrix:mps_a_real
                            resultMatrix:mps_c_real];
        }

        // GEMM 2: C_r += (-alpha_r * conj_a * conj_b) * B_i^T * A_i^T
        {
            MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc]
                initWithDevice:metal_device_
                 transposeLeft:mps_transpose_left
                transposeRight:mps_transpose_right
                    resultRows:n
                 resultColumns:m
               interiorColumns:k
                         alpha:(-alpha_r * conjugate_sign_a * conjugate_sign_b)
                          beta:1.0];
            [kernel encodeToCommandBuffer:command_buffer
                              leftMatrix:mps_b_imag
                             rightMatrix:mps_a_imag
                            resultMatrix:mps_c_real];
        }

        // GEMM 3: C_i = (alpha_r * conj_b) * B_i^T * A_r^T + beta_r * C_i
        {
            MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc]
                initWithDevice:metal_device_
                 transposeLeft:mps_transpose_left
                transposeRight:mps_transpose_right
                    resultRows:n
                 resultColumns:m
               interiorColumns:k
                         alpha:(alpha_r * conjugate_sign_b)
                          beta:beta_r];
            [kernel encodeToCommandBuffer:command_buffer
                              leftMatrix:mps_b_imag
                             rightMatrix:mps_a_real
                            resultMatrix:mps_c_imag];
        }

        // GEMM 4: C_i += (alpha_r * conj_a) * B_r^T * A_i^T
        {
            MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc]
                initWithDevice:metal_device_
                 transposeLeft:mps_transpose_left
                transposeRight:mps_transpose_right
                    resultRows:n
                 resultColumns:m
               interiorColumns:k
                         alpha:(alpha_r * conjugate_sign_a)
                          beta:1.0];
            [kernel encodeToCommandBuffer:command_buffer
                              leftMatrix:mps_b_real
                             rightMatrix:mps_a_imag
                            resultMatrix:mps_c_imag];
        }

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        // ── Interleave result back to complex ────────────────────────────

        interleave_split_to_complex(
            workspace_c_real, workspace_c_imag, m, n, leading_dimension_c, matrix_c);
    }

    /**
     * Complex single-precision conjugated dot product via cblas_cdotc_sub.
     */
    ComplexSingle dot_product_single(int number_of_elements,
                                     const ComplexSingle* vector_x,
                                     const ComplexSingle* vector_y) const override {
        ComplexSingle result;
        cblas_cdotc_sub(number_of_elements, vector_x, 1, vector_y, 1, &result);
        return result;
    }

    /**
     * Euclidean norm of a complex single-precision vector via cblas_scnrm2.
     */
    float norm_single(int number_of_elements, const ComplexSingle* vector_x) const override {
        return cblas_scnrm2(number_of_elements, vector_x, 1);
    }

    // ── Real double-precision (FP64) — for gamma-only ────────────────────

    /**
     * Real double-precision DGEMM via cblas_dgemm.
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
        cblas_dgemm(CblasColMajor,
                    translate_transpose(transpose_a),
                    translate_transpose(transpose_b),
                    m, n, k,
                    alpha, matrix_a, leading_dimension_a,
                    matrix_b, leading_dimension_b,
                    beta, matrix_c, leading_dimension_c);
    }

  private:
    id<MTLDevice> metal_device_;
    id<MTLCommandQueue> metal_command_queue_;

    // Workspace MTLBuffer for split-complex GEMM (lazily allocated, grows as needed)
    mutable id<MTLBuffer> workspace_buffer_ = nil;
    mutable size_t workspace_capacity_bytes_ = 0;

    /**
     * MPS GPU GEMM is slower than Accelerate AMX for DFT-typical
     * tall-skinny matrices (NPW × bands × basis). MPS only wins for
     * near-square matrices ≥2000³ due to deinterleave/interleave overhead
     * and 4× GEMM count. Keep MPS infrastructure but dispatch to CPU.
     * TODO: re-enable when MPSGraph supports native complex GEMM or when
     * persistent device buffers eliminate the copy overhead.
     */
    static constexpr long long gpu_dispatch_threshold_ = std::numeric_limits<long long>::max();

    /**
     * Ensure the workspace MTLBuffer has at least the specified capacity.
     *
     * Grows by 2× to amortize reallocation cost.
     */
    void ensure_workspace(size_t needed_floats) const {
        const size_t needed_bytes = needed_floats * sizeof(float);
        if (workspace_capacity_bytes_ >= needed_bytes) {
            return;
        }

        // Grow to 2× the needed size for headroom
        workspace_capacity_bytes_ = needed_bytes * 2;
        workspace_buffer_ = [metal_device_ newBufferWithLength:workspace_capacity_bytes_
                                                       options:MTLResourceStorageModeShared];
        if (workspace_buffer_ == nil) {
            throw std::runtime_error(
                "MetalGPUBLAS: failed to allocate workspace MTLBuffer of " +
                std::to_string(workspace_capacity_bytes_) + " bytes");
        }
    }
};

std::unique_ptr<GPUBLAS>
create_accelerate_gpu_blas(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<MetalGPUBLAS>(std::move(device_backend));
}

} // namespace blas
} // namespace gpu
} // namespace execution
} // namespace q3
