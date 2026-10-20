#pragma once

/**
 * MatrixOperation — what a product applies to each of its operands before
 * multiplying: the matrix as it stands, or its conjugate transpose.
 *
 * The flag BLAS spells "N" and "C". Named here so a call reads as the algebra
 * it performs -- multiply(ConjugateTranspose, left, None, right) is left^dagger
 * right -- rather than as a letter whose meaning depends on the storage order
 * the reader has in mind.
 */

namespace q3 {

enum class MatrixOperation { None, ConjugateTranspose };

/// The letter BLAS takes for the operation.
[[nodiscard]] inline const char* blas_transpose_flag(MatrixOperation operation) {
    return operation == MatrixOperation::ConjugateTranspose ? "C" : "N";
}

} // namespace q3
