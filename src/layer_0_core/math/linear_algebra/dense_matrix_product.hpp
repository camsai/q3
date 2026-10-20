#pragma once

/**
 * Dense matrix products over BLAS: alpha * op(left) * op(right), with each
 * operand taken as it stands or conjugate-transposed.
 *
 * A product of two column-major blocks is one call here, which is what makes
 * the shapes checkable in one place -- an inner dimension that does not match
 * is refused with both shapes named -- and what puts the arithmetic in zgemm
 * rather than in a loop per caller. zgemm accumulates in its own order, so a
 * product agrees with a hand-written triple loop to the rounding of a few
 * operations, not to the bit.
 */

#include <stdexcept>
#include <string>

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

namespace internal::dense_matrix_product {

/// The rows and columns of op(matrix).
struct OperandShape {
    int rows;
    int columns;
};

[[nodiscard]] inline OperandShape shape_of(MatrixOperation operation,
                                           const DenseMatrix<Complex>& matrix) {
    if (operation == MatrixOperation::ConjugateTranspose) {
        return {matrix.columns(), matrix.rows()};
    }
    return {matrix.rows(), matrix.columns()};
}

[[nodiscard]] inline std::string describe(MatrixOperation operation,
                                          const DenseMatrix<Complex>& matrix) {
    const std::string shape =
        std::to_string(matrix.rows()) + " x " + std::to_string(matrix.columns());
    return operation == MatrixOperation::ConjugateTranspose ? "the adjoint of a " + shape
                                                            : "a " + shape;
}

} // namespace internal::dense_matrix_product

/**
 * result = alpha * op(left) * op(right) + result. The result must already have
 * the product's shape; an empty operand dimension leaves it as it was.
 */
inline void multiply_add(MatrixOperation left_operation,
                         const DenseMatrix<Complex>& left,
                         MatrixOperation right_operation,
                         const DenseMatrix<Complex>& right,
                         Complex alpha,
                         DenseMatrix<Complex>& result) {
    using internal::dense_matrix_product::describe;
    using internal::dense_matrix_product::shape_of;

    const auto [rows, inner] = shape_of(left_operation, left);
    const auto [inner_of_right, columns] = shape_of(right_operation, right);
    if (inner != inner_of_right) {
        throw std::invalid_argument("multiply: " + describe(left_operation, left) +
                                    " matrix cannot multiply " + describe(right_operation, right) +
                                    " one; the inner dimensions are " + std::to_string(inner) +
                                    " and " + std::to_string(inner_of_right));
    }
    if (result.rows() != rows || result.columns() != columns) {
        throw std::invalid_argument("multiply: the product is " + std::to_string(rows) + " x " +
                                    std::to_string(columns) + " but the result given is " +
                                    std::to_string(result.rows()) + " x " +
                                    std::to_string(result.columns()));
    }
    if (rows == 0 || columns == 0 || inner == 0) {
        return;
    }

    const int leading_left = left.rows();
    const int leading_right = right.rows();
    const int leading_result = result.rows();
    const Complex beta(1.0, 0.0);
    blas::zgemm_(blas_transpose_flag(left_operation),
                 blas_transpose_flag(right_operation),
                 &rows,
                 &columns,
                 &inner,
                 &alpha,
                 left.data(),
                 &leading_left,
                 right.data(),
                 &leading_right,
                 &beta,
                 result.data(),
                 &leading_result);
}

/// alpha * op(left) * op(right), as a new matrix of the product's shape.
[[nodiscard]] inline DenseMatrix<Complex> multiply(MatrixOperation left_operation,
                                                   const DenseMatrix<Complex>& left,
                                                   MatrixOperation right_operation,
                                                   const DenseMatrix<Complex>& right,
                                                   Complex alpha = Complex(1.0, 0.0)) {
    using internal::dense_matrix_product::shape_of;
    const auto [rows, inner] = shape_of(left_operation, left);
    const auto [inner_of_right, columns] = shape_of(right_operation, right);
    (void)inner;
    (void)inner_of_right;
    DenseMatrix<Complex> result(rows, columns, Complex(0.0, 0.0));
    multiply_add(left_operation, left, right_operation, right, alpha, result);
    return result;
}

} // namespace q3
