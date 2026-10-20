#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/type_aliases.hpp"

using namespace q3;

namespace {

/// The product by hand, so the BLAS call is checked against the definition.
DenseMatrix<Complex> by_hand(const DenseMatrix<Complex>& left, const DenseMatrix<Complex>& right) {
    DenseMatrix<Complex> result(left.rows(), right.columns(), Complex(0.0, 0.0));
    for (int row = 0; row < left.rows(); row++) {
        for (int column = 0; column < right.columns(); column++) {
            for (int inner = 0; inner < left.columns(); inner++) {
                result(row, column) += left(row, inner) * right(inner, column);
            }
        }
    }
    return result;
}

DenseMatrix<Complex> sample(int rows, int columns, Real seed) {
    DenseMatrix<Complex> matrix(rows, columns, Complex(0.0, 0.0));
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            matrix(row, column) = Complex(seed + row - 0.5 * column, 0.25 * row * column - seed);
        }
    }
    return matrix;
}

void expect_near(const DenseMatrix<Complex>& actual, const DenseMatrix<Complex>& expected) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.columns(), expected.columns());
    for (int row = 0; row < actual.rows(); row++) {
        for (int column = 0; column < actual.columns(); column++) {
            EXPECT_NEAR(std::abs(actual(row, column) - expected(row, column)), 0.0, 1.0e-12)
                << "(" << row << ", " << column << ")";
        }
    }
}

} // namespace

TEST(DenseMatrixProduct, MultipliesTwoMatricesAsTheyStand) {
    const DenseMatrix<Complex> left = sample(2, 3, 1.0);
    const DenseMatrix<Complex> right = sample(3, 4, -2.0);
    expect_near(multiply(MatrixOperation::None, left, MatrixOperation::None, right),
                by_hand(left, right));
}

TEST(DenseMatrixProduct, ConjugateTransposesEitherOperand) {
    const DenseMatrix<Complex> left = sample(3, 2, 0.5);
    const DenseMatrix<Complex> right = sample(3, 4, 2.0);
    expect_near(multiply(MatrixOperation::ConjugateTranspose, left, MatrixOperation::None, right),
                by_hand(left.adjoint(), right));
    const DenseMatrix<Complex> other = sample(4, 3, -1.0);
    expect_near(
        multiply(MatrixOperation::None, left.adjoint(), MatrixOperation::ConjugateTranspose, other),
        by_hand(left.adjoint(), other.adjoint()));
}

TEST(DenseMatrixProduct, ScalesByAlphaAndAccumulatesIntoTheResult) {
    const DenseMatrix<Complex> left = sample(2, 2, 1.5);
    const DenseMatrix<Complex> right = sample(2, 2, -0.5);
    const Complex alpha(0.0, 2.0);
    DenseMatrix<Complex> accumulated = DenseMatrix<Complex>::identity(2);
    multiply_add(MatrixOperation::None, left, MatrixOperation::None, right, alpha, accumulated);

    DenseMatrix<Complex> expected = by_hand(left, right).scaled(alpha);
    expected(0, 0) += Complex(1.0, 0.0);
    expected(1, 1) += Complex(1.0, 0.0);
    expect_near(accumulated, expected);
    expect_near(multiply(MatrixOperation::None, left, MatrixOperation::None, right, alpha),
                by_hand(left, right).scaled_rows(std::vector<Complex>{alpha, alpha}));
}

TEST(DenseMatrixProduct, RefusesInnerDimensionsThatDoNotMatch) {
    const DenseMatrix<Complex> left = sample(2, 3, 1.0);
    const DenseMatrix<Complex> right = sample(2, 2, 1.0);
    EXPECT_THROW(multiply(MatrixOperation::None, left, MatrixOperation::None, right),
                 std::invalid_argument);
    // Transposing the left operand makes the inner dimension 2, which matches.
    EXPECT_NO_THROW(
        multiply(MatrixOperation::ConjugateTranspose, left, MatrixOperation::None, right));
    DenseMatrix<Complex> wrong_shape(3, 3, Complex(0.0, 0.0));
    EXPECT_THROW(multiply_add(MatrixOperation::ConjugateTranspose,
                              left,
                              MatrixOperation::None,
                              right,
                              Complex(1.0, 0.0),
                              wrong_shape),
                 std::invalid_argument);
}

TEST(DenseMatrixProduct, AnEmptyOperandGivesAnEmptyOrUnchangedResult) {
    const DenseMatrix<Complex> empty_columns(2, 0, Complex(0.0, 0.0));
    const DenseMatrix<Complex> empty_rows(0, 3, Complex(0.0, 0.0));
    const DenseMatrix<Complex> product =
        multiply(MatrixOperation::None, empty_columns, MatrixOperation::None, empty_rows);
    EXPECT_EQ(product.rows(), 2);
    EXPECT_EQ(product.columns(), 3);
    for (const Complex& value : product.raw()) {
        EXPECT_EQ(value, Complex(0.0, 0.0));
    }
}
