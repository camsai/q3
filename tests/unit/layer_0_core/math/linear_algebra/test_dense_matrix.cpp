#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/type_aliases.hpp"

using namespace q3;

TEST(DenseMatrix, StoresColumnMajor) {
    DenseMatrix<Real> matrix(2, 3, 0.0);
    matrix(0, 0) = 1.0;
    matrix(1, 0) = 2.0;
    matrix(0, 1) = 3.0;
    matrix(1, 1) = 4.0;
    matrix(0, 2) = 5.0;
    matrix(1, 2) = 6.0;
    // Element (row, column) sits at column * rows + row: a column is contiguous.
    const std::vector<Real> expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_EQ(matrix.raw(), expected);
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.columns(), 3);
    EXPECT_EQ(matrix.number_of_elements(), 6u);
    EXPECT_FALSE(matrix.empty());
    EXPECT_TRUE(DenseMatrix<Real>().empty());
}

TEST(DenseMatrix, AdoptsABufferOfTheRightSizeAndRefusesTheWrongOne) {
    const DenseMatrix<Real> matrix(2, 2, std::vector<Real>{1.0, 2.0, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(matrix(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 3.0);
    EXPECT_THROW(DenseMatrix<Real>(2, 2, std::vector<Real>{1.0, 2.0, 3.0}), std::invalid_argument);
    EXPECT_THROW(DenseMatrix<Real>(-1, 2, 0.0), std::invalid_argument);
}

TEST(DenseMatrix, RefusesAnElementOutsideItself) {
    DenseMatrix<Real> matrix(2, 2, 0.0);
    EXPECT_THROW(matrix(2, 0), std::out_of_range);
    EXPECT_THROW(matrix(0, -1), std::out_of_range);
}

TEST(DenseMatrix, IdentityAndEquality) {
    const DenseMatrix<Complex> identity = DenseMatrix<Complex>::identity(3);
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            EXPECT_EQ(identity(row, column), Complex(row == column ? 1.0 : 0.0, 0.0));
        }
    }
    DenseMatrix<Complex> other = identity;
    EXPECT_TRUE(identity == other);
    other(0, 1) = Complex(0.5, 0.0);
    EXPECT_TRUE(identity != other);
    EXPECT_TRUE(identity != DenseMatrix<Complex>::identity(2));
}

TEST(DenseMatrix, AdjointConjugatesAndTransposes) {
    DenseMatrix<Complex> matrix(2, 1, Complex(0.0, 0.0));
    matrix(0, 0) = Complex(1.0, 2.0);
    matrix(1, 0) = Complex(3.0, -4.0);
    const DenseMatrix<Complex> adjoint = matrix.adjoint();
    EXPECT_EQ(adjoint.rows(), 1);
    EXPECT_EQ(adjoint.columns(), 2);
    EXPECT_EQ(adjoint(0, 0), Complex(1.0, -2.0));
    EXPECT_EQ(adjoint(0, 1), Complex(3.0, 4.0));
    EXPECT_DOUBLE_EQ(matrix.frobenius_norm_squared(), 1.0 + 4.0 + 9.0 + 16.0);
}

TEST(DenseMatrix, SubmatrixTakesTheRowsAndColumnsNamedInOrder) {
    DenseMatrix<Real> matrix(3, 3, 0.0);
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            matrix(row, column) = 10.0 * row + column;
        }
    }
    const DenseMatrix<Real> picked = matrix.submatrix({2, 0}, {1});
    EXPECT_EQ(picked.rows(), 2);
    EXPECT_EQ(picked.columns(), 1);
    EXPECT_DOUBLE_EQ(picked(0, 0), 21.0);
    EXPECT_DOUBLE_EQ(picked(1, 0), 1.0);
    EXPECT_THROW(matrix.submatrix({3}, {0}), std::out_of_range);
}

TEST(DenseMatrix, ScalesRowsAndColumnsByOneFactorEach) {
    DenseMatrix<Complex> matrix(2, 2, Complex(1.0, 0.0));
    const DenseMatrix<Complex> rows = matrix.scaled_rows(std::vector<Real>{2.0, 3.0});
    EXPECT_EQ(rows(0, 1), Complex(2.0, 0.0));
    EXPECT_EQ(rows(1, 0), Complex(3.0, 0.0));
    const DenseMatrix<Complex> columns =
        matrix.scaled_columns(std::vector<Complex>{Complex(0.0, 1.0), Complex(-1.0, 0.0)});
    EXPECT_EQ(columns(1, 0), Complex(0.0, 1.0));
    EXPECT_EQ(columns(0, 1), Complex(-1.0, 0.0));
    EXPECT_THROW(matrix.scaled_rows(std::vector<Real>{1.0}), std::invalid_argument);
    EXPECT_THROW(matrix.scaled_columns(std::vector<Real>{1.0, 2.0, 3.0}), std::invalid_argument);
}

TEST(DenseMatrix, ScaledMultipliesEveryElementAndLeavesTheOriginal) {
    DenseMatrix<Complex> matrix(2, 2, Complex(0.0, 0.0));
    matrix(0, 0) = Complex(1.0, 2.0);
    matrix(1, 0) = Complex(-3.0, 0.5);
    matrix(0, 1) = Complex(0.0, -1.0);

    const DenseMatrix<Complex> by_real = matrix.scaled(2.0);
    EXPECT_EQ(by_real(0, 0), Complex(2.0, 4.0));
    EXPECT_EQ(by_real(1, 0), Complex(-6.0, 1.0));
    EXPECT_EQ(matrix(0, 0), Complex(1.0, 2.0)) << "scaled returns a copy";

    /// By -i, which is how an anti-Hermitian generator becomes the Hermitian it is solved as.
    const DenseMatrix<Complex> by_complex = matrix.scaled(Complex(0.0, -1.0));
    EXPECT_EQ(by_complex(0, 0), Complex(2.0, -1.0));
    EXPECT_EQ(by_complex(0, 1), Complex(-1.0, 0.0));
}

TEST(DenseMatrix, AddScaledAccumulatesInPlaceAndRefusesAnotherShape) {
    DenseMatrix<Complex> accumulated(2, 1, Complex(1.0, 1.0));
    DenseMatrix<Complex> other(2, 1, Complex(0.0, 0.0));
    other(0, 0) = Complex(2.0, 0.0);
    other(1, 0) = Complex(0.0, 4.0);

    accumulated.add_scaled(0.5, other).add_scaled(-1.0, other);
    EXPECT_EQ(accumulated(0, 0), Complex(0.0, 1.0));
    EXPECT_EQ(accumulated(1, 0), Complex(1.0, -1.0));

    const DenseMatrix<Complex> wider(2, 2, Complex(1.0, 0.0));
    EXPECT_THROW(accumulated.add_scaled(1.0, wider), std::invalid_argument);
}

TEST(DenseMatrix, TheRealInnerProductIsTheRealPartOfTheTraceOfTheAdjointProduct) {
    DenseMatrix<Complex> left(2, 2, Complex(0.0, 0.0));
    DenseMatrix<Complex> right(2, 2, Complex(0.0, 0.0));
    left(0, 0) = Complex(1.0, 2.0);
    left(1, 1) = Complex(0.0, -3.0);
    right(0, 0) = Complex(4.0, -1.0);
    right(1, 1) = Complex(2.0, 5.0);

    /// Re[(1 - 2i)(4 - i)] + Re[(3i)(2 + 5i)] = 2 - 15.
    EXPECT_DOUBLE_EQ(left.real_inner_product(right), -13.0);
    EXPECT_DOUBLE_EQ(right.real_inner_product(left), -13.0) << "symmetric in its arguments";
    EXPECT_DOUBLE_EQ(left.real_inner_product(left), left.frobenius_norm_squared());

    const DenseMatrix<Complex> other_shape(2, 1, Complex(1.0, 0.0));
    EXPECT_THROW((void)left.real_inner_product(other_shape), std::invalid_argument);
}
