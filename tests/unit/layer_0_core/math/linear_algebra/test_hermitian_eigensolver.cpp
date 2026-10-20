#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/hermitian_eigensolver.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/type_aliases.hpp"

using namespace q3;

namespace {

/// [[2, i], [-i, 2]]: eigenvalues 1 and 3.
DenseMatrix<Complex> two_by_two() {
    DenseMatrix<Complex> matrix(2, 2, Complex(0.0, 0.0));
    matrix(0, 0) = Complex(2.0, 0.0);
    matrix(0, 1) = Complex(0.0, 1.0);
    matrix(1, 0) = Complex(0.0, -1.0);
    matrix(1, 1) = Complex(2.0, 0.0);
    return matrix;
}

} // namespace

TEST(HermitianEigensolver, ReturnsAscendingEigenvaluesWithTheirVectorsInColumns) {
    const HermitianEigenDecomposition decomposition =
        HermitianEigensolver("test").solve(two_by_two());
    ASSERT_EQ(decomposition.eigenvalues.size(), 2u);
    EXPECT_NEAR(decomposition.eigenvalues[0], 1.0, 1.0e-12);
    EXPECT_NEAR(decomposition.eigenvalues[1], 3.0, 1.0e-12);

    // A v = lambda v for each column, and V diag(lambda) V^dagger rebuilds A.
    const DenseMatrix<Complex> matrix = two_by_two();
    const DenseMatrix<Complex>& vectors = decomposition.eigenvectors;
    const DenseMatrix<Complex> rebuilt = multiply(MatrixOperation::None,
                                                  vectors.scaled_columns(decomposition.eigenvalues),
                                                  MatrixOperation::ConjugateTranspose,
                                                  vectors);
    for (int row = 0; row < 2; row++) {
        for (int column = 0; column < 2; column++) {
            EXPECT_NEAR(std::abs(rebuilt(row, column) - matrix(row, column)), 0.0, 1.0e-12);
        }
    }
    const DenseMatrix<Complex> gram =
        multiply(MatrixOperation::ConjugateTranspose, vectors, MatrixOperation::None, vectors);
    EXPECT_NEAR(std::abs(gram(0, 0) - Complex(1.0, 0.0)), 0.0, 1.0e-12);
    EXPECT_NEAR(std::abs(gram(0, 1)), 0.0, 1.0e-12);
}

TEST(HermitianEigensolver, AnEmptyMatrixHasNoEigenvalues) {
    const HermitianEigenDecomposition decomposition =
        HermitianEigensolver("test").solve(DenseMatrix<Complex>());
    EXPECT_TRUE(decomposition.eigenvalues.empty());
    EXPECT_TRUE(decomposition.eigenvectors.empty());
}

TEST(HermitianEigensolver, RefusesANonSquareMatrixNamingItsContext) {
    try {
        HermitianEigensolver("SubspaceDisentangler: Z(k)")
            .solve(DenseMatrix<Complex>(2, 3, Complex(0.0, 0.0)));
        FAIL() << "a 2 x 3 matrix has no Hermitian eigenproblem";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("SubspaceDisentangler: Z(k)"), std::string::npos);
        EXPECT_NE(message.find("2 x 3"), std::string::npos);
    }
}
