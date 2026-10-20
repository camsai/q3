#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/math/linear_algebra/singular_value_decomposition.hpp"
#include "layer_0_core/types/type_aliases.hpp"

using namespace q3;

TEST(SingularValueDecomposition, DecomposesATallMatrixInTheEconomyForm) {
    // Columns of length 3 and 2, orthogonal: the singular values are 3 and 2.
    DenseMatrix<Complex> matrix(3, 2, Complex(0.0, 0.0));
    matrix(0, 0) = Complex(0.0, 3.0);
    matrix(1, 1) = Complex(2.0, 0.0);
    const SingularValueDecompositionResult result =
        SingularValueDecomposition("test").compute(matrix);

    ASSERT_EQ(result.singular_values.size(), 2u);
    EXPECT_NEAR(result.singular_values[0], 3.0, 1.0e-12);
    EXPECT_NEAR(result.singular_values[1], 2.0, 1.0e-12);
    EXPECT_EQ(result.left_vectors.rows(), 3);
    EXPECT_EQ(result.left_vectors.columns(), 2);
    EXPECT_EQ(result.right_vectors_adjoint.rows(), 2);
    EXPECT_EQ(result.right_vectors_adjoint.columns(), 2);

    // U S V^dagger rebuilds the matrix.
    const DenseMatrix<Complex> rebuilt =
        multiply(MatrixOperation::None,
                 result.left_vectors.scaled_columns(result.singular_values),
                 MatrixOperation::None,
                 result.right_vectors_adjoint);
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 2; column++) {
            EXPECT_NEAR(std::abs(rebuilt(row, column) - matrix(row, column)), 0.0, 1.0e-12);
        }
    }
}

TEST(SingularValueDecomposition, RefusesAMatrixWithNoRowsOrColumns) {
    EXPECT_THROW(SingularValueDecomposition("test").compute(DenseMatrix<Complex>()),
                 std::invalid_argument);
}
