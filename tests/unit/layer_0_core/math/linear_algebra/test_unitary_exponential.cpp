#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/math/linear_algebra/unitary_exponential.hpp"
#include "layer_0_core/types/type_aliases.hpp"

using namespace q3;

TEST(UnitaryExponential, ExponentiatesZeroToTheIdentity) {
    const DenseMatrix<Complex> result =
        exponential_of_anti_hermitian(DenseMatrix<Complex>(3, 3, Complex(0.0, 0.0)));
    const DenseMatrix<Complex> identity = DenseMatrix<Complex>::identity(3);
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            EXPECT_NEAR(std::abs(result(row, column) - identity(row, column)), 0.0, 1.0e-14);
        }
    }
}

TEST(UnitaryExponential, ADiagonalGeneratorGivesThePhasesItNames) {
    const Real angle = 0.7;
    DenseMatrix<Complex> generator(2, 2, Complex(0.0, 0.0));
    generator(0, 0) = Complex(0.0, angle);
    generator(1, 1) = Complex(0.0, -angle);
    const DenseMatrix<Complex> result = exponential_of_anti_hermitian(generator);
    EXPECT_NEAR(std::abs(result(0, 0) - std::exp(Complex(0.0, angle))), 0.0, 1.0e-14);
    EXPECT_NEAR(std::abs(result(1, 1) - std::exp(Complex(0.0, -angle))), 0.0, 1.0e-14);
    EXPECT_NEAR(std::abs(result(0, 1)), 0.0, 1.0e-14);
}

TEST(UnitaryExponential, TheExponentialOfAnAntiHermitianGeneratorIsUnitary) {
    DenseMatrix<Complex> generator(3, 3, Complex(0.0, 0.0));
    generator(0, 1) = Complex(0.3, 0.4);
    generator(1, 0) = -std::conj(generator(0, 1));
    generator(0, 2) = Complex(-0.2, 0.1);
    generator(2, 0) = -std::conj(generator(0, 2));
    generator(1, 2) = Complex(0.05, -0.6);
    generator(2, 1) = -std::conj(generator(1, 2));
    generator(2, 2) = Complex(0.0, 0.9);

    const DenseMatrix<Complex> rotation = exponential_of_anti_hermitian(generator);
    const DenseMatrix<Complex> gram =
        multiply(MatrixOperation::ConjugateTranspose, rotation, MatrixOperation::None, rotation);
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            EXPECT_NEAR(std::abs(gram(row, column) - Complex(row == column ? 1.0 : 0.0, 0.0)),
                        0.0,
                        1.0e-12);
        }
    }
}
