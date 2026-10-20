#include <gtest/gtest.h>

#include <complex>
#include <vector>

#include "layer_0_core/math/linear_algebra/blas.hpp"

using Complex = std::complex<double>;
using namespace q3::blas;

// ── q3_complex_conjugate_dot_product tests ──────────────────────────

TEST(BlasWrappers, ConjugateDotProductZeroLengthReturnsZero) {
    Complex vector_x[] = {Complex(1.0, 2.0)};
    Complex vector_y[] = {Complex(3.0, 4.0)};
    Complex result = q3_complex_conjugate_dot_product(0, vector_x, vector_y);
    EXPECT_DOUBLE_EQ(result.real(), 0.0);
    EXPECT_DOUBLE_EQ(result.imag(), 0.0);
}

TEST(BlasWrappers, ConjugateDotProductWithSelfReturnsSumOfSquaredModuli) {
    // dotc(x, x) = sum of |x_i|^2, which is always real and non-negative
    std::vector<Complex> vector_x = {Complex(1.0, 2.0), Complex(3.0, -1.0), Complex(0.0, 4.0)};
    // |1+2i|^2 = 5, |3-i|^2 = 10, |4i|^2 = 16 => total = 31
    Complex result = q3_complex_conjugate_dot_product(
        static_cast<int>(vector_x.size()), vector_x.data(), vector_x.data());
    EXPECT_NEAR(result.real(), 31.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST(BlasWrappers, ConjugateDotProductOfOrthogonalVectorsIsZero) {
    // Standard basis vectors: {1,0} and {0,1}
    std::vector<Complex> vector_x = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
    std::vector<Complex> vector_y = {Complex(0.0, 0.0), Complex(1.0, 0.0)};
    Complex result = q3_complex_conjugate_dot_product(
        static_cast<int>(vector_x.size()), vector_x.data(), vector_y.data());
    EXPECT_NEAR(result.real(), 0.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST(BlasWrappers, ConjugateDotProductConjugationSymmetry) {
    // dotc(x,y) = conj(dotc(y,x))
    std::vector<Complex> vector_x = {Complex(1.0, 2.0), Complex(-1.0, 3.0)};
    std::vector<Complex> vector_y = {Complex(4.0, -1.0), Complex(2.0, 1.0)};

    Complex dot_xy = q3_complex_conjugate_dot_product(
        static_cast<int>(vector_x.size()), vector_x.data(), vector_y.data());
    Complex dot_yx = q3_complex_conjugate_dot_product(
        static_cast<int>(vector_y.size()), vector_y.data(), vector_x.data());

    EXPECT_NEAR(dot_xy.real(), dot_yx.real(), 1e-14);
    EXPECT_NEAR(dot_xy.imag(), -dot_yx.imag(), 1e-14);
}

TEST(BlasWrappers, ConjugateDotProductKnownValues) {
    /**
     * dotc({1+i, 2-i}, {3+2i, 1+i})
     * = conj(1+i)*(3+2i) + conj(2-i)*(1+i)
     * = (1-i)*(3+2i) + (2+i)*(1+i)
     * = (3+2i-3i-2i^2) + (2+2i+i+i^2)
     * = (3+2i-3i+2) + (2+2i+i-1)
     * = (5-i) + (1+3i)
     * = 6+2i
     */
    std::vector<Complex> vector_x = {Complex(1.0, 1.0), Complex(2.0, -1.0)};
    std::vector<Complex> vector_y = {Complex(3.0, 2.0), Complex(1.0, 1.0)};
    Complex result = q3_complex_conjugate_dot_product(
        static_cast<int>(vector_x.size()), vector_x.data(), vector_y.data());
    EXPECT_NEAR(result.real(), 6.0, 1e-14);
    EXPECT_NEAR(result.imag(), 2.0, 1e-14);
}

TEST(BlasWrappers, ConjugateDotProductSingleElement) {
    // dotc({3+4i}, {3+4i}) = conj(3+4i)*(3+4i) = (3-4i)*(3+4i) = 9+16 = 25
    std::vector<Complex> vector_x = {Complex(3.0, 4.0)};
    Complex result = q3_complex_conjugate_dot_product(
        static_cast<int>(vector_x.size()), vector_x.data(), vector_x.data());
    EXPECT_NEAR(result.real(), 25.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

// ── ZGEMM test ──────────────────────────────────────────────────────

TEST(BlasWrappers, ZgemmIdentityMatrixTimesVector) {
    /**
     * C = 1.0 * I * B + 0.0 * C => C = B
     * Use a 2x2 identity and a 2x1 "matrix" (column vector)
     */
    int matrix_dimension = 2;
    int number_of_columns = 1;

    // Column-major 2x2 identity
    std::vector<Complex> identity_matrix = {
        Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};

    std::vector<Complex> input_vector = {Complex(3.0, 1.0), Complex(-2.0, 4.0)};
    std::vector<Complex> result_vector(2, Complex(0.0, 0.0));

    Complex alpha(1.0, 0.0);
    Complex beta(0.0, 0.0);

    zgemm_("N",
           "N",
           &matrix_dimension,
           &number_of_columns,
           &matrix_dimension,
           &alpha,
           identity_matrix.data(),
           &matrix_dimension,
           input_vector.data(),
           &matrix_dimension,
           &beta,
           result_vector.data(),
           &matrix_dimension);

    EXPECT_NEAR(result_vector[0].real(), 3.0, 1e-14);
    EXPECT_NEAR(result_vector[0].imag(), 1.0, 1e-14);
    EXPECT_NEAR(result_vector[1].real(), -2.0, 1e-14);
    EXPECT_NEAR(result_vector[1].imag(), 4.0, 1e-14);
}

// ── ZGEMV test ──────────────────────────────────────────────────────

TEST(BlasWrappers, ZgemvIdentityMatrixTimesVector) {
    // y = 1.0 * I * x + 0.0 * y => y = x
    int matrix_rows = 2;
    int matrix_columns = 2;

    std::vector<Complex> identity_matrix = {
        Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};

    std::vector<Complex> input_vector = {Complex(5.0, -3.0), Complex(1.0, 7.0)};
    std::vector<Complex> result_vector(2, Complex(0.0, 0.0));

    Complex alpha(1.0, 0.0);
    Complex beta(0.0, 0.0);
    int increment = 1;

    zgemv_("N",
           &matrix_rows,
           &matrix_columns,
           &alpha,
           identity_matrix.data(),
           &matrix_rows,
           input_vector.data(),
           &increment,
           &beta,
           result_vector.data(),
           &increment);

    EXPECT_NEAR(result_vector[0].real(), 5.0, 1e-14);
    EXPECT_NEAR(result_vector[0].imag(), -3.0, 1e-14);
    EXPECT_NEAR(result_vector[1].real(), 1.0, 1e-14);
    EXPECT_NEAR(result_vector[1].imag(), 7.0, 1e-14);
}

// ── ZAXPY test ──────────────────────────────────────────────────────

TEST(BlasWrappers, ZaxpyAlphaTimesXPlusY) {
    /**
     * y = alpha * x + y
     * alpha = (2, 1), x = {(1, 0), (0, 1)}, y = {(3, -1), (2, 2)}
     * alpha*x[0] = (2+i)*(1+0i) = (2, 1)
     * alpha*x[1] = (2+i)*(0+i) = (2i + i^2) = (-1, 2)
     * y[0] = (3,-1) + (2,1) = (5, 0)
     * y[1] = (2, 2) + (-1, 2) = (1, 4)
     */
    int vector_length = 2;
    Complex alpha(2.0, 1.0);
    std::vector<Complex> vector_x = {Complex(1.0, 0.0), Complex(0.0, 1.0)};
    std::vector<Complex> vector_y = {Complex(3.0, -1.0), Complex(2.0, 2.0)};
    int increment = 1;

    zaxpy_(&vector_length, &alpha, vector_x.data(), &increment, vector_y.data(), &increment);

    EXPECT_NEAR(vector_y[0].real(), 5.0, 1e-14);
    EXPECT_NEAR(vector_y[0].imag(), 0.0, 1e-14);
    EXPECT_NEAR(vector_y[1].real(), 1.0, 1e-14);
    EXPECT_NEAR(vector_y[1].imag(), 4.0, 1e-14);
}
