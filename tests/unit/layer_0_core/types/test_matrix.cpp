#include <gtest/gtest.h>

#include <complex>

#include "layer_0_core/types/matrix.hpp"
#include "layer_0_core/types/type_aliases.hpp"

using namespace q3;

TEST(Matrix, DefaultConstruction) {
    Matrix<double> matrix;
    EXPECT_EQ(matrix.rows(), 0u);
    EXPECT_EQ(matrix.cols(), 0u);
    EXPECT_EQ(matrix.size(), 0u);
    EXPECT_TRUE(matrix.empty());
}

TEST(Matrix, ConstructionWithDimensions) {
    Matrix<double> matrix(3, 4);
    EXPECT_EQ(matrix.rows(), 3u);
    EXPECT_EQ(matrix.cols(), 4u);
    EXPECT_EQ(matrix.size(), 3u);
    EXPECT_FALSE(matrix.empty());
}

TEST(Matrix, ConstructionWithInitialValue) {
    Matrix<int> matrix(2, 3, 42);
    EXPECT_EQ(matrix.rows(), 2u);
    EXPECT_EQ(matrix.cols(), 3u);
    for (size_t row = 0; row < 2; row++) {
        for (size_t column = 0; column < 3; column++) {
            EXPECT_EQ(matrix(row, column), 42);
        }
    }
}

TEST(Matrix, BracketIndexing) {
    Matrix<double> matrix(2, 3, 0.0);
    matrix[0][0] = 1.0;
    matrix[0][1] = 2.0;
    matrix[0][2] = 3.0;
    matrix[1][0] = 4.0;
    matrix[1][1] = 5.0;
    matrix[1][2] = 6.0;
    EXPECT_DOUBLE_EQ(matrix[0][0], 1.0);
    EXPECT_DOUBLE_EQ(matrix[1][2], 6.0);
}

TEST(Matrix, ParenthesisIndexing) {
    Matrix<double> matrix(3, 2, 0.0);
    matrix(0, 0) = 10.0;
    matrix(2, 1) = 99.0;
    EXPECT_DOUBLE_EQ(matrix(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(matrix(2, 1), 99.0);
}

TEST(Matrix, ConstIndexing) {
    Matrix<double> matrix(2, 2, 7.5);
    const Matrix<double>& const_reference = matrix;
    EXPECT_DOUBLE_EQ(const_reference(0, 0), 7.5);
    EXPECT_DOUBLE_EQ(const_reference[1][1], 7.5);
}

TEST(Matrix, Resize) {
    Matrix<double> matrix(2, 2, 1.0);
    matrix.resize(3, 4);
    EXPECT_EQ(matrix.rows(), 3u);
    EXPECT_EQ(matrix.cols(), 4u);
}

TEST(Matrix, ResizeWithInitialValue) {
    Matrix<int> matrix(2, 2, 0);
    matrix.resize(3, 3, 99);
    EXPECT_EQ(matrix.rows(), 3u);
    EXPECT_EQ(matrix.cols(), 3u);
    for (size_t row = 0; row < 3; row++) {
        for (size_t column = 0; column < 3; column++) {
            EXPECT_EQ(matrix(row, column), 99);
        }
    }
}

TEST(Matrix, Assign) {
    Matrix<double> matrix;
    matrix.assign(4, 5, 3.14);
    EXPECT_EQ(matrix.rows(), 4u);
    EXPECT_EQ(matrix.cols(), 5u);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 3.14);
    EXPECT_DOUBLE_EQ(matrix(3, 4), 3.14);
}

TEST(Matrix, DataPointer) {
    Matrix<double> matrix(2, 3, 0.0);
    matrix(0, 0) = 1.0;
    matrix(1, 2) = 5.0;
    double* data = matrix.data();
    EXPECT_DOUBLE_EQ(data[0], 1.0);
    // Row-major: element (1, 2) is at index 1 * 3 + 2 = 5
    EXPECT_DOUBLE_EQ(data[5], 5.0);
}

TEST(Matrix, ConstDataPointer) {
    Matrix<int> matrix(2, 2, 42);
    const Matrix<int>& const_reference = matrix;
    const int* data = const_reference.data();
    EXPECT_EQ(data[0], 42);
    EXPECT_EQ(data[3], 42);
}

TEST(Matrix, RawVector) {
    Matrix<double> matrix(2, 3, 1.0);
    std::vector<double>& raw = matrix.raw();
    ASSERT_EQ(raw.size(), 6u);
    EXPECT_DOUBLE_EQ(raw[0], 1.0);
    // Modifying raw should affect matrix
    raw[0] = 99.0;
    EXPECT_DOUBLE_EQ(matrix(0, 0), 99.0);
}

TEST(Matrix, ConstRawVector) {
    Matrix<int> matrix(3, 2, 5);
    const Matrix<int>& const_reference = matrix;
    const std::vector<int>& raw = const_reference.raw();
    ASSERT_EQ(raw.size(), 6u);
    EXPECT_EQ(raw[0], 5);
}

TEST(Matrix, EmptyAfterDefaultConstruction) {
    Matrix<double> matrix;
    EXPECT_TRUE(matrix.empty());
}

TEST(Matrix, NotEmptyAfterConstruction) {
    Matrix<double> matrix(1, 1, 0.0);
    EXPECT_FALSE(matrix.empty());
}

TEST(Matrix, SizeReturnRows) {
    Matrix<double> matrix(5, 3, 0.0);
    EXPECT_EQ(matrix.size(), 5u);
}

TEST(Matrix, WorksWithComplexType) {
    Matrix<std::complex<double>> matrix(2, 2, std::complex<double>(1.0, 2.0));
    EXPECT_DOUBLE_EQ(matrix(0, 0).real(), 1.0);
    EXPECT_DOUBLE_EQ(matrix(0, 0).imag(), 2.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1).real(), 1.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1).imag(), 2.0);
}
