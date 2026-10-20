#include <gtest/gtest.h>

#include "layer_0_core/types/matrix_3x3.hpp"
#include "layer_0_core/types/vector_3d.hpp"

using namespace q3;

TEST(Matrix3x3, DefaultConstruction) {
    Matrix3x3 matrix;
    // Default-constructed Vector3D has all zeros
    EXPECT_DOUBLE_EQ(matrix[0].x, 0.0);
    EXPECT_DOUBLE_EQ(matrix[1].y, 0.0);
    EXPECT_DOUBLE_EQ(matrix[2].z, 0.0);
}

TEST(Matrix3x3, ParameterizedConstruction) {
    Matrix3x3 matrix(Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0), Vector3D(7.0, 8.0, 9.0));
    EXPECT_DOUBLE_EQ(matrix[0].x, 1.0);
    EXPECT_DOUBLE_EQ(matrix[1].y, 5.0);
    EXPECT_DOUBLE_EQ(matrix[2].z, 9.0);
}

TEST(Matrix3x3, RowAccess) {
    Matrix3x3 matrix(Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 2.0, 0.0), Vector3D(0.0, 0.0, 3.0));
    EXPECT_DOUBLE_EQ(matrix[0].x, 1.0);
    EXPECT_DOUBLE_EQ(matrix[1].y, 2.0);
    EXPECT_DOUBLE_EQ(matrix[2].z, 3.0);
}

TEST(Matrix3x3, MutableRowAccess) {
    Matrix3x3 matrix;
    matrix[0] = Vector3D(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(matrix[0].x, 1.0);
    EXPECT_DOUBLE_EQ(matrix[0].y, 2.0);
    EXPECT_DOUBLE_EQ(matrix[0].z, 3.0);
}

TEST(Matrix3x3, MatrixVectorProduct) {
    // Identity matrix times a vector should return the same vector
    Matrix3x3 identity(Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 1.0, 0.0), Vector3D(0.0, 0.0, 1.0));
    Vector3D vector(3.0, 4.0, 5.0);
    Vector3D result = identity * vector;
    EXPECT_DOUBLE_EQ(result.x, 3.0);
    EXPECT_DOUBLE_EQ(result.y, 4.0);
    EXPECT_DOUBLE_EQ(result.z, 5.0);
}

TEST(Matrix3x3, MatrixVectorProductGeneral) {
    Matrix3x3 matrix(Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0), Vector3D(7.0, 8.0, 9.0));
    Vector3D vector(1.0, 0.0, 0.0);
    Vector3D result = matrix * vector;
    // First column of the matrix
    EXPECT_DOUBLE_EQ(result.x, 1.0);
    EXPECT_DOUBLE_EQ(result.y, 4.0);
    EXPECT_DOUBLE_EQ(result.z, 7.0);
}

TEST(Matrix3x3, Transpose) {
    Matrix3x3 matrix(Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0), Vector3D(7.0, 8.0, 9.0));
    Matrix3x3 transposed = matrix.transpose();
    EXPECT_DOUBLE_EQ(transposed[0].x, 1.0);
    EXPECT_DOUBLE_EQ(transposed[0].y, 4.0);
    EXPECT_DOUBLE_EQ(transposed[0].z, 7.0);
    EXPECT_DOUBLE_EQ(transposed[1].x, 2.0);
    EXPECT_DOUBLE_EQ(transposed[1].y, 5.0);
    EXPECT_DOUBLE_EQ(transposed[1].z, 8.0);
    EXPECT_DOUBLE_EQ(transposed[2].x, 3.0);
    EXPECT_DOUBLE_EQ(transposed[2].y, 6.0);
    EXPECT_DOUBLE_EQ(transposed[2].z, 9.0);
}

TEST(Matrix3x3, TransposeOfTransposeIsOriginal) {
    Matrix3x3 matrix(Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0), Vector3D(7.0, 8.0, 9.0));
    Matrix3x3 double_transposed = matrix.transpose().transpose();
    for (int row_index = 0; row_index < 3; row_index++) {
        for (int column_index = 0; column_index < 3; column_index++) {
            EXPECT_DOUBLE_EQ(double_transposed[row_index][column_index],
                             matrix[row_index][column_index]);
        }
    }
}

TEST(Matrix3x3, ColumnExtraction) {
    Matrix3x3 matrix(Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0), Vector3D(7.0, 8.0, 9.0));
    Vector3D column_0 = matrix.column(0);
    EXPECT_DOUBLE_EQ(column_0.x, 1.0);
    EXPECT_DOUBLE_EQ(column_0.y, 4.0);
    EXPECT_DOUBLE_EQ(column_0.z, 7.0);

    Vector3D column_2 = matrix.column(2);
    EXPECT_DOUBLE_EQ(column_2.x, 3.0);
    EXPECT_DOUBLE_EQ(column_2.y, 6.0);
    EXPECT_DOUBLE_EQ(column_2.z, 9.0);
}

TEST(Matrix3x3, ScalarMultiplication) {
    Matrix3x3 matrix(Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0), Vector3D(7.0, 8.0, 9.0));
    Matrix3x3 scaled = matrix * 2.0;
    EXPECT_DOUBLE_EQ(scaled[0].x, 2.0);
    EXPECT_DOUBLE_EQ(scaled[1].y, 10.0);
    EXPECT_DOUBLE_EQ(scaled[2].z, 18.0);
}

TEST(Matrix3x3, DeterminantOfIdentity) {
    Matrix3x3 identity(Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 1.0, 0.0), Vector3D(0.0, 0.0, 1.0));
    EXPECT_DOUBLE_EQ(identity.determinant(), 1.0);
}

TEST(Matrix3x3, DeterminantOfKnownMatrix) {
    // det([[2,0,0],[0,3,0],[0,0,4]]) = 24
    Matrix3x3 diagonal(Vector3D(2.0, 0.0, 0.0), Vector3D(0.0, 3.0, 0.0), Vector3D(0.0, 0.0, 4.0));
    EXPECT_DOUBLE_EQ(diagonal.determinant(), 24.0);
}

TEST(Matrix3x3, DeterminantOfGeneralMatrix) {
    // det([[1,2,3],[0,1,4],[5,6,0]]) = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24+40-15 = 1
    Matrix3x3 matrix(Vector3D(1.0, 2.0, 3.0), Vector3D(0.0, 1.0, 4.0), Vector3D(5.0, 6.0, 0.0));
    EXPECT_DOUBLE_EQ(matrix.determinant(), 1.0);
}

TEST(Matrix3x3, InverseOfIdentity) {
    Matrix3x3 identity(Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 1.0, 0.0), Vector3D(0.0, 0.0, 1.0));
    Matrix3x3 inverse = identity.inverse();
    for (int row_index = 0; row_index < 3; row_index++) {
        for (int column_index = 0; column_index < 3; column_index++) {
            Real expected = (row_index == column_index) ? 1.0 : 0.0;
            EXPECT_NEAR(inverse[row_index][column_index], expected, 1e-14);
        }
    }
}

TEST(Matrix3x3, InverseTimesOriginalIsIdentity) {
    Matrix3x3 matrix(Vector3D(1.0, 2.0, 3.0), Vector3D(0.0, 1.0, 4.0), Vector3D(5.0, 6.0, 0.0));
    Matrix3x3 inverse = matrix.inverse();
    // Verify M * M^-1 = I by checking M * (M^-1 * e_i) = e_i for each basis vector
    for (int column_index = 0; column_index < 3; column_index++) {
        Vector3D basis_vector;
        basis_vector[column_index] = 1.0;
        Vector3D inverse_column = inverse * basis_vector;
        Vector3D product = matrix * inverse_column;
        for (int row_index = 0; row_index < 3; row_index++) {
            Real expected = (row_index == column_index) ? 1.0 : 0.0;
            EXPECT_NEAR(product[row_index], expected, 1e-12);
        }
    }
}

TEST(Matrix3x3, InverseOfDiagonal) {
    Matrix3x3 diagonal(Vector3D(2.0, 0.0, 0.0), Vector3D(0.0, 4.0, 0.0), Vector3D(0.0, 0.0, 5.0));
    Matrix3x3 inverse = diagonal.inverse();
    EXPECT_NEAR(inverse[0].x, 0.5, 1e-14);
    EXPECT_NEAR(inverse[1].y, 0.25, 1e-14);
    EXPECT_NEAR(inverse[2].z, 0.2, 1e-14);
    // Off-diagonal should be zero
    EXPECT_NEAR(inverse[0].y, 0.0, 1e-14);
    EXPECT_NEAR(inverse[0].z, 0.0, 1e-14);
    EXPECT_NEAR(inverse[1].x, 0.0, 1e-14);
}
