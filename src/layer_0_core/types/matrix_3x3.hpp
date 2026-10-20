#pragma once

/**
 * Matrix3x3: a 3x3 real matrix for lattice, rotation, and stress tensor operations.
 *
 * Stored row-major with each row as a Vector3D. Provides matrix-vector product,
 * transpose, determinant, inverse, and scalar multiplication.
 */

#include "layer_0_core/types/vector_3d.hpp"

namespace q3 {

/// 3×3 matrix stored row-major: row i = mat[i]
struct [[nodiscard]] Matrix3x3 {
    Vector3D rows[3];

    Matrix3x3() = default;
    Matrix3x3(Vector3D row_0, Vector3D row_1, Vector3D row_2) : rows{row_0, row_1, row_2} {}

    [[nodiscard]] const Vector3D& operator[](int row_index) const {
        return rows[row_index];
    }
    Vector3D& operator[](int row_index) {
        return rows[row_index];
    }

    /// Matrix-vector product  M * vector
    [[nodiscard]] Vector3D operator*(const Vector3D& vector) const {
        return {rows[0].dot(vector), rows[1].dot(vector), rows[2].dot(vector)};
    }

    /// Transpose
    [[nodiscard]] Matrix3x3 transpose() const {
        return {{rows[0].x, rows[1].x, rows[2].x},
                {rows[0].y, rows[1].y, rows[2].y},
                {rows[0].z, rows[1].z, rows[2].z}};
    }

    /// Column column_index as a Vector3D
    [[nodiscard]] Vector3D column(int column_index) const {
        return {rows[0][column_index], rows[1][column_index], rows[2][column_index]};
    }

    /// Scalar multiplication
    [[nodiscard]] Matrix3x3 operator*(Real scalar) const {
        return {rows[0] * scalar, rows[1] * scalar, rows[2] * scalar};
    }

    /// Determinant
    [[nodiscard]] Real determinant() const {
        return rows[0].x * (rows[1].y * rows[2].z - rows[1].z * rows[2].y) -
               rows[0].y * (rows[1].x * rows[2].z - rows[1].z * rows[2].x) +
               rows[0].z * (rows[1].x * rows[2].y - rows[1].y * rows[2].x);
    }

    /// Inverse (assumes non-singular)
    [[nodiscard]] Matrix3x3 inverse() const {
        Real determinant_value = determinant();
        Matrix3x3 inverse_matrix;
        inverse_matrix[0] = {(rows[1].y * rows[2].z - rows[1].z * rows[2].y) / determinant_value,
                             (rows[0].z * rows[2].y - rows[0].y * rows[2].z) / determinant_value,
                             (rows[0].y * rows[1].z - rows[0].z * rows[1].y) / determinant_value};
        inverse_matrix[1] = {(rows[1].z * rows[2].x - rows[1].x * rows[2].z) / determinant_value,
                             (rows[0].x * rows[2].z - rows[0].z * rows[2].x) / determinant_value,
                             (rows[0].z * rows[1].x - rows[0].x * rows[1].z) / determinant_value};
        inverse_matrix[2] = {(rows[1].x * rows[2].y - rows[1].y * rows[2].x) / determinant_value,
                             (rows[0].y * rows[2].x - rows[0].x * rows[2].y) / determinant_value,
                             (rows[0].x * rows[1].y - rows[0].y * rows[1].x) / determinant_value};
        return inverse_matrix;
    }
};

} // namespace q3
