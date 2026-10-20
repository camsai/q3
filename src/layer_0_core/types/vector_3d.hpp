#pragma once

/**
 * Three-dimensional real-valued vector type.
 *
 * Provides the `Vector3D` struct used throughout Q3 for positions, momenta,
 * lattice vectors, and reciprocal space vectors. Supports basic linear
 * algebra operations: dot product, cross product, norm, and element-wise
 * arithmetic.
 */

#include <cmath>

#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

// ---------------------------------------------------------------------------

// 3D real-valued vector

// ---------------------------------------------------------------------------
struct [[nodiscard]] Vector3D {
    Real x{}, y{}, z{};

    Vector3D() = default;
    constexpr Vector3D(Real x_component, Real y_component, Real z_component)
        : x(x_component), y(y_component), z(z_component) {}

    [[nodiscard]] Real operator[](int component_index) const {
        return (component_index == 0) ? x : (component_index == 1) ? y : z;
    }
    Real& operator[](int component_index) {
        return (component_index == 0) ? x : (component_index == 1) ? y : z;
    }

    [[nodiscard]] Vector3D operator+(const Vector3D& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    [[nodiscard]] Vector3D operator-(const Vector3D& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    [[nodiscard]] Vector3D operator*(Real scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
    [[nodiscard]] Vector3D operator/(Real scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }
    Vector3D& operator+=(const Vector3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    [[nodiscard]] Vector3D operator-() const {
        return {-x, -y, -z};
    }

    [[nodiscard]] Real dot(const Vector3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    [[nodiscard]] Vector3D cross(const Vector3D& other) const {
        return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }
    [[nodiscard]] Real norm2() const {
        return dot(*this);
    }
    [[nodiscard]] Real norm() const {
        return std::sqrt(norm2());
    }
};

[[nodiscard]] inline Vector3D operator*(Real scalar, const Vector3D& vector) {
    return vector * scalar;
}

} // namespace q3
