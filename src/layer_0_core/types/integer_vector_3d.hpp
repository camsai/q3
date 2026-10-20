#pragma once

/**
 * IntegerVector3D: an integer 3D vector for Miller indices and FFT grid indices.
 *
 * Provides element access by index and equality comparison.
 */

#include "layer_0_core/types/vector_3d.hpp"

namespace q3 {

/// Integer 3-vector for Miller indices / FFT grid indices
struct [[nodiscard]] IntegerVector3D {
    int x{}, y{}, z{};
    IntegerVector3D() = default;
    constexpr IntegerVector3D(int x_component, int y_component, int z_component)
        : x(x_component), y(y_component), z(z_component) {}
    [[nodiscard]] int operator[](int component_index) const {
        return (component_index == 0) ? x : (component_index == 1) ? y : z;
    }
    int& operator[](int component_index) {
        return (component_index == 0) ? x : (component_index == 1) ? y : z;
    }
    [[nodiscard]] bool operator==(const IntegerVector3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

} // namespace q3
