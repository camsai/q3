#pragma once

/**
 * G-vector.
 *
 * A reciprocal lattice vector characterized by integer Miller indices.
 * Extends ReciprocalSpaceVector with the integer representation,
 * Cartesian coordinates in 2π/alat units, precomputed |G|², and FFT index.
 */

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/reciprocal_space_vector.hpp"

namespace q3 {

/// A reciprocal lattice vector with Miller indices, used in plane-wave calculations.
struct [[nodiscard]] GVector : public ReciprocalSpaceVector {
    /// Integer Miller indices (h, k, l).
    IntegerVector3D miller;

    /// Cartesian coordinates in units of 2π/alat.
    Vector3D g_cart;

    /**
     * Squared magnitude |G|² in units of (2π/alat)² (so real |G|² = magnitude_squared *
     * (2π/alat)²).
     */
    Real magnitude_squared = 0.0;
};

} // namespace q3
