#pragma once

/**
 * Reciprocal space vector.
 *
 * A vector in reciprocal (Fourier) space, distinct from real-space
 * vectors by type. Stored in Cartesian coordinates in units of
 * 2π/alat (tpiba).
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// A vector in reciprocal space (Cartesian, units of 2π/alat).
struct [[nodiscard]] ReciprocalSpaceVector {
    Vector3D coordinates;
};

} // namespace q3
