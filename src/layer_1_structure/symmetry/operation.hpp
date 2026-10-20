#pragma once

/**
 * SymmetryOperation: a crystallographic point-group symmetry operation.
 *
 * Stores the real-space rotation matrix (integer, in crystal coordinates),
 * the corresponding reciprocal-space rotation for G-vectors, a fractional
 * translation vector, and the atom permutation map under this operation.
 */

#include <array>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * A single symmetry operation: rotation matrix (integer, crystal coords) and
 * fractional translation (crystal coords).
 * Real-space fractional coords transform as: x' = rotation * x + fractional_translation
 * G-vector Miller indices transform as: G' = reciprocal_rotation * G
 * where reciprocal_rotation = (rotation^{-1})^T.
 * For non-symmorphic operations, the phase factor is exp(-i 2π G · τ).
 */
struct [[nodiscard]] SymmetryOperation {
    std::array<std::array<int, 3>, 3> rotation;            // real-space rotation (crystal coords)
    std::array<std::array<int, 3>, 3> reciprocal_rotation; // (rotation^{-1})^T for G-vectors
    Vector3D fractional_translation; // fractional translation (crystal coords)
    std::vector<int> atom_map; // atom_map[na] = nb: atom na maps to atom nb under this operation
};

} // namespace q3
