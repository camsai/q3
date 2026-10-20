#pragma once

/**
 * Atom: data structure for a single atom within a crystal unit cell.
 *
 * Stores the species index (into the crystal's species list) and the
 * position in fractional (crystal) coordinates.
 */

#include <array>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Describes one atom in the cell.
 */
struct Atom {
    int species_index; // index into species list
    Vector3D position; // in crystal (fractional) coordinates
    /// Which Cartesian components a relaxation may move (pw.x if_pos; ESSE basis.constraints).
    std::array<bool, 3> movable_components{true, true, true};
};

} // namespace q3
