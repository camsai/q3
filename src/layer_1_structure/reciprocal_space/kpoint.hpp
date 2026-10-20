#pragma once

#include "layer_0_core/points_grid.hpp"

namespace q3 {

/**
 * A single k-point with coordinates and integration weight.
 *
 * Inherits Vector3D coordinates and Real weight from WeightedPoint3D.
 * Coordinates are in crystal (fractional) units.
 * Weight sums to 2 for number_of_spin_components=1, accounting for spin degeneracy.
 */
struct [[nodiscard]] KPoint : WeightedPoint3D {};

} // namespace q3
