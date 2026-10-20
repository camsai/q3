#pragma once

/**
 * Projector.
 *
 * A single Kleinman-Bylander projector function (β_i) in a
 * pseudopotential. Defined on a radial grid.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// A single projector β_i(r) from a pseudopotential.
struct Projector {
    int angular_momentum;
    int number_of_radial_cutoff_points;
    std::vector<Real> radial_values; // beta(r) * r
};

} // namespace q3
