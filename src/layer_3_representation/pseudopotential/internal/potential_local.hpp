#pragma once

/**
 * Local potential from a pseudopotential.
 *
 * The local (l-independent) part of the pseudopotential V_loc(r),
 * defined on the radial grid.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Local potential V_loc(r) from a pseudopotential (Ry).
struct PotentialLocal {
    std::vector<Real> radial_values; // V_loc(r) on radial grid
};

} // namespace q3
