#pragma once

/**
 * Atomic charge density from a pseudopotential.
 *
 * The pseudo-atomic charge density ρ_at(r) used for initial
 * charge density construction. Defined on the radial grid.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Atomic charge density ρ_at(r) from a pseudopotential.
struct ChargeAtomic {
    std::vector<Real> radial_values; // ρ_at(r) on radial grid
};

} // namespace q3
