#pragma once

/**
 * Core charge density from a pseudopotential.
 *
 * The core charge density ρ_core(r) for the nonlinear core correction
 * (NLCC). Only populated when has_nonlinear_core_correction is true.
 * Defined on the radial grid.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Core charge density ρ_core(r) from a pseudopotential (NLCC).
struct ChargeCore {
    std::vector<Real> radial_values; // ρ_core(r) on radial grid
};

} // namespace q3
