#pragma once

/**
 * Magnetization — the cell-integrated moments of a collinear spin density.
 *
 * Pure data container. Integration over the density is done in layer 4, at
 * the point the SCF reports it.
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * ∫(ρ↑−ρ↓) and ∫|ρ↑−ρ↓| over the cell, in Bohr magnetons.
 *
 * Both are kept because the total alone cannot describe an antiferromagnet:
 * it is zero both for a converged AFM state and for one that collapsed to
 * non-magnetic. A run with no spin density has no Magnetization at all
 * (an empty std::optional), which is distinct from moments that came out
 * at zero.
 */
struct Magnetization {
    Real total_in_bohr_magneton = 0.0;
    Real absolute_in_bohr_magneton = 0.0;
};

} // namespace q3
