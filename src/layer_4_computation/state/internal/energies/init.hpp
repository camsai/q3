#pragma once

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Base energy components in Rydberg.
 *
 * Abstract base for all SCF energy breakdowns. Concrete subtypes add
 * physics-level energy terms (Hartree, XC, Ewald, …) and override total()
 * to sum them.
 *
 * There is a single canonical energy value: the return value of total().
 * No separately-stored field mirrors it — that would create a silent
 * divergence risk when new terms are added to an override chain.
 */
struct SCFEnergies {
    virtual ~SCFEnergies() = default;

    /// Return the total energy as the sum of all component fields.
    virtual Real total() const = 0;
};

} // namespace q3
