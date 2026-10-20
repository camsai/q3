#pragma once

/**
 * EffectivePotentialResult — result of an effective-potential evaluation.
 *
 * Holds the charge density, effective potential in real and reciprocal space,
 * and the associated energy contributions, all in Rydberg atomic units.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

struct EffectivePotentialResult {
    std::vector<Complex> charge_density_g;               // total charge density in reciprocal space
    std::vector<std::vector<Real>> real_space_potential; // effective potential per spin
    std::vector<Complex> total_potential_g;              // spin-averaged effective potential

    /**
     * The same potential per spin channel, un-averaged.
     *
     * `total_potential_g` above is the average over channels, which is what the
     * ultrasoft D matrix was historically built from. That average is exact for
     * a non-magnetic cell and wrong for a magnetic one: D_ij = D_ion +
     * ∫v_eff·Q_ij inherits v_eff's spin dependence, so a shared D leaves both
     * channels feeling the same nonlocal potential and the moment collapses.
     *
     * One entry per spin component; for an unpolarized run it has a single
     * entry equal to `total_potential_g`.
     */
    std::vector<std::vector<Complex>> total_potential_g_by_spin;
    std::vector<Complex> exchange_correlation_potential_g;
    Real hartree_energy = 0.0;
    Real exchange_correlation_energy = 0.0;
    Real exchange_correlation_potential_energy = 0.0;
};

} // namespace q3
