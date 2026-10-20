#pragma once

#include "layer_4_computation/state/internal/energies/kohn_sham/init.hpp"

namespace q3 {

/**
 * Plane-wave Kohn-Sham energy components in Rydberg.
 *
 * Extends KohnShamEnergies with energy terms specific to a plane-wave
 * calculation: the Ewald ion-ion electrostatic energy and the smearing
 * entropy contribution (-TS) that arises when fractional occupations
 * are used (metals).
 */
struct PlanewaveKohnShamEnergies : public KohnShamEnergies {
    Real smearing_entropy_contribution = 0.0; // -TS term
    Real ewald_energy = 0.0;                  // ion-ion electrostatics (Ewald summation)

    Real total() const override {
        return KohnShamEnergies::total() + smearing_entropy_contribution + ewald_energy;
    }

    /**
     * Dispersion (van der Waals) correction, in Ry. 0 unless a subclass
     * (e.g. the pseudopotential energies struct) overrides this to expose
     * its own dispersion-correction field, for reporting purposes.
     */
    [[nodiscard]] virtual Real dispersion_energy() const {
        return 0.0;
    }

    /**
     * DFT+U (Hubbard) correction, in Ry. 0 unless a subclass overrides this
     * to expose its own hubbard-energy field, for reporting purposes.
     */
    [[nodiscard]] virtual Real hubbard_energy_contribution() const {
        return 0.0;
    }

    /**
     * PAW one-center contribution Σ_a (E¹_a − Ẽ¹_a), in Ry. 0 unless a
     * subclass overrides this to expose its own field, for reporting purposes.
     */
    [[nodiscard]] virtual Real one_center_energy_contribution() const {
        return 0.0;
    }
};

} // namespace q3
