#pragma once

#include "layer_4_computation/state/internal/energies/kohn_sham/planewave/init.hpp"

namespace q3 {

/**
 * Pseudopotential plane-wave Kohn-Sham energy components in Rydberg.
 *
 * Extends PlanewaveKohnShamEnergies with correction terms that are specific
 * to pseudopotential calculations: DFT+U Hubbard correction and dispersion
 * (van der Waals) correction. Both are zero unless explicitly enabled.
 */
struct NormConservingPlanewaveKohnShamEnergies : public PlanewaveKohnShamEnergies {
    Real hubbard_energy = 0.0;       // DFT+U (Dudarev) correction
    Real van_der_waals_energy = 0.0; // DFT-D3(BJ) dispersion correction

    /**
     * PAW one-center contribution Σ_a (E¹_a − Ẽ¹_a), in Ry. 0 for NC and
     * ultrasoft.
     *
     * Its counterpart, −Σ_a Σ_ij ρ_ij D^oc_ij, goes into
     * `double_counting_correction` rather than here, because that is what it
     * is: the on-site potential entered the band energy through D, and the
     * band energy is what the double-counting term exists to correct. Keeping
     * the two in separate fields is what lets the A2.6 failure ladder compare
     * the one-center total against QE's printed value directly.
     */
    Real one_center_energy = 0.0;

    Real total() const override {
        return PlanewaveKohnShamEnergies::total() + hubbard_energy + van_der_waals_energy +
               one_center_energy;
    }

    [[nodiscard]] Real dispersion_energy() const override {
        return van_der_waals_energy;
    }

    [[nodiscard]] Real hubbard_energy_contribution() const override {
        return hubbard_energy;
    }

    [[nodiscard]] Real one_center_energy_contribution() const override {
        return one_center_energy;
    }
};

} // namespace q3
