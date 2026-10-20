#pragma once

/**
 * PlanewaveKohnShamEnergyCalculator — compute the band-structure energy and the smearing
 * entropy contribution (-TS) from the current Kohn-Sham quantum state.
 *
 * Holds const references to the three read-only data sources it needs so that
 * the iterator can simply call `compute_band_energy()` / `compute_smearing_entropy()`
 * with a single `number_of_bands` and `number_of_spin_components` argument each,
 * rather than threading every dependency through every call.
 *
 * Note: the double-counting correction (which depends on the mutable real-space
 * density and potential arrays) remains in the iterator because it operates on
 * state that changes every SCF iteration and is tightly coupled to those fields.
 */

#include <cmath>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/iterator/internal/smearing/init.hpp"

namespace q3 {

/**
 * Computes the band energy and smearing entropy contribution from the occupied
 * Kohn-Sham eigenvalues and occupations.
 *
 * All referenced objects must outlive this calculator (non-owning references).
 * The `smearing_scheme` pointer is nullable; pass `nullptr` for insulators.
 */
class PlanewaveKohnShamEnergyCalculator {
  public:
    /**
     * Construct a calculator bound to the given k-point grid, quantum system,
     * and smearing scheme.
     *
     * @param kpoint_grid     K-point grid providing coordinates and integration weights.
     * @param quantum_system  The Kohn-Sham quantum system (eigenvalues, occupations, Fermi energy).
     * @param smearing_scheme Pointer to the active smearing scheme, or nullptr for no smearing.
     */
    PlanewaveKohnShamEnergyCalculator(const KPointsGrid& kpoint_grid,
                                      const PlanewaveKohnShamQuantumSystem& quantum_system,
                                      const SmearingScheme* smearing_scheme)
        : kpoint_grid_(kpoint_grid)
        , quantum_system_(quantum_system)
        , smearing_scheme_(smearing_scheme) {}

    /**
     * Compute the band energy: Σ_{k,n} w_k * f_{k,n} * ε_{k,n} (Ry).
     *
     * @param number_of_bands           Number of bands per k-point state.
     * @param number_of_spin_components 1 (non-spin-polarized) or 2 (LSDA).
     */
    [[nodiscard]] Real compute_band_energy(int number_of_bands,
                                           int number_of_spin_components) const {
        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        int number_of_kpoint_states = number_of_kpoints * number_of_spin_components;
        Real band_energy = 0.0;

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             ++kpoint_index_state) {
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            for (int band_index = 0; band_index < number_of_bands; ++band_index) {
                Real occupation = quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                    kpoint_index_state, band_index);
                band_energy += kpoint_grid_[kpoint_index].weight * occupation *
                               quantum_system_.eigenvalue(kpoint_index_state, band_index);
            }
        }

        return band_energy;
    }

    /**
     * Compute the smearing entropy contribution -T*S (Ry).
     *
     * Returns 0.0 immediately if no smearing scheme is active.
     *
     * @param number_of_bands           Number of bands per k-point state.
     * @param number_of_spin_components 1 (non-spin-polarized) or 2 (LSDA).
     */
    [[nodiscard]] Real compute_smearing_entropy(int number_of_bands,
                                                int number_of_spin_components) const {
        if (!smearing_scheme_ || !smearing_scheme_->uses_smearing()) {
            return 0.0;
        }

        Real smearing_width = smearing_scheme_->smearing_width();
        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        int number_of_kpoint_states = number_of_kpoints * number_of_spin_components;
        Real entropy_contribution = 0.0;

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             ++kpoint_index_state) {
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            for (int band_index = 0; band_index < number_of_bands; ++band_index) {
                Real scaling_factor = (quantum_system_.eigenvalue(kpoint_index_state, band_index) -
                                       quantum_system_.fermi_energy()) /
                                      smearing_width;
                entropy_contribution += kpoint_grid_[kpoint_index].weight * smearing_width *
                                        smearing_scheme_->entropy_weight(scaling_factor);
            }
        }

        return entropy_contribution;
    }

  private:
    const KPointsGrid& kpoint_grid_;
    const PlanewaveKohnShamQuantumSystem& quantum_system_;
    const SmearingScheme* smearing_scheme_; // non-owning, nullable
};

} // namespace q3
