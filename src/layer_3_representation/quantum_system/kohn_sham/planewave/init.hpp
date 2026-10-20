#pragma once

/**
 * PlanewaveKohnShamQuantumSystem — planewave quantum system.
 *
 * Stores the electronic quantum states (wavefunction + eigenvalue + occupation)
 * for all k-point states and bands in a planewave representation.
 *
 * Inheritance chain:
 *   KohnShamQuantumSystem (L2: Crystal, spin)
 *     └─ PlanewaveKohnShamQuantumSystem (this: kpoints, basis, quantum states)
 *         └─ NormConservingPlanewaveKohnShamQuantumSystem (NC pseudopotentials)
 *             └─ UltrasoftPlanewaveKohnShamQuantumSystem (augmentation)
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_2_quantum_system/core/occupation.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_state/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/init.hpp"

namespace q3 {

/**
 * Planewave representation of the electronic quantum system.
 *
 * Stores an array of PlanewaveKohnShamQuantumState indexed as
 * [kpoint_state_index * number_of_bands + band_index].
 */
class PlanewaveKohnShamQuantumSystem : public KohnShamQuantumSystem<PlanewaveKohnShamQuantumState> {
  public:
    PlanewaveKohnShamQuantumSystem() = default;
    ~PlanewaveKohnShamQuantumSystem() override = default;

    PlanewaveKohnShamQuantumSystem(const KPointsGrid& kpoint_grid,
                                   const PlanewaveBasis& basis,
                                   int number_of_bands,
                                   int number_of_spin_components_value,
                                   Real number_of_electrons)
        : kpoint_grid_(&kpoint_grid)
        , basis_(&basis)
        , number_of_bands_(number_of_bands)
        , number_of_electrons_(number_of_electrons) {
        this->number_of_spin_components = number_of_spin_components_value;
    }

    // ─── System parameters ─────────────────────────────────────
    [[nodiscard]] int number_of_bands() const {
        return number_of_bands_;
    }
    [[nodiscard]] int number_of_kpoint_states() const {
        if (!kpoint_grid_)
            return 0;
        return kpoint_grid_->number_of_kpoints() * number_of_spin_components;
    }
    [[nodiscard]] Real number_of_electrons() const {
        return number_of_electrons_;
    }
    [[nodiscard]] Real spin_degeneracy() const {
        return (number_of_spin_components == 1) ? 2.0 : 1.0;
    }

    /// Return a snapshot of the current occupation configuration.
    [[nodiscard]] Occupation occupation_configuration() const {
        Occupation occupation;
        occupation.number_of_electrons = number_of_electrons_;
        occupation.spin_degeneracy = spin_degeneracy();
        occupation.occupied_bands_per_spin = number_of_occupied_bands_per_spin();
        occupation.is_smeared = has_fractional_occupations_;
        return occupation;
    }

    // ─── K-point grid ──────────────────────────────────────────
    [[nodiscard]] const KPointsGrid& kpoint_grid() const {
        return *kpoint_grid_;
    }

    // ─── Basis ─────────────────────────────────────────────────
    [[nodiscard]] const PlanewaveBasis& basis() const {
        return *basis_;
    }

    // ─── Per-state access ──────────────────────────────────────
    [[nodiscard]] const PlanewaveKohnShamQuantumState& quantum_state(int kpoint_state_index,
                                                                     int band_index) const {
        return this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index];
    }
    PlanewaveKohnShamQuantumState& quantum_state(int kpoint_state_index, int band_index) {
        return this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index];
    }

    // ─── Wavefunction access ───────────────────────────────────
    [[nodiscard]] Complex wavefunction(int kpoint_index, int band_index, int g_vector_index) const {
        return this->quantum_states_[kpoint_index * number_of_bands_ + band_index]
            .wavefunction.values()[g_vector_index];
    }

    /**
     * Get raw coefficient data for a specific (kpoint_state, band).
     */
    Complex* wavefunction_band_data(int kpoint_state_index, int band_index) {
        return this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index]
            .wavefunction.coefficient_data();
    }
    [[nodiscard]] const Complex* wavefunction_band_data(int kpoint_state_index,
                                                        int band_index) const {
        return this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index]
            .wavefunction.coefficient_data();
    }

    // ─── Eigenvalue access ─────────────────────────────────────
    [[nodiscard]] Real eigenvalue(int kpoint_state_index, int band_index) const {
        return this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index].eigenvalue;
    }
    void set_eigenvalue(int kpoint_state_index, int band_index, Real value) {
        this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index].eigenvalue =
            value;
    }

    // ─── Occupation access ─────────────────────────────────────
    [[nodiscard]] Real occupation_by_kpoint_state_index_and_band_index(int kpoint_state_index,
                                                                       int band_index) const {
        if (has_fractional_occupations_) {
            return this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index]
                .occupation;
        }
        // Default: fixed occupations
        int occupied = number_of_occupied_bands_per_spin();
        return (band_index < occupied) ? 1.0 : 0.0;
    }

    void set_occupation(int kpoint_state_index, int band_index, Real value) {
        if (!has_fractional_occupations_) {
            has_fractional_occupations_ = true;
            // Initialize all states with fixed occupations (done in the loop below).
            int occupied = number_of_occupied_bands_per_spin();
            int total_states = number_of_kpoint_states();
            for (int state_index = 0; state_index < total_states; state_index++) {
                for (int band_init_index = 0; band_init_index < number_of_bands_;
                     band_init_index++) {
                    this->quantum_states_[state_index * number_of_bands_ + band_init_index]
                        .occupation = (band_init_index < occupied) ? 1.0 : 0.0;
                }
            }
        }
        this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index].occupation =
            value;
    }

    [[nodiscard]] int number_of_occupied_bands_per_spin() const {
        /**
         * Each spin channel holds N/2 electrons: 2 per band when unpolarized
         * (spin_degeneracy 2), 1 per band in LSDA where the two channels are
         * separate k-point states. Fixed occupations therefore describe a
         * zero-total-moment configuration; magnetized systems need smearing
         * (per-channel electron counts then come from the common Fermi level).
         */
        return static_cast<int>(number_of_electrons_ / 2.0 + 0.5);
    }

    // ─── Bulk eigenvalue/occupation access (for compatibility) ──
    /**
     * Copy all eigenvalues into a Matrix<Real>.
     */
    [[nodiscard]] Matrix<Real> eigenvalues_matrix() const {
        int total_kpoint_states = number_of_kpoint_states();
        Matrix<Real> result(total_kpoint_states, number_of_bands_, 0.0);
        for (int kpoint_state_index = 0; kpoint_state_index < total_kpoint_states;
             kpoint_state_index++) {
            for (int band_index = 0; band_index < number_of_bands_; band_index++) {
                result(kpoint_state_index, band_index) =
                    this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index]
                        .eigenvalue;
            }
        }
        return result;
    }

    /**
     * Set all eigenvalues from a Matrix<Real>.
     */
    void set_eigenvalues_from_matrix(const Matrix<Real>& eigenvalues) {
        int total_kpoint_states = number_of_kpoint_states();
        for (int kpoint_state_index = 0; kpoint_state_index < total_kpoint_states;
             kpoint_state_index++) {
            for (int band_index = 0; band_index < number_of_bands_; band_index++) {
                this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index]
                    .eigenvalue = eigenvalues(kpoint_state_index, band_index);
            }
        }
    }

    // ─── Fermi energy ──────────────────────────────────────────
    [[nodiscard]] Real fermi_energy() const {
        return fermi_energy_;
    }
    void set_fermi_energy(Real value) {
        fermi_energy_ = value;
    }

    // ─── Convenience setters for initialization ────────────────
    void set_number_of_bands(int value) {
        number_of_bands_ = value;
    }
    void set_number_of_electrons(Real value) {
        number_of_electrons_ = value;
    }
    void set_kpoint_grid(const KPointsGrid* kpoint_grid) {
        kpoint_grid_ = kpoint_grid;
    }
    void set_basis(const PlanewaveBasis* basis) {
        basis_ = basis;
    }

    /**
     * Copy all electronic state data from another PlanewaveKohnShamQuantumSystem.
     *
     * Used to populate the owned quantum_system_snapshot_ in
     * NormConservingPlanewaveKohnShamSCFState at convergence. Copies wavefunction coefficients,
     * eigenvalues, occupations, Fermi energy, spin configuration, and the k-point grid /
     * basis pointers needed to interpret them. Does NOT touch PP-specific data — that must
     * already be set up by the factory on the destination.
     */
    void
    copy_quantum_states_from(const QuantumSystem<PlanewaveKohnShamQuantumState>& source) override {
        const KPointsGrid* saved_grid = kpoint_grid_;
        const PlanewaveBasis* saved_basis = basis_;

        QuantumSystem<PlanewaveKohnShamQuantumState>::copy_quantum_states_from(source);

        const auto& planewave_source = static_cast<const PlanewaveKohnShamQuantumSystem&>(source);
        number_of_bands_ = planewave_source.number_of_bands_;
        number_of_electrons_ = planewave_source.number_of_electrons_;
        fermi_energy_ = planewave_source.fermi_energy_;
        has_fractional_occupations_ = planewave_source.has_fractional_occupations_;
        number_of_spin_components = planewave_source.number_of_spin_components;

        if (saved_grid != nullptr) {
            kpoint_grid_ = saved_grid;
        } else {
            kpoint_grid_ = planewave_source.kpoint_grid_;
        }
        if (saved_basis != nullptr) {
            basis_ = saved_basis;
        } else {
            basis_ = planewave_source.basis_;
        }
    }

    /**
     * Allocate quantum states using per-k-point PW counts from the basis.
     * Creates number_of_kpoint_states * number_of_bands states, each with
     * its wavefunction sized to the appropriate number of plane waves.
     */
    void allocate_electronic_states(const std::vector<int>& number_of_plane_waves_per_kpoint) {
        int total_kpoint_states = number_of_kpoint_states();
        int number_of_kpoints = kpoint_grid_->number_of_kpoints();
        this->quantum_states_.resize(total_kpoint_states * number_of_bands_);

        for (int kpoint_state_index = 0; kpoint_state_index < total_kpoint_states;
             kpoint_state_index++) {
            int kpoint_index = kpoint_state_index % number_of_kpoints;
            int number_of_plane_waves = number_of_plane_waves_per_kpoint[kpoint_index];
            for (int band_index = 0; band_index < number_of_bands_; band_index++) {
                auto& state =
                    this->quantum_states_[kpoint_state_index * number_of_bands_ + band_index];
                state.wavefunction.resize(number_of_plane_waves);
                state.eigenvalue = 0.0;
                state.occupation = 0.0;
            }
        }
    }

    /**
     * Number of plane waves for a given k-point state.
     */
    [[nodiscard]] int number_of_plane_waves_for_kpoint_state(int kpoint_state_index) const {
        return this->quantum_states_[kpoint_state_index * number_of_bands_]
            .wavefunction.number_of_plane_waves();
    }

  protected:
    const KPointsGrid* kpoint_grid_ = nullptr;
    const PlanewaveBasis* basis_ = nullptr;
    int number_of_bands_ = 0;
    Real number_of_electrons_ = 0.0;
    Real fermi_energy_ = 0.0;
    bool has_fractional_occupations_ = false;
};

} // namespace q3
