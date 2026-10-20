#pragma once

/**
 * UltrasoftPlanewaveKohnShamQuantumSystem — ultrasoft pseudopotential
 * extension.
 *
 * Extends NormConservingPlanewaveKohnShamQuantumSystem with:
 * - Ultrasoft augmentation via UltrasoftPlanewaveBasis (static tables)
 * - D-matrix access via UltrasoftPlanewaveEffectivePotential
 *
 * All static augmentation data (Q-tables, spherical harmonics, Gaunt coefficients)
 * lives in UltrasoftPlanewaveBasis. The D-matrix is owned by this
 * class when used as a post-SCF snapshot, or pointed to by a non-owning pointer
 * during the SCF iteration where the iterator holds the authoritative copy.
 */

#include <memory>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/effective_potential/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

/**
 * Ultrasoft pseudopotential quantum system.
 *
 * Owns static augmentation data via UltrasoftPlanewaveBasis.
 * Owns a copy of the effective D-matrix when used as a post-SCF snapshot
 * (populated by set_nonlocal_effective_potential(const&) in result()).
 * During SCF iterations the iterator holds the authoritative D-matrix and
 * wires it in via set_nonlocal_effective_potential(pointer).
 */
class UltrasoftPlanewaveKohnShamQuantumSystem
    : public NormConservingPlanewaveKohnShamQuantumSystem {
  public:
    UltrasoftPlanewaveKohnShamQuantumSystem() = default;
    ~UltrasoftPlanewaveKohnShamQuantumSystem() override = default;

    void init(const Crystal& crystal_structure,
              const PlanewaveBasis& planewave_basis,
              const PseudopotentialSet& pseudopotential_set) override {
        basis_ = &planewave_basis;
        auto ultrasoft_basis = std::make_unique<UltrasoftPlanewaveBasis>();
        ultrasoft_basis->init(crystal_structure, planewave_basis, pseudopotential_set);
        ultrasoft_planewave_basis_ = ultrasoft_basis.get();
        norm_conserving_basis_ = std::move(ultrasoft_basis);
    }

    /**
     * Access the ultrasoft basis for direct augmentation table queries.
     */
    [[nodiscard]] const UltrasoftPlanewaveBasis& ultrasoft_basis() const {
        return *ultrasoft_planewave_basis_;
    }

    /**
     * This quantum system uses ultrasoft pseudopotentials.
     */
    [[nodiscard]] bool has_ultrasoft() const override {
        return true;
    }

    /**
     * Copy the converged non-local effective potential into this object's owned storage.
     * Used by UltrasoftPlanewaveKohnShamSCFIterator::result() to produce
     * a self-contained snapshot that does not rely on the iterator's lifetime.
     */
    void set_nonlocal_effective_potential(
        const UltrasoftPlanewaveEffectivePotential& nonlocal_effective_potential) {
        owned_nonlocal_effective_potential_ = nonlocal_effective_potential;
        active_nonlocal_effective_potential_ = &owned_nonlocal_effective_potential_;
    }

    /**
     * Wire in the non-local effective potential via non-owning pointer.
     * Used during SCF iterations where the iterator owns the authoritative D-matrix.
     */
    void set_nonlocal_effective_potential(
        UltrasoftPlanewaveEffectivePotential* nonlocal_effective_potential) {
        active_nonlocal_effective_potential_ = nonlocal_effective_potential;
    }

    [[nodiscard]] Real d_matrix_element(int atom_index,
                                        int species_type,
                                        int projector_index_i,
                                        int projector_index_j,
                                        int spin = 0) const override {
        if (active_nonlocal_effective_potential_ &&
            active_nonlocal_effective_potential_->is_initialized()) {
            return active_nonlocal_effective_potential_->element(
                atom_index, projector_index_i, projector_index_j, spin);
        }
        // Fallback to bare D-matrix (NC pseudopotentials or uninitialised state).
        const auto& basis = pseudopotential_basis();
        int number_of_channels = basis.number_of_projector_channels(species_type);
        if (projector_index_i < number_of_channels && projector_index_j < number_of_channels) {
            int beta_projector_a = basis.projector_to_beta_index(species_type, projector_index_i);
            int beta_projector_b = basis.projector_to_beta_index(species_type, projector_index_j);
            if (basis.projector_combined_lm_index(species_type, projector_index_i) ==
                basis.projector_combined_lm_index(species_type, projector_index_j)) {
                return basis.bare_d_matrix_element(
                    species_type, beta_projector_a, beta_projector_b);
            }
        }
        return 0.0;
    }

  protected:
    /// Non-owning pointer to the ultrasoft-typed basis (owned by parent's unique_ptr).
    UltrasoftPlanewaveBasis* ultrasoft_planewave_basis_ = nullptr;

    /**
     * Owned copy of the converged effective D-matrix.
     * Populated only for post-SCF snapshots (via the const-ref overload).
     */
    UltrasoftPlanewaveEffectivePotential owned_nonlocal_effective_potential_;

    /**
     * Active effective potential pointer. Points to owned_nonlocal_effective_potential_
     * for snapshots, or to the iterator's member during SCF iterations.
     */
    UltrasoftPlanewaveEffectivePotential* active_nonlocal_effective_potential_ = nullptr;
};

} // namespace q3
