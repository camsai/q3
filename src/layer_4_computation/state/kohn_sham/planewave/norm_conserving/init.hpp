#pragma once

#include <memory>
#include <string>

#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/factory.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"
#include "layer_4_computation/state/internal/energies/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/init.hpp"

namespace q3 {

/**
 * Converged SCF state for a pseudopotential plane-wave Kohn-Sham calculation.
 *
 * Concrete leaf of the SCFState hierarchy. Owns all converged data:
 *
 *   - quantum_system_snapshot_: shared_ptr to the NormConservingPlanewaveKohnShamQuantumSystem
 *     at convergence. Carries both the electronic state (wavefunction coefficients, eigenvalues,
 *     occupations) and all static pseudopotential data (projectors, local potential, core charge).
 *     Shared with the iterator to avoid re-initializing the expensive pseudopotential basis.
 *
 *   - energies_data_: the converged energy breakdown.
 *
 * Non-owning (iterator-owned, valid for the iterator's lifetime):
 *   - planewave_basis* (inherited from PlanewaveKohnShamSCFState)
 *   - exchange_correlation* (inherited from PlanewaveKohnShamSCFState)
 */
struct NormConservingPlanewaveKohnShamSCFState : public PlanewaveKohnShamSCFState {
    /**
     * Explicitly default move operations: when a base class has a deleted copy constructor
     * the derived class's implicit move constructor is also suppressed. We must restore it.
     */
    NormConservingPlanewaveKohnShamSCFState() = default;
    NormConservingPlanewaveKohnShamSCFState(NormConservingPlanewaveKohnShamSCFState&&) = default;
    NormConservingPlanewaveKohnShamSCFState&
    operator=(NormConservingPlanewaveKohnShamSCFState&&) = default;
    ~NormConservingPlanewaveKohnShamSCFState() override = default;

    // ─── Owned data ─────────────────────────────────────────────────────────

    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system_snapshot_;
    NormConservingPlanewaveKohnShamEnergies energies_data_;

    /**
     * Projector overlap coefficients ⟨ψ|β_i⟩⟨β_j|ψ⟩ summed over bands.
     * Empty for norm-conserving; populated for ultrasoft by the US iterator.
     */
    Matrix<Real> projector_overlap_coefficients;
    /**
     * The same coefficients per spin channel, un-summed. Empty for an
     * unpolarized run and for norm-conserving.
     *
     * Paired with `total_potential_g_by_spin`: the augmentation force and
     * stress must contract spin channel against matching spin channel.
     */
    std::vector<Matrix<Real>> projector_overlap_coefficients_by_spin;

    /// DFT+U (Hubbard) resolved manifolds at convergence; empty means no correction is active.
    std::vector<HubbardManifold> hubbard_manifolds;
    /**
     * Converged per-spin Hubbard D-matrix, block-diagonal column-major buffers (one per spin),
     * matching accumulate_hubbard_occupation_and_potential()'s own storage convention.
     */
    std::vector<std::vector<Complex>> hubbard_d_matrix_per_spin;
    /**
     * "atomic" or "ortho-atomic" -- see InputData::hubbard_projector_type. Meaningless when
     * hubbard_manifolds is empty.
     */
    std::string hubbard_projector_type = "atomic";

    // ─── Virtual accessors with covariant return types ───────────────────────

    /**
     * Return the converged pseudopotential quantum system (covariant return).
     * Callers with the concrete type get NormConservingPlanewaveKohnShamQuantumSystem&;
     * callers with PlanewaveKohnShamSCFState& get PlanewaveKohnShamQuantumSystem& via base.
     */
    [[nodiscard]] const NormConservingPlanewaveKohnShamQuantumSystem&
    quantum_system() const override {
        return *quantum_system_snapshot_;
    }

    /// Return the converged energy breakdown (covariant return).
    [[nodiscard]] const NormConservingPlanewaveKohnShamEnergies& energies() const override {
        return energies_data_;
    }
};

} // namespace q3
