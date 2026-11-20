#pragma once

#include <functional>
#include <memory>
#include <string>

#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/basis/planewave/structure_factor_calculator.hpp"
#include "layer_3_representation/basis/planewave/structure_factor_calculator_serial.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator_serial.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set_store.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder_serial.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_strain_overlap_calculator.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_strain_overlap_calculator_serial.hpp"
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

    // ─── Kleinman-Bylander projectors ────────────────────────────────────────

    /**
     * The iterator's projector sets, one per k-point, shared with it. Null for a state that no
     * iterator produced; projector_set() then makes a store of the state's own.
     */
    mutable std::shared_ptr<KleinmanBylanderProjectorSetStore> projector_sets;

    /**
     * The Kleinman-Bylander projector set of one k-point: what forces, stress and exports read
     * instead of building projectors themselves. Built on demand, through the same store the
     * SCF used. Once the SCF has released its workspace the store holds one set at a time, so
     * the reference is good until this is called for another k-point.
     */
    [[nodiscard]] const KleinmanBylanderProjectorSet& projector_set(int kpoint_index) const {
        if (!projector_sets) {
            projector_sets = std::make_shared<KleinmanBylanderProjectorSetStore>(
                *crystal_structure,
                *kpoint_grid,
                planewave_basis->kpoint_g_vector_indices(),
                std::make_unique<SerialProjectorSetBuilder>(
                    *crystal_structure,
                    *planewave_basis,
                    quantum_system().pseudopotential_basis()));
            projector_sets->keep_latest_set_only();
        }
        return projector_sets->at(kpoint_index);
    }

    // ─── Band projections ────────────────────────────────────────────────────

    /**
     * How the run that produced this state contracts projectors with bands: on its device, if
     * it had one. Null for a state that no iterator produced; band_projection_calculator()
     * then makes a host one.
     */
    mutable std::shared_ptr<const BandProjectionCalculator> band_projections;

    /**
     * What forces and stress form their <projector|band> overlaps with, so that those products
     * run where the SCF's did.
     */
    [[nodiscard]] const BandProjectionCalculator& band_projection_calculator() const {
        if (!band_projections) {
            band_projections = std::make_shared<SerialBandProjectionCalculator>();
        }
        return *band_projections;
    }

    // ─── Projector strain overlaps ───────────────────────────────────────────

    /// Makes what forms the overlaps of bands with projectors and their strain derivatives.
    using ProjectorStrainOverlapCalculatorFactory =
        std::function<std::unique_ptr<ProjectorStrainOverlapCalculator>(
            const Crystal&, const PlanewaveBasis&, const NormConservingPlanewaveBasis&)>;

    /**
     * How the run that produced this state forms those overlaps: on its device, if it had
     * one. Empty for a host run and for a state that no iterator produced;
     * create_projector_strain_overlap_calculator() then makes the host one.
     */
    ProjectorStrainOverlapCalculatorFactory projector_strain_overlap_calculator_factory;

    /**
     * What the nonlocal stress takes its projector overlaps from, for this state's crystal,
     * basis and pseudopotentials. The host one multiplies through this state's band
     * projection calculator, so the state must outlive what is returned.
     */
    [[nodiscard]] std::unique_ptr<ProjectorStrainOverlapCalculator>
    create_projector_strain_overlap_calculator() const {
        if (projector_strain_overlap_calculator_factory) {
            return projector_strain_overlap_calculator_factory(
                *crystal_structure, *planewave_basis, quantum_system().pseudopotential_basis());
        }
        return std::make_unique<SerialProjectorStrainOverlapCalculator>(
            *crystal_structure,
            *planewave_basis,
            quantum_system().pseudopotential_basis(),
            band_projection_calculator());
    }

    // ─── Structure factors ───────────────────────────────────────────────────

    /// Makes what forms the phase sums of the local, core-correction and Ewald terms.
    using StructureFactorCalculatorFactory =
        std::function<std::unique_ptr<StructureFactorCalculator>(const Crystal&,
                                                                 const PlanewaveBasis&)>;

    /**
     * How the run that produced this state forms those sums: on its device, if it had one.
     * Empty for a host run and for a state that no iterator produced;
     * create_structure_factor_calculator() then makes the host one.
     */
    StructureFactorCalculatorFactory structure_factor_calculator_factory;

    /**
     * What the local, core-correction and Ewald forces and stress take their sums over
     * atoms and G-vectors from, for this state's crystal and basis. The state must outlive
     * what is returned.
     */
    [[nodiscard]] std::unique_ptr<StructureFactorCalculator>
    create_structure_factor_calculator() const {
        if (structure_factor_calculator_factory) {
            return structure_factor_calculator_factory(*crystal_structure, *planewave_basis);
        }
        return std::make_unique<SerialStructureFactorCalculator>(*crystal_structure,
                                                                 *planewave_basis);
    }
};

} // namespace q3
