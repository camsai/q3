#pragma once

/**
 * NormConservingPlanewaveForceCalculator — aggregates all pseudopotential
 * force contributions.
 *
 * Coordinates:
 *   - Local pseudopotential force
 *   - Nonlocal (KB projector) force — norm-conserving and ultrasoft
 *   - Core-correction force (NLCC)
 *   - Augmentation force — ultrasoft only
 *
 * Individual contributions are computed by their respective calculator classes
 * in the internal/ and ultrasoft/internal/ subdirectories.
 */

#include <vector>

#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/contributions.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/internal/core_correction.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/internal/hubbard.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/internal/local.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/internal/nonlocal.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/ultrasoft/internal/augmentation.hpp"

namespace q3 {

class NormConservingPlanewaveForceCalculator {
  public:
    NormConservingPlanewaveForceCalculator(const Crystal& crystal_structure,
                                           const PlanewaveBasis& planewave_basis,
                                           const KPointsGrid& kpoint_grid,
                                           const NormConservingPlanewaveKohnShamSCFState& scf_state,
                                           const Matrix<Real>& projector_overlap_coefficients)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , kpoint_grid_(kpoint_grid)
        , scf_state_(scf_state)
        , projector_overlap_coefficients_(projector_overlap_coefficients) {}

    PseudopotentialForceContributions compute() {
        int number_of_atoms = crystal_structure_.number_of_atoms();

        // 1. Local pseudopotential force
        LocalPseudopotentialForceCalculator local_calculator(
            crystal_structure_,
            planewave_basis_,
            scf_state_.quantum_system().pseudopotential_basis());
        std::vector<Vector3D> local_forces = local_calculator.compute(scf_state_.charge_density_g);

        // 2. Nonlocal (KB projector) force — handles both NC and US
        NonlocalPseudopotentialForceCalculator nonlocal_calculator(
            crystal_structure_, planewave_basis_, kpoint_grid_, scf_state_);
        std::vector<Vector3D> nonlocal_forces = nonlocal_calculator.compute();

        // 3. Core-correction force (NLCC) — zero if no NLCC
        std::vector<Vector3D> core_correction_forces(number_of_atoms, {0.0, 0.0, 0.0});
        if (scf_state_.quantum_system().pseudopotential_basis().has_nonlinear_core_correction()) {
            CoreCorrectionForceCalculator core_correction_calculator(
                crystal_structure_,
                planewave_basis_,
                scf_state_.quantum_system().pseudopotential_basis());
            core_correction_forces =
                core_correction_calculator.compute(scf_state_.exchange_correlation_potential_g);
        }

        // 4. Augmentation force — zero for NC, nonzero for USPP
        std::vector<Vector3D> augmentation_forces(number_of_atoms, {0.0, 0.0, 0.0});
        if (scf_state_.quantum_system().has_ultrasoft() &&
            !projector_overlap_coefficients_.empty()) {
            UltrasoftForceCalculator ultrasoft_calculator(crystal_structure_, planewave_basis_);
            auto [potential_by_spin, coefficients_by_spin] = spin_resolved_augmentation_inputs();
            augmentation_forces = ultrasoft_calculator.compute(
                scf_state_.quantum_system(), potential_by_spin, coefficients_by_spin);
        }

        // 5. DFT+U (Hubbard) force — zero unless the input activated it
        std::vector<Vector3D> hubbard_forces(number_of_atoms, {0.0, 0.0, 0.0});
        if (!scf_state_.hubbard_manifolds.empty()) {
            HubbardForceCalculator hubbard_calculator(
                crystal_structure_, planewave_basis_, kpoint_grid_, scf_state_);
            hubbard_forces = hubbard_calculator.compute();
        }

        return PseudopotentialForceContributions{local_forces,
                                                 nonlocal_forces,
                                                 core_correction_forces,
                                                 augmentation_forces,
                                                 hubbard_forces};
    }

  private:
    /**
     * The effective potential and the projector occupancies, split by spin.
     *
     * An unpolarized run stores neither by spin, because there is nothing to
     * split: the single stored channel *is* the answer, and it is returned as a
     * one-element pair. A polarized run stores both, and they must be
     * contracted channel against matching channel — spin-averaging one against
     * the spin-sum of the other drops ∫(V↑−V↓)(ρ↑−ρ↓) dQ/dR / 2.
     *
     * Returns copies rather than references because the unpolarized fallback
     * has nothing to reference: the by-spin containers are empty there.
     */
    [[nodiscard]] std::pair<std::vector<std::vector<Complex>>, std::vector<Matrix<Real>>>
    spin_resolved_augmentation_inputs() const {
        const auto& potential_by_spin = scf_state_.total_potential_g_by_spin;
        const auto& coefficients_by_spin = scf_state_.projector_overlap_coefficients_by_spin;
        if (potential_by_spin.size() < 2 || coefficients_by_spin.size() != potential_by_spin.size())
            return {{scf_state_.total_potential_g}, {projector_overlap_coefficients_}};
        return {potential_by_spin, coefficients_by_spin};
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const KPointsGrid& kpoint_grid_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    const Matrix<Real>& projector_overlap_coefficients_;
};

} // namespace q3
