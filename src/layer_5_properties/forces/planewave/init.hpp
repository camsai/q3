#pragma once

/**
 * Force calculator — orchestrator and public API.
 *
 * Coordinates all Hellmann-Feynman force contributions and provides:
 * - The ForceCalculator::compute_forces() static method (top-level public API)
 * - Total assembly, symmetrization, and logging
 *
 * Returns std::pair<AtomicForcesResult, AtomicForcesContributions>:
 *   - AtomicForcesResult: ESSE-schema total forces (values field)
 *   - AtomicForcesContributions: diagnostic per-contribution breakdown
 *
 * Individual contributions are computed by their respective calculator classes:
 *   EwaldForceCalculator                            — internal/ewald.hpp
 *   NormConservingPlanewaveForceCalculator         — pseudopotential/init.hpp
 *     LocalPseudopotentialForceCalculator           — pseudopotential/internal/local.hpp
 *     NonlocalPseudopotentialForceCalculator        — pseudopotential/internal/nonlocal.hpp
 *     CoreCorrectionForceCalculator                 — pseudopotential/internal/core_correction.hpp
 *     UltrasoftForceCalculator                      —
 * pseudopotential/ultrasoft/internal/augmentation.hpp
 */

#include <cmath>
#include <utility>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/factory.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/forces/planewave/contributions.hpp"
#include "layer_5_properties/forces/planewave/internal/ewald.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/init.hpp"

namespace q3 {

/// Compute Hellmann-Feynman forces on all atoms.
struct ForceCalculator {
    /**
     * Compute all Hellmann-Feynman force contributions from the converged SCF state.
     *
     * Extracts crystal, planewave_basis, kpoint_grid, and projector_overlap_coefficients
     * from the state. Returns (AtomicForcesResult, AtomicForcesContributions).
     *
     * The summed forces are averaged over the crystal's space group unless
     * disable_symmetry is set, in which case they are returned as computed -- the
     * run used no symmetry anywhere else either (InputData::disable_symmetry).
     */
    static std::pair<std::vector<Vector3D>, AtomicForcesContributions>
    compute_forces(const NormConservingPlanewaveKohnShamSCFState& scf_state,
                   const std::string& dispersion_correction = "none",
                   bool disable_symmetry = false) {
        const auto& crystal_structure = *scf_state.crystal_structure;
        const auto& planewave_basis = *scf_state.planewave_basis;
        const auto& kpoint_grid = *scf_state.kpoint_grid;
        const auto& projector_overlap_coefficients = scf_state.projector_overlap_coefficients;
        int number_of_atoms = crystal_structure.number_of_atoms();

        // 1. Ewald forces
        EwaldForceCalculator ewald_calculator(crystal_structure, planewave_basis);
        auto ewald_forces = ewald_calculator.compute();

        /**
         * 1b. Dispersion forces -- geometry-only, independent of the
         * electronic structure, so no pool reduction is needed below. The
         * factory resolves the scheme ("none" contributes exact zeros), so
         * no per-scheme branching happens here.
         */
        auto dispersion_calculator =
            create_dispersion_calculator(dispersion_correction, crystal_structure);
        std::vector<Vector3D> dispersion_forces = dispersion_calculator->compute_gradients().forces;
        bool has_dispersion = dispersion_calculator->name() != "none";

        // 2. Pseudopotential forces (local + nonlocal + core-correction + augmentation)
        NormConservingPlanewaveForceCalculator pseudopotential_calculator(
            crystal_structure,
            planewave_basis,
            kpoint_grid,
            scf_state,
            projector_overlap_coefficients);
        auto pseudopotential_contributions = pseudopotential_calculator.compute();

        /**
         * 3. Pool reduction for k-point-dependent force contributions.
         *
         * Nonlocal KB projector and DFT+U (Hubbard) forces depend on wavefunctions: each
         * pool accumulates only its subset of k-points, so a global sum is required.
         * Within a pool whose ranks split the plane waves, both calculators sum their
         * overlaps across the pool before multiplying them, so every rank of a pool
         * holds the same complete value here and this sum counts each pool once.
         *
         * Ewald, local, core-correction, and augmentation forces are computed from the
         * complete crystal structure or density (available on every rank), so they give
         * the same value on all ranks and must NOT be reduced.
         */
        if (scf_state.execution_policy != nullptr) {
            auto reduce_kpoint_dependent_force = [&](std::vector<Vector3D>& force) {
                std::vector<Real> flat;
                flat.reserve(static_cast<std::size_t>(number_of_atoms * 3));
                for (const auto& value : force) {
                    flat.push_back(value.x);
                    flat.push_back(value.y);
                    flat.push_back(value.z);
                }
                scf_state.execution_policy->reduce_vector(flat);
                for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
                    force[atom_index].x = flat[static_cast<std::size_t>(atom_index * 3)];
                    force[atom_index].y = flat[static_cast<std::size_t>(atom_index * 3 + 1)];
                    force[atom_index].z = flat[static_cast<std::size_t>(atom_index * 3 + 2)];
                }
            };
            reduce_kpoint_dependent_force(pseudopotential_contributions.nonlocal);
            if (!scf_state.hubbard_manifolds.empty()) {
                reduce_kpoint_dependent_force(pseudopotential_contributions.hubbard);
            }
        }

        return assemble_total(
            crystal_structure,
            number_of_atoms,
            pseudopotential_contributions.local,
            ewald_forces,
            pseudopotential_contributions.nonlocal,
            pseudopotential_contributions.core_correction,
            pseudopotential_contributions.augmentation,
            dispersion_forces,
            pseudopotential_contributions.hubbard,
            scf_state.quantum_system().pseudopotential_basis().has_nonlinear_core_correction(),
            scf_state.quantum_system().has_ultrasoft(),
            has_dispersion,
            !scf_state.hubbard_manifolds.empty(),
            disable_symmetry);
    }

  private:
    static std::pair<std::vector<Vector3D>, AtomicForcesContributions>
    assemble_total(const Crystal& crystal_structure,
                   int number_of_atoms,
                   const std::vector<Vector3D>& local_forces,
                   const std::vector<Vector3D>& ewald_forces,
                   const std::vector<Vector3D>& nonlocal_forces,
                   const std::vector<Vector3D>& core_correction_forces,
                   const std::vector<Vector3D>& augmentation_forces,
                   const std::vector<Vector3D>& dispersion_forces,
                   const std::vector<Vector3D>& hubbard_forces,
                   bool has_nonlinear_core_correction,
                   bool has_augmentation,
                   bool has_dispersion,
                   bool has_hubbard,
                   bool disable_symmetry) {
        std::vector<Vector3D> total_forces(number_of_atoms, {0.0, 0.0, 0.0});
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            for (int direction = 0; direction < 3; direction++) {
                total_forces[atom_index][direction] =
                    local_forces[atom_index][direction] + ewald_forces[atom_index][direction] +
                    nonlocal_forces[atom_index][direction] +
                    core_correction_forces[atom_index][direction] +
                    augmentation_forces[atom_index][direction] +
                    dispersion_forces[atom_index][direction] +
                    hubbard_forces[atom_index][direction];
            }
        }

        if (!disable_symmetry) {
            auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
            if (symmetry_operations.size() > 1) {
                SymmetryAnalyzer::symmetrize_vector(
                    crystal_structure, symmetry_operations, total_forces);
            }
        }

        log::section("Forces (Ry/bohr):");
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            log::info("     atom {:3d} ({:2s}):  {:12.8f}  {:12.8f}  {:12.8f}",
                      atom_index + 1,
                      crystal_structure.species()[crystal_structure.species_type_index(atom_index)]
                          .symbol,
                      total_forces[atom_index].x,
                      total_forces[atom_index].y,
                      total_forces[atom_index].z);
        }

        log::info("");
        log::info("     Force contributions per atom (Ry/bohr):");
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            log::info("       atom {:2d} local:    {:12.8f}  {:12.8f}  {:12.8f}",
                      atom_index + 1,
                      local_forces[atom_index].x,
                      local_forces[atom_index].y,
                      local_forces[atom_index].z);
            log::info("       atom {:2d} Ewald:    {:12.8f}  {:12.8f}  {:12.8f}",
                      atom_index + 1,
                      ewald_forces[atom_index].x,
                      ewald_forces[atom_index].y,
                      ewald_forces[atom_index].z);
            log::info("       atom {:2d} nonlocal: {:12.8f}  {:12.8f}  {:12.8f}",
                      atom_index + 1,
                      nonlocal_forces[atom_index].x,
                      nonlocal_forces[atom_index].y,
                      nonlocal_forces[atom_index].z);
            log::info("       atom {:2d} NLCC:     {:12.8f}  {:12.8f}  {:12.8f}",
                      atom_index + 1,
                      core_correction_forces[atom_index].x,
                      core_correction_forces[atom_index].y,
                      core_correction_forces[atom_index].z);
            if (has_dispersion) {
                log::info("       atom {:2d} disp:     {:12.8f}  {:12.8f}  {:12.8f}",
                          atom_index + 1,
                          dispersion_forces[atom_index].x,
                          dispersion_forces[atom_index].y,
                          dispersion_forces[atom_index].z);
            }
            if (has_hubbard) {
                log::info("       atom {:2d} hubbard:  {:12.8f}  {:12.8f}  {:12.8f}",
                          atom_index + 1,
                          hubbard_forces[atom_index].x,
                          hubbard_forces[atom_index].y,
                          hubbard_forces[atom_index].z);
            }
        }
        log::info("");
        log::info("     Force contributions (diagonal RMS, Ry/bohr):");
        auto compute_root_mean_square = [&](const std::vector<Vector3D>& forces_vector) -> Real {
            Real sum_squared = 0.0;
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                sum_squared += forces_vector[atom_index].x * forces_vector[atom_index].x +
                               forces_vector[atom_index].y * forces_vector[atom_index].y +
                               forces_vector[atom_index].z * forces_vector[atom_index].z;
            }
            return std::sqrt(sum_squared / (3.0 * number_of_atoms +
                                            math::tolerances::d_matrix_element_threshold));
        };
        log::info("       local:    {:.8f}", compute_root_mean_square(local_forces));
        log::info("       Ewald:    {:.8f}", compute_root_mean_square(ewald_forces));
        log::info("       nonlocal: {:.8f}", compute_root_mean_square(nonlocal_forces));
        if (has_nonlinear_core_correction) {
            log::info("       NLCC:     {:.8f}", compute_root_mean_square(core_correction_forces));
        }
        if (has_augmentation) {
            log::info("       augment:  {:.8f}", compute_root_mean_square(augmentation_forces));
        }
        if (has_dispersion) {
            log::info("       disp:     {:.8f}", compute_root_mean_square(dispersion_forces));
        }
        if (has_hubbard) {
            log::info("       hubbard:  {:.8f}", compute_root_mean_square(hubbard_forces));
        }
        log::info("       TOTAL:    {:.8f}", compute_root_mean_square(total_forces));

        AtomicForcesContributions contributions;
        contributions.local = local_forces;
        contributions.ewald = ewald_forces;
        contributions.nonlocal = nonlocal_forces;
        contributions.core_correction = core_correction_forces;
        contributions.augmentation = augmentation_forces;
        contributions.dispersion = dispersion_forces;
        contributions.hubbard = hubbard_forces;

        return {total_forces, contributions};
    }
};

} // namespace q3
