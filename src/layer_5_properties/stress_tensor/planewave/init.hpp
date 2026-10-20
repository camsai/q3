#pragma once

/**
 * Stress tensor calculator — orchestrator and public API.
 *
 * Coordinates all stress tensor contributions and provides:
 * - The compute_stress() free function (top-level public API)
 * - Symmetrization of individual contributions
 * - Merging of augmentation into nonlocal
 * - Total assembly, unit conversion, and logging
 *
 * Returns std::pair<StressTensorResult, StressTensorContributions>:
 *   - StressTensorResult: ESSE-schema total stress tensor and pressure (values, pressure fields)
 *   - StressTensorContributions: diagnostic per-contribution breakdown
 *
 * Individual contributions are computed by their respective calculator classes:
 *   KineticStressTensorCalculator                               — internal/kinetic.hpp
 *   HartreeStressTensorCalculator                               — internal/hartree.hpp
 *   EwaldStressTensorCalculator                                 — internal/ewald.hpp
 *   NormConservingPlanewaveStressTensorCalculator              — pseudopotential/init.hpp
 *     PseudopotentialExchangeCorrelationStressTensorCalculator  —
 * pseudopotential/internal/exchange_correlation.hpp LocalPseudopotentialStressTensorCalculator —
 * pseudopotential/internal/local.hpp CoreCorrectionStressTensorCalculator                      —
 * pseudopotential/internal/core_correction.hpp NonlocalPseudopotentialStressTensorCalculator —
 * pseudopotential/internal/nonlocal.hpp UltrasoftAugmentationStressTensorCalculator               —
 * pseudopotential/ultrasoft/internal/augmentation.hpp
 */

#include <chrono>
#include <string>
#include <utility>

#include "io/logger.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/factory.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/ewald.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/hartree.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/kinetic.hpp"
#include "layer_5_properties/stress_tensor/planewave/pseudopotential/init.hpp"

namespace q3 {

/// Apply crystal symmetry operations to all stress tensor contributions.
static inline void symmetrize_stress_contributions(const Crystal& crystal_structure,
                                                   StressTensorContributions& contributions) {
    auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    if (symmetry_operations.size() <= 1)
        return;

    auto symmetrize_contribution = [&](Real stress_tensor[3][3]) {
        Matrix3x3 stress_matrix;
        for (int polarization_index_a = 0; polarization_index_a < 3; polarization_index_a++) {
            for (int polarization_index_b = 0; polarization_index_b < 3; polarization_index_b++) {
                stress_matrix[polarization_index_a][polarization_index_b] =
                    stress_tensor[polarization_index_a][polarization_index_b];
            }
        }
        SymmetryAnalyzer::symmetrize_matrix(crystal_structure, symmetry_operations, stress_matrix);
        for (int polarization_index_a = 0; polarization_index_a < 3; polarization_index_a++) {
            for (int polarization_index_b = 0; polarization_index_b < 3; polarization_index_b++) {
                stress_tensor[polarization_index_a][polarization_index_b] =
                    stress_matrix[polarization_index_a][polarization_index_b];
            }
        }
    };

    symmetrize_contribution(contributions.kinetic);
    symmetrize_contribution(contributions.hartree);
    symmetrize_contribution(contributions.exchange_correlation);
    symmetrize_contribution(contributions.ewald);
    symmetrize_contribution(contributions.local);
    symmetrize_contribution(contributions.nonlocal);
    symmetrize_contribution(contributions.core_correction);
    symmetrize_contribution(contributions.augmentation);
    symmetrize_contribution(contributions.hubbard);
    symmetrize_contribution(contributions.dispersion);
}

/// Merge US-PP augmentation stress into nonlocal contribution (zero for norm-conserving).
static inline void merge_augmentation_into_nonlocal(bool has_ultrasoft,
                                                    StressTensorContributions& contributions) {
    if (!has_ultrasoft)
        return;
    for (int polarization_index_a = 0; polarization_index_a < 3; polarization_index_a++) {
        for (int polarization_index_b = 0; polarization_index_b < 3; polarization_index_b++) {
            contributions.nonlocal[polarization_index_a][polarization_index_b] +=
                contributions.augmentation[polarization_index_a][polarization_index_b];
        }
    }
}

/**
 * Sum all contributions into total, convert to kbar, log results, and return StressTensorResult.
 *
 * Modifies contributions in-place to convert from Ry/bohr³ to kbar.
 * Returns the ESSE-aligned StressTensorResult (values + pressure).
 */
static inline StressTensorTotal
assemble_total_stress_and_log(bool has_ultrasoft,
                              bool has_dispersion,
                              bool has_hubbard,
                              StressTensorContributions& contributions) {
    // Accumulate total from contributions
    StressTensorTotal stress_tensor_result;
    for (int polarization_index_a = 0; polarization_index_a < 3; polarization_index_a++) {
        for (int polarization_index_b = 0; polarization_index_b < 3; polarization_index_b++) {
            stress_tensor_result.values[polarization_index_a][polarization_index_b] =
                contributions.kinetic[polarization_index_a][polarization_index_b] +
                contributions.hartree[polarization_index_a][polarization_index_b] +
                contributions.exchange_correlation[polarization_index_a][polarization_index_b] +
                contributions.ewald[polarization_index_a][polarization_index_b] +
                contributions.local[polarization_index_a][polarization_index_b] +
                contributions.nonlocal[polarization_index_a][polarization_index_b] +
                contributions.core_correction[polarization_index_a][polarization_index_b] +
                contributions.hubbard[polarization_index_a][polarization_index_b] +
                contributions.dispersion[polarization_index_a][polarization_index_b];
        }
    }

    stress_tensor_result.pressure =
        (stress_tensor_result.values[0][0] + stress_tensor_result.values[1][1] +
         stress_tensor_result.values[2][2]) /
        3.0;

    // Convert all fields from Ry/bohr³ to kbar
    for (int direction_a = 0; direction_a < 3; direction_a++) {
        for (int direction_b = 0; direction_b < 3; direction_b++) {
            stress_tensor_result.values[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
            contributions.kinetic[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
            contributions.hartree[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
            contributions.exchange_correlation[direction_a][direction_b] *=
                constants::RYDBERG_TO_KILOBAR;
            contributions.ewald[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
            contributions.local[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
            contributions.nonlocal[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
            contributions.core_correction[direction_a][direction_b] *=
                constants::RYDBERG_TO_KILOBAR;
            contributions.augmentation[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
            contributions.hubbard[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
            contributions.dispersion[direction_a][direction_b] *= constants::RYDBERG_TO_KILOBAR;
        }
    }
    stress_tensor_result.pressure *= constants::RYDBERG_TO_KILOBAR;

    log::section("Stress tensor (kbar):");
    log::info("     ┌──────────────────────────────────────────┐");
    for (int direction_a = 0; direction_a < 3; direction_a++) {
        log::info("     │ {:>12.2f}  {:>12.2f}  {:>12.2f} │",
                  stress_tensor_result.values[direction_a][0],
                  stress_tensor_result.values[direction_a][1],
                  stress_tensor_result.values[direction_a][2]);
    }
    log::info("     └──────────────────────────────────────────┘");
    log::info("     Pressure = {:.2f} kbar", stress_tensor_result.pressure);
    log::info("");

    log::info("     Contributions:");
    auto print_contribution = [](const char* name, const Real stress_tensor[3][3]) {
        Real stress_pressure =
            (stress_tensor[0][0] + stress_tensor[1][1] + stress_tensor[2][2]) / 3.0;
        log::info("       {:<12s}: {:>10.2f} {:>10.2f} {:>10.2f}  (P={:.2f})",
                  name,
                  stress_tensor[0][0],
                  stress_tensor[1][1],
                  stress_tensor[2][2],
                  stress_pressure);
    };
    print_contribution("kinetic", contributions.kinetic);
    print_contribution("Hartree", contributions.hartree);
    print_contribution("XC", contributions.exchange_correlation);
    print_contribution("Ewald", contributions.ewald);
    print_contribution("local PP", contributions.local);
    print_contribution("nonlocal", contributions.nonlocal);
    print_contribution("core corr", contributions.core_correction);
    if (has_ultrasoft)
        print_contribution("augment", contributions.augmentation);
    if (has_hubbard)
        print_contribution("DFT+U", contributions.hubbard);
    if (has_dispersion)
        print_contribution("dispersion", contributions.dispersion);
    log::info("");

    return stress_tensor_result;
}

/**
 * Compute the stress tensor from converged SCF.
 * Uses the Nielsen-Martin formalism.
 *
 * This is the top-level public API that orchestrates all component calculators.
 * Extracts crystal, planewave_basis, kpoint_grid, and projector_overlap_coefficients
 * from the state. Returns (StressTensorResult, StressTensorContributions).
 *
 * Every contribution is averaged over the crystal's space group unless disable_symmetry
 * is set, in which case the tensors are returned as computed -- the run used no symmetry
 * anywhere else either (InputData::disable_symmetry).
 */
inline std::pair<StressTensorTotal, StressTensorContributions>
compute_stress(const NormConservingPlanewaveKohnShamSCFState& scf_state,
               const std::string& nonlocal_stress_derivative_method = "analytical",
               const std::string& dispersion_correction = "none",
               bool disable_symmetry = false) {
    const auto& crystal_structure = *scf_state.crystal_structure;
    const auto& planewave_basis = *scf_state.planewave_basis;
    const auto& kpoint_grid = *scf_state.kpoint_grid;
    const auto& projector_overlap_coefficients = scf_state.projector_overlap_coefficients;

    StressTensorContributions contributions;

    auto timer_start = std::chrono::high_resolution_clock::now();

    // 1. Kinetic stress
    KineticStressTensorCalculator kinetic_calculator(
        crystal_structure, planewave_basis, kpoint_grid, scf_state, contributions);
    kinetic_calculator.compute();
    auto timer_kinetic = std::chrono::high_resolution_clock::now();

    // 2. Hartree stress
    HartreeStressTensorCalculator hartree_calculator(
        crystal_structure, planewave_basis, scf_state, contributions);
    hartree_calculator.compute();
    auto timer_hartree = std::chrono::high_resolution_clock::now();

    // 3. Ewald stress
    EwaldStressTensorCalculator ewald_calculator(crystal_structure, planewave_basis, contributions);
    ewald_calculator.compute();
    auto timer_ewald = std::chrono::high_resolution_clock::now();

    /**
     * 3b. Dispersion stress -- geometry-only, independent of the electronic
     * structure, so no pool reduction is needed below. The factory resolves
     * the scheme ("none" contributes exact zeros), so no per-scheme branching
     * happens here; has_dispersion only gates the reporting rows.
     */
    auto dispersion_calculator =
        create_dispersion_calculator(dispersion_correction, crystal_structure);
    bool has_dispersion = dispersion_calculator->name() != "none";
    Matrix3x3 dispersion_stress = dispersion_calculator->compute_gradients().stress;
    for (int direction_a = 0; direction_a < 3; direction_a++) {
        for (int direction_b = 0; direction_b < 3; direction_b++) {
            contributions.dispersion[direction_a][direction_b] =
                dispersion_stress[direction_a][direction_b];
        }
    }

    // 4. Pseudopotential stress (XC + local + core-correction + nonlocal + augmentation)
    NormConservingPlanewaveStressTensorCalculator pseudopotential_calculator(
        crystal_structure,
        planewave_basis,
        kpoint_grid,
        scf_state,
        projector_overlap_coefficients,
        contributions,
        nonlocal_stress_derivative_method);
    pseudopotential_calculator.compute();
    auto timer_pseudo = std::chrono::high_resolution_clock::now();

    bool has_ultrasoft = scf_state.quantum_system().has_ultrasoft();

    /**
     * Pool reduction for k-point-dependent stress contributions.
     *
     * Kinetic, nonlocal and Hubbard stress are k-point sums: each pool accumulates only
     * its subset of k-points, so a global sum is required before symmetrization.
     *
     * Hartree, Ewald, XC, local, and core-correction stress terms are computed from
     * the complete density or crystal positions (identical on every rank) and must
     * NOT be reduced.
     */
    if (scf_state.execution_policy != nullptr) {
        // Flatten 3x3 C-arrays to vectors, reduce, then write back.
        auto reduce_within_pool = [&](Real contribution[3][3]) {
            std::vector<Real> flat(9);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    flat[static_cast<std::size_t>(row * 3 + column)] = contribution[row][column];
                }
            }
            scf_state.execution_policy->reduce_within_pool_vector(flat);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    contribution[row][column] = flat[static_cast<std::size_t>(row * 3 + column)];
                }
            }
        };
        auto reduce_across_pools = [&](Real contribution[3][3]) {
            std::vector<Real> flat(9);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    flat[static_cast<std::size_t>(row * 3 + column)] = contribution[row][column];
                }
            }
            scf_state.execution_policy->reduce_vector(flat);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    contribution[row][column] = flat[static_cast<std::size_t>(row * 3 + column)];
                }
            }
        };

        /**
         * 1. All contributions are partial sums over G-vectors or grid slabs within a pool.
         *    Reduce all of them within the pool first.
         */
        reduce_within_pool(contributions.kinetic);
        reduce_within_pool(contributions.hartree);
        reduce_within_pool(contributions.exchange_correlation);
        reduce_within_pool(contributions.ewald);
        reduce_within_pool(contributions.local);
        reduce_within_pool(contributions.nonlocal);
        reduce_within_pool(contributions.core_correction);
        reduce_within_pool(contributions.augmentation);
        reduce_within_pool(contributions.hubbard);

        /**
         * 2. Kinetic, nonlocal and Hubbard stress are k-point sums: each pool accumulates
         *    only its subset of k-points, so a global sum across pools is required.
         */
        reduce_across_pools(contributions.kinetic);
        reduce_across_pools(contributions.nonlocal);
        reduce_across_pools(contributions.hubbard);
    }

    // Symmetrize unless the run uses no symmetry; merge augmentation into nonlocal; log
    if (!disable_symmetry) {
        symmetrize_stress_contributions(crystal_structure, contributions);
    }
    merge_augmentation_into_nonlocal(has_ultrasoft, contributions);
    StressTensorTotal stress_tensor_result = assemble_total_stress_and_log(
        has_ultrasoft, has_dispersion, !scf_state.hubbard_manifolds.empty(), contributions);

    log::debug("     Stress timing breakdown:");
    log::debug("       kinetic:       {:.1f} ms",
               std::chrono::duration<double, std::milli>(timer_kinetic - timer_start).count());
    log::debug("       hartree:       {:.1f} ms",
               std::chrono::duration<double, std::milli>(timer_hartree - timer_kinetic).count());
    log::debug("       ewald:         {:.1f} ms",
               std::chrono::duration<double, std::milli>(timer_ewald - timer_hartree).count());
    log::debug("       pseudo(total): {:.1f} ms",
               std::chrono::duration<double, std::milli>(timer_pseudo - timer_ewald).count());

    return {stress_tensor_result, contributions};
}

} // namespace q3
