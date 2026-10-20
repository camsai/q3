#pragma once

/**
 * NormConservingPlanewaveStressTensorCalculator — aggregates all pseudopotential
 * stress tensor contributions.
 *
 * Coordinates:
 *   - Exchange-correlation stress (NLCC / pseudopotential XC)
 *   - Local pseudopotential stress
 *   - Core-correction stress (NLCC)
 *   - Nonlocal (KB projector) stress — norm-conserving and ultrasoft
 *   - Augmentation stress — ultrasoft only
 *
 * Individual contributions are computed by their respective calculator classes
 * in the internal/ and ultrasoft/internal/ subdirectories.
 */

#include <chrono>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/pseudopotential/internal/core_correction.hpp"
#include "layer_5_properties/stress_tensor/planewave/pseudopotential/internal/exchange_correlation.hpp"
#include "layer_5_properties/stress_tensor/planewave/pseudopotential/internal/hubbard.hpp"
#include "layer_5_properties/stress_tensor/planewave/pseudopotential/internal/local.hpp"
#include "layer_5_properties/stress_tensor/planewave/pseudopotential/internal/nonlocal.hpp"
#include "layer_5_properties/stress_tensor/planewave/pseudopotential/ultrasoft/internal/augmentation.hpp"

namespace q3 {

class NormConservingPlanewaveStressTensorCalculator {
  public:
    NormConservingPlanewaveStressTensorCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const KPointsGrid& kpoint_grid,
        const NormConservingPlanewaveKohnShamSCFState& scf_state,
        const Matrix<Real>& projector_overlap_coefficients,
        StressTensorContributions& result,
        const std::string& nonlocal_stress_derivative_method = "analytical")
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , kpoint_grid_(kpoint_grid)
        , scf_state_(scf_state)
        , projector_overlap_coefficients_(projector_overlap_coefficients)
        , result_(result)
        , nonlocal_stress_derivative_method_(nonlocal_stress_derivative_method) {}

    void compute() {
        auto timer_0 = std::chrono::high_resolution_clock::now();

        /**
         * Atomic and ortho-atomic projectors have complete stress derivatives for every
         * dataset family. Keep this refusal for any future projector type until its
         * derivative is implemented and validated.
         */
        if (!HubbardStressTensorCalculator::supports(scf_state_)) {
            log::warn("     Stress omits the DFT+U contribution for unsupported Hubbard "
                      "projector type '{}'. Reported stress and any cell relaxation built on "
                      "it are NOT trustworthy for this system. Energies are unaffected.",
                      scf_state_.hubbard_projector_type);
        }

        // 1. Exchange-correlation stress (NLCC correction)
        PseudopotentialExchangeCorrelationStressTensorCalculator exchange_correlation_calculator(
            crystal_structure_,
            planewave_basis_,
            scf_state_.quantum_system().pseudopotential_basis(),
            scf_state_,
            result_);
        exchange_correlation_calculator.compute();
        auto timer_1 = std::chrono::high_resolution_clock::now();

        // 2. Local pseudopotential stress
        LocalPseudopotentialStressTensorCalculator local_calculator(
            crystal_structure_,
            planewave_basis_,
            scf_state_.quantum_system().pseudopotential_basis(),
            scf_state_,
            result_);
        local_calculator.compute();
        auto timer_2 = std::chrono::high_resolution_clock::now();

        // 3. Core-correction stress (NLCC)
        CoreCorrectionStressTensorCalculator core_correction_calculator(
            crystal_structure_,
            planewave_basis_,
            scf_state_.quantum_system().pseudopotential_basis(),
            scf_state_,
            result_);
        core_correction_calculator.compute();
        auto timer_3 = std::chrono::high_resolution_clock::now();

        // 4. Nonlocal (KB projector) stress — handles both NC (Dion) and US (Deeq)
        NonlocalPseudopotentialStressTensorCalculator nonlocal_calculator(
            crystal_structure_,
            planewave_basis_,
            kpoint_grid_,
            scf_state_,
            result_,
            nonlocal_stress_derivative_method_);
        nonlocal_calculator.compute();
        auto timer_4 = std::chrono::high_resolution_clock::now();

        // 5. Augmentation stress — zero for NC, nonzero for USPP
        auto [potential_by_spin, coefficients_by_spin] = spin_resolved_augmentation_inputs();
        UltrasoftAugmentationStressTensorCalculator augmentation_calculator(
            crystal_structure_,
            planewave_basis_,
            scf_state_.quantum_system(),
            potential_by_spin,
            coefficients_by_spin,
            result_);
        augmentation_calculator.compute();
        auto timer_5 = std::chrono::high_resolution_clock::now();

        /**
         * 6. DFT+U (Hubbard) stress — zero when no U is applied. Kept out of
         * `nonlocal` because it is a different derivative (the Hubbard projector's,
         * not the KB projector's) and reporting them merged would hide a term worth
         * 29% of the total pressure on silicon at U = 3 eV behind a much larger one.
         */
        HubbardStressTensorCalculator hubbard_calculator(
            crystal_structure_, planewave_basis_, kpoint_grid_, scf_state_, result_);
        hubbard_calculator.compute();
        auto timer_6 = std::chrono::high_resolution_clock::now();

        log::debug("     Pseudo stress sub-timing:");
        log::debug("       xc:           {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_1 - timer_0).count());
        log::debug("       local:        {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_2 - timer_1).count());
        log::debug("       core_corr:    {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_3 - timer_2).count());
        log::debug("       nonlocal:     {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_4 - timer_3).count());
        log::debug("       augmentation: {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_5 - timer_4).count());
        log::debug("       hubbard:      {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_6 - timer_5).count());
    }

  private:
    /**
     * The effective potential and projector occupancies split by spin, with an
     * unpolarized run returning its single stored channel unchanged. Mirrors
     * the force calculator's helper of the same name — both terms are the same
     * pseudopotential derivative and must resolve spin the same way.
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
    StressTensorContributions& result_;
    const std::string nonlocal_stress_derivative_method_;
};

} // namespace q3
