#pragma once

/**
 * NormConservingPlanewaveChargeDensity — extends PlanewaveChargeDensity with
 * pseudopotential-aware data.
 *
 * Adds:
 * - Projector overlap coefficients for ultrasoft augmentation
 * - Augmentation charge management via needs_augmentation()
 */

#include <memory>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/charge_density/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

/// Norm-conserving (and ultrasoft) planewave charge density.
class NormConservingPlanewaveChargeDensity : public PlanewaveChargeDensity {
  public:
    NormConservingPlanewaveChargeDensity() = default;
    NormConservingPlanewaveChargeDensity(
        const PlanewaveBasis& basis,
        int number_of_spin_components,
        const NormConservingPlanewaveKohnShamQuantumSystem& norm_conserving_quantum_system)
        : PlanewaveChargeDensity(basis, number_of_spin_components)
        , norm_conserving_quantum_system_(&norm_conserving_quantum_system) {}

    // ─── Projector overlap coefficients (Quantum ESPRESSO's `becsum`) ───
    [[nodiscard]] const Matrix<Real>& projector_overlap_coefficients() const {
        return projector_overlap_coefficients_;
    }
    Matrix<Real>& projector_overlap_coefficients() {
        return projector_overlap_coefficients_;
    }

    /// Initialize the coefficient storage for given number of atoms and max channels.
    void initialize_projector_overlap_coefficients(int number_of_atoms,
                                                   int maximum_projector_channels) {
        projector_overlap_coefficients_.assign(
            number_of_atoms, maximum_projector_channels * maximum_projector_channels, 0.0);
    }

    /// Reset the projector overlap coefficients to zero.
    void reset_projector_overlap_coefficients() {
        for (size_t row_index = 0; row_index < projector_overlap_coefficients_.rows();
             row_index++) {
            for (size_t column_index = 0; column_index < projector_overlap_coefficients_.cols();
                 column_index++) {
                projector_overlap_coefficients_(row_index, column_index) = 0.0;
            }
        }
    }

    // ─── Augmentation charges ──────────────────────────────────
    [[nodiscard]] bool needs_augmentation() const {
        return norm_conserving_quantum_system_ && norm_conserving_quantum_system_->has_ultrasoft();
    }

    // ─── Quantum system access ────────────────────────────────────
    [[nodiscard]] const NormConservingPlanewaveKohnShamQuantumSystem&
    norm_conserving_quantum_system() const {
        return *norm_conserving_quantum_system_;
    }

  private:
    const NormConservingPlanewaveKohnShamQuantumSystem* norm_conserving_quantum_system_ = nullptr;
    Matrix<Real> projector_overlap_coefficients_;
};

} // namespace q3
