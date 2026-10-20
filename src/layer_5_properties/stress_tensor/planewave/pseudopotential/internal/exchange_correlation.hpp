#pragma once

/**
 * Pseudopotential exchange-correlation stress tensor contribution.
 *
 * Extends the base ExchangeCorrelationStressTensorCalculator to include
 * the nonlinear core correction (NLCC) core charge density in the total
 * charge density used for GGA gradient corrections.
 *
 * For pseudopotentials without NLCC, the core charge density is empty
 * and this class behaves identically to the base class.
 */

#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation.hpp"

namespace q3 {

class PseudopotentialExchangeCorrelationStressTensorCalculator
    : public ExchangeCorrelationStressTensorCalculator {
  public:
    PseudopotentialExchangeCorrelationStressTensorCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveBasis& pseudopotential_basis,
        const NormConservingPlanewaveKohnShamSCFState& scf_state,
        StressTensorContributions& result)
        : ExchangeCorrelationStressTensorCalculator(
              crystal_structure, planewave_basis, scf_state, result)
        , pseudopotential_basis_(pseudopotential_basis) {}

  protected:
    /**
     * Build the total charge density including NLCC core charge.
     * Adds the pseudopotential core charge density to the valence charge density.
     */
    [[nodiscard]] std::vector<Complex> build_total_charge_density_g() const override {
        auto total_charge_density_g =
            ExchangeCorrelationStressTensorCalculator::build_total_charge_density_g();
        const auto& core_charge_density_g = pseudopotential_basis_.core_charge_density_reciprocal();
        for (int g_vector_index = 0;
             g_vector_index < static_cast<int>(core_charge_density_g.size());
             g_vector_index++) {
            total_charge_density_g[g_vector_index] += core_charge_density_g[g_vector_index];
        }
        return total_charge_density_g;
    }

    /**
     * Build the per-spin charge density including NLCC core charge.
     * The core charge carries no net moment, so it is split evenly across
     * the two spin channels (preserving core_up + core_down = core_total).
     */
    [[nodiscard]] std::vector<Complex> build_spin_charge_density_g(int spin_index) const override {
        auto spin_charge_density_g =
            ExchangeCorrelationStressTensorCalculator::build_spin_charge_density_g(spin_index);
        const auto& core_charge_density_g = pseudopotential_basis_.core_charge_density_reciprocal();
        for (int g_vector_index = 0;
             g_vector_index < static_cast<int>(core_charge_density_g.size());
             g_vector_index++) {
            spin_charge_density_g[g_vector_index] += 0.5 * core_charge_density_g[g_vector_index];
        }
        return spin_charge_density_g;
    }

  private:
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
};

} // namespace q3
