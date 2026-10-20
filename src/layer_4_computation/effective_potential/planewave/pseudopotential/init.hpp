#pragma once

/**
 * NormConservingPlanewaveEffectivePotentialCalculator — extends the base
 * effective potential calculator to include the nonlinear core correction (NLCC)
 * core charge density in the exchange-correlation density.
 *
 * For pseudopotentials without NLCC, the core charge density vector is empty
 * and this class behaves identically to the base class.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/effective_potential/planewave/init.hpp"

namespace q3 {

class NormConservingPlanewaveEffectivePotentialCalculator
    : public PlanewaveEffectivePotentialCalculator {
  public:
    NormConservingPlanewaveEffectivePotentialCalculator(
        const PlanewaveBasis& planewave_basis,
        const Crystal& crystal_structure,
        const XCFunctional& exchange_correlation,
        const execution::PlanewaveExecutionPolicy& execution_policy,
        const std::vector<Complex>& core_charge_density_g)
        : PlanewaveEffectivePotentialCalculator(
              planewave_basis, crystal_structure, exchange_correlation, execution_policy)
        , core_charge_density_g_(core_charge_density_g) {}

  protected:
    /**
     * Build the XC density including the NLCC core charge.
     * Adds the core charge density (split equally among spin channels) to the
     * valence charge density, in both real and reciprocal space.
     * Caches the real-space core charge density to avoid redundant FFTs.
     */
    void build_exchange_correlation_density(
        const std::vector<std::vector<Real>>& charge_density_real,
        int number_of_spin_components,
        std::vector<std::vector<Real>>& exchange_correlation_density,
        std::vector<std::vector<Complex>>& exchange_correlation_density_g) const override {
        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();

        exchange_correlation_density.assign(number_of_spin_components,
                                            std::vector<Real>(local_grid_points));
        exchange_correlation_density_g.assign(number_of_spin_components,
                                              std::vector<Complex>(number_of_g_vectors));

        // Cache the real-space core charge density (constant throughout SCF)
        if (!core_charge_density_g_.empty() && cached_core_charge_density_real_.empty()) {
            planewave_basis_.rho_g_to_r(core_charge_density_g_, cached_core_charge_density_real_);
        }

        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                exchange_correlation_density[spin_index][grid_index] =
                    charge_density_real[spin_index][grid_index];
            }
            if (!cached_core_charge_density_real_.empty()) {
                for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                    exchange_correlation_density[spin_index][grid_index] +=
                        cached_core_charge_density_real_[grid_index] / number_of_spin_components;
                }
            }

            planewave_basis_.rho_r_to_g(charge_density_real[spin_index],
                                        exchange_correlation_density_g[spin_index]);
            for (int g_vector_index = 0;
                 g_vector_index < static_cast<int>(core_charge_density_g_.size());
                 ++g_vector_index) {
                exchange_correlation_density_g[spin_index][g_vector_index] +=
                    core_charge_density_g_[g_vector_index] /
                    static_cast<Real>(number_of_spin_components);
            }
        }
    }

  private:
    const std::vector<Complex>& core_charge_density_g_;
    mutable std::vector<Real> cached_core_charge_density_real_;
};

} // namespace q3
