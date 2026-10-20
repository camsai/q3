#pragma once

/**
 * create_exchange_correlation_potential_calculator() — the one place that asks
 * a functional whether it needs density gradients.
 *
 * PlanewaveEffectivePotentialCalculator used to branch on
 * `requires_density_gradient()` inside both spin paths of its V_xc assembly;
 * this factory replaces those branches with a single selection at construction
 * time, the same shape as create_dispersion_calculator() and
 * create_isolated_system_correction().
 */

#include <memory>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/gradient_corrected.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/init.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/local_density.hpp"

namespace q3 {

inline std::unique_ptr<ExchangeCorrelationPotentialCalculator>
create_exchange_correlation_potential_calculator(const PlanewaveBasis& planewave_basis,
                                                 const Crystal& crystal_structure,
                                                 const XCFunctional& exchange_correlation) {
    if (exchange_correlation.requires_density_gradient()) {
        return std::make_unique<GradientCorrectedExchangeCorrelationPotentialCalculator>(
            planewave_basis, crystal_structure, exchange_correlation);
    }
    return std::make_unique<LocalDensityExchangeCorrelationPotentialCalculator>(
        exchange_correlation);
}

} // namespace q3
