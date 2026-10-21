#pragma once

/**
 * create_exchange_correlation_gradient_stress_calculator() — the one place the
 * stress path asks a functional whether it needs density gradients.
 *
 * A local functional resolves to the Null object (its whole XC stress is the
 * diagonal term the caller already added); a gradient-corrected functional
 * gets the Nielsen-Martin term (O. H. Nielsen & R. M. Martin, Phys. Rev. B 32,
 * 3780 (1985), extended to gradient corrections by A. Dal Corso & R. Resta,
 * Phys. Rev. B 50, 4327 (1994)). Mirrors
 * create_exchange_correlation_potential_calculator() on the effective-potential
 * side, so the two consumers of `requires_density_gradient()` are both
 * factories.
 */

#include <memory>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/gradient_corrected.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/null.hpp"

namespace q3 {

inline std::unique_ptr<ExchangeCorrelationGradientStressCalculator>
create_exchange_correlation_gradient_stress_calculator(
    const XCFunctional& exchange_correlation_functional,
    const Crystal& crystal_structure,
    const PlanewaveBasis& planewave_basis,
    const NormConservingPlanewaveKohnShamSCFState& scf_state,
    const ExchangeCorrelationStressDensitySource& density_source,
    StressTensorContributions& result) {
    if (exchange_correlation_functional.requires_density_gradient()) {
        return std::make_unique<GradientCorrectedExchangeCorrelationGradientStressCalculator>(
            crystal_structure, planewave_basis, scf_state, density_source, result);
    }
    return std::make_unique<NullExchangeCorrelationGradientStressCalculator>();
}

} // namespace q3
