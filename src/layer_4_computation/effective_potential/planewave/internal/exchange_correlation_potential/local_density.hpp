#pragma once

/**
 * LocalDensityExchangeCorrelationPotentialCalculator — V_xc for functionals
 * that depend on the density alone (LDA).
 *
 * A direct pass through the functional's pointwise evaluation; no gradients,
 * so the reciprocal-space density arguments are deliberately unused. Holds no
 * basis or crystal reference for the same reason.
 */

#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/init.hpp"

namespace q3 {

class LocalDensityExchangeCorrelationPotentialCalculator
    : public ExchangeCorrelationPotentialCalculator {
  public:
    explicit LocalDensityExchangeCorrelationPotentialCalculator(
        const XCFunctional& exchange_correlation)
        : exchange_correlation_(exchange_correlation) {}

    void
    evaluate_unpolarized(const std::vector<Real>& exchange_correlation_density,
                         const std::vector<Complex>& /*exchange_correlation_density_g*/,
                         std::vector<Real>& exchange_correlation_potential,
                         std::vector<Real>& exchange_correlation_energy_density) const override {
        exchange_correlation_.compute(exchange_correlation_density,
                                      exchange_correlation_potential,
                                      exchange_correlation_energy_density);
    }

    void
    evaluate_spin_polarized(const std::vector<Real>& exchange_correlation_density_up,
                            const std::vector<Real>& exchange_correlation_density_down,
                            const std::vector<Complex>& /*exchange_correlation_density_up_g*/,
                            const std::vector<Complex>& /*exchange_correlation_density_down_g*/,
                            std::vector<Real>& exchange_correlation_potential_up,
                            std::vector<Real>& exchange_correlation_potential_down,
                            std::vector<Real>& exchange_correlation_energy_density) const override {
        exchange_correlation_.compute_spin(exchange_correlation_density_up,
                                           exchange_correlation_density_down,
                                           exchange_correlation_potential_up,
                                           exchange_correlation_potential_down,
                                           exchange_correlation_energy_density);
    }

  private:
    const XCFunctional& exchange_correlation_;
};

} // namespace q3
