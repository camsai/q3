#pragma once

/**
 * NullExchangeCorrelationGradientStressCalculator — the local-functional case.
 *
 * A local (LDA-family) functional has no ∇ρ dependence, so its entire XC
 * stress is the diagonal term the base calculator already added; there is
 * nothing further to contribute. Resolving LDA to this Null object is what
 * lets compute() run the same two statements for every functional family.
 */

#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/init.hpp"

namespace q3 {

class NullExchangeCorrelationGradientStressCalculator
    : public ExchangeCorrelationGradientStressCalculator {
  public:
    void add_gradient_stress() override {}
};

} // namespace q3
