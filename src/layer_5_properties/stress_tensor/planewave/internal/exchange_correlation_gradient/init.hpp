#pragma once

/**
 * The gradient (GGA) contribution to the exchange-correlation stress, as a
 * strategy selected by functional family.
 *
 * ExchangeCorrelationStressTensorCalculator's diagonal term is exact and
 * complete for a local functional; only gradient-corrected functionals add
 * the Nielsen-Martin ∇ρ⊗∇ρ term. That difference used to live behind a
 * `requires_density_gradient()` branch inside compute(); it now lives in
 * create_exchange_correlation_gradient_stress_calculator(), which resolves a
 * local functional to the Null implementation (AGENTS.md § Calculator Design,
 * same shape as the dispersion and isolated-system-correction factories).
 *
 * ExchangeCorrelationStressDensitySource is the seam that keeps the NLCC axis
 * where it already was: the outer calculator implements it, and its
 * pseudopotential subclass keeps overriding the same two builders to add core
 * charge. The gradient strategy consumes densities only through this
 * interface, so the functional axis and the pseudopotential axis stay
 * independent.
 */

#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Provider of the (possibly NLCC-augmented) densities the gradient stress needs.
class ExchangeCorrelationStressDensitySource {
  public:
    virtual ~ExchangeCorrelationStressDensitySource() = default;

    /// Total charge density in G-space (valence, plus core charge in subclasses).
    [[nodiscard]] virtual std::vector<Complex> build_total_charge_density_g() const = 0;

    /// Per-spin charge density in G-space for the LSDA gradient stress.
    [[nodiscard]] virtual std::vector<Complex>
    build_spin_charge_density_g(int spin_index) const = 0;
};

/// Adds a functional family's gradient stress (nothing, for a local functional).
class ExchangeCorrelationGradientStressCalculator {
  public:
    virtual ~ExchangeCorrelationGradientStressCalculator() = default;

    virtual void add_gradient_stress() = 0;
};

} // namespace q3
