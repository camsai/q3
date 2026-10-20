#pragma once

/**
 * ExchangeCorrelationPotentialCalculator — the V_xc evaluation strategy.
 *
 * One implementation per functional family (AGENTS.md § Calculator Design):
 * LocalDensityExchangeCorrelationPotentialCalculator evaluates a purely local
 * functional, GradientCorrectedExchangeCorrelationPotentialCalculator adds the
 * White-Bird gradient pipeline a GGA needs. Which one a run gets is decided
 * once, in create_exchange_correlation_potential_calculator(), so
 * PlanewaveEffectivePotentialCalculator never branches on the functional's
 * capabilities.
 *
 * Contract shared by both entry points: the potential buffers arrive already
 * assigned to local_grid_points zeros and the energy-density buffer already
 * sized, exactly as the orchestrator prepared them before this seam existed --
 * implementations fill them in place. The reciprocal-space densities are
 * consumed only by the gradient-corrected implementation (a local functional
 * has no use for them); they are part of the interface because the strategy
 * with the largest needs defines it.
 */

#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

class ExchangeCorrelationPotentialCalculator {
  public:
    virtual ~ExchangeCorrelationPotentialCalculator() = default;

    /// Fill V_xc and the energy density for a spin-unpolarized density.
    virtual void
    evaluate_unpolarized(const std::vector<Real>& exchange_correlation_density,
                         const std::vector<Complex>& exchange_correlation_density_g,
                         std::vector<Real>& exchange_correlation_potential,
                         std::vector<Real>& exchange_correlation_energy_density) const = 0;

    /// Fill per-spin V_xc and the energy density for an LSDA density pair.
    virtual void
    evaluate_spin_polarized(const std::vector<Real>& exchange_correlation_density_up,
                            const std::vector<Real>& exchange_correlation_density_down,
                            const std::vector<Complex>& exchange_correlation_density_up_g,
                            const std::vector<Complex>& exchange_correlation_density_down_g,
                            std::vector<Real>& exchange_correlation_potential_up,
                            std::vector<Real>& exchange_correlation_potential_down,
                            std::vector<Real>& exchange_correlation_energy_density) const = 0;
};

} // namespace q3
