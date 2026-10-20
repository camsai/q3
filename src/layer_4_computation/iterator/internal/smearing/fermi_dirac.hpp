#pragma once

/**
 * FermiDiracSmearing — physical finite-temperature Fermi-Dirac occupation.
 *
 * f(x) = 1 / (exp(x) + 1), the actual electronic occupation at temperature
 * T = smearing_width (in Ry). Unlike Gaussian/Methfessel-Paxton, the entropy
 * term here is the true electronic entropy -[f ln f + (1-f) ln(1-f)], not a
 * generalized/ad hoc correction.
 */

#include <cmath>

#include "layer_4_computation/iterator/internal/smearing/init.hpp"

namespace q3 {

/// Fermi-Dirac (physical finite-T) smearing.
class FermiDiracSmearing : public SmearingScheme {
  public:
    explicit FermiDiracSmearing(Real smearing_width = 0.0) : SmearingScheme(smearing_width) {}

    [[nodiscard]] Real occupation(Real x) const override {
        /**
         * No saturation branch, and none is needed: IEEE-754 arithmetic
         * already makes this expression total and exact at both limits.
         * Large positive x overflows exp() to +inf (clause 7.4, default
         * non-trapping), and 1/(inf + 1) is exactly 0.0; large negative x
         * underflows exp() to +0 (clause 7.5), and 1/(0 + 1) is exactly
         * 1.0. Both are the values a threshold would have returned, so a
         * threshold would only decide *where* to stop computing an answer
         * the format hands back for free.
         *
         * Do not add one back. Any bound below ~709 makes the function
         * discontinuous at the bound in entropy_weight(), which reaches
         * f through this method -- see EntropyWeightIsContinuousAcrossX36.
         */
        return 1.0 / (std::exp(x) + 1.0);
    }

    /**
     * Returns f ln f + (1-f) ln(1-f), which is minus the electronic
     * entropy S = -[f ln f + (1-f) ln(1-f)] >= 0. The negative sign is
     * deliberate and load-bearing: Σ w_k σ entropy_weight(x) is then
     * directly the -T*S total-energy term, matching the sign convention
     * of the other schemes. Do not "correct" this to return a positive
     * entropy -- the sign is consumed by the caller.
     */
    [[nodiscard]] Real entropy_weight(Real x) const override {
        Real f = occupation(x);
        Real one_minus_f = 1.0 - f;
        /**
         * Load-bearing, not defensive: f reaches exactly 0.0 and exactly
         * 1.0 in the saturated tails, where f*log(f) would be 0*(-inf) =
         * NaN. The floor is what keeps the tails at 0 instead. Removing it
         * makes entropy_weight NaN at every saturated point.
         */
        constexpr Real probability_floor = 1e-16;
        Real term_occupied = (f > probability_floor) ? f * std::log(f) : 0.0;
        Real term_empty =
            (one_minus_f > probability_floor) ? one_minus_f * std::log(one_minus_f) : 0.0;
        return term_occupied + term_empty;
    }

    [[nodiscard]] std::string name() const override {
        return "fermi-dirac";
    }
};

} // namespace q3
