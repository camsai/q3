#pragma once

/**
 * SpreadMinimizationResult — the gauge SpreadMinimizer stopped at, the spread there,
 * and how it stopped: converged or not, stationary or not, and by which guide.
 */

#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_terms.hpp"

namespace q3 {

struct [[nodiscard]] SpreadMinimizationResult {
    /// One N x N unitary per k-point; post-multiplies the gauge.
    std::vector<DenseMatrix<Complex>> gauges;
    SpreadTerms terms;
    int iterations = 0;
    /// Omega_D + Omega_OD before any rotation, for the log line and the report.
    Real initial_minimizable_bohr_squared = 0.0;
    /**
     * Whether the minimisation ended on its own terms rather than by running
     * out of iterations. A line search that can find no step to take is the
     * ordinary way to arrive at a minimum, not a failure, so it counts here.
     */
    bool converged = false;
    /**
     * Whether the gauge is actually stationary. This is the stricter question
     * and it is a separate one: a step that has collapsed produces a spread
     * that stops changing, which satisfies any progress test ever written while
     * the gradient is still enormous. Copper stops with a gradient a thousandth
     * of where it began, at a real minimum; a rattled four-atom supercell stops
     * having barely moved, with a gradient still at 80% of its initial value.
     * Only the first has earned the phrase "maximally localized".
     */
    bool stationary = false;
    /// Root-mean-square gradient per k-point where the minimisation stopped.
    Real gradient_norm = 0.0;
};

} // namespace q3
