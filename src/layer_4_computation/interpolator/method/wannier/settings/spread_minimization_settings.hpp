#pragma once

/**
 * SpreadMinimizationSettings — the budget and the tolerances of the spread
 * minimisation (SpreadMinimizer), each with the measurement that set it.
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

struct [[nodiscard]] SpreadMinimizationSettings {
    /// Relative change in Omega_D + Omega_OD that counts as converged.
    Real convergence_tolerance = 1.0e-10;
    int maximum_iterations = 3000;
    /// Iterations the relative change must stay below the tolerance for.
    int convergence_window = 3;
    /// Halvings of the trial step before a stationary point is declared.
    int maximum_step_reductions = 30;
    /// What a trial step is multiplied by each time it fails to lower Omega.
    Real step_reduction_factor = 0.5;
    /**
     * What the step is multiplied by after every success, without a ceiling.
     * Without this the first hard k-point permanently halves the step for the
     * whole run, and a minimisation that started well finishes at a crawl.
     * Backtracking already refuses anything that overshoots, so a cap only
     * slows the recovery from one bad direction -- capping this at the natural
     * scale, and separately resetting it there on every conjugate-gradient
     * restart, were both tried and both cost copper its convergence: it
     * reaches the same Omega_OD of 4.772 either way, but spends the entire
     * iteration budget getting there instead of a hundred iterations.
     */
    Real step_growth_factor = 1.2;
    /**
     * A gradient this small a fraction of where it started ends the run early.
     * It is a shortcut, not the convergence test: the gradient carries the
     * b-weights, which scale as 1/b^2 and so with the cell, and its size at a
     * genuine minimum varies by orders of magnitude between systems -- measured
     * here, a toy chain at Omega_OD = 7e-7, its exact minimum, still shows 3e-4,
     * while copper at its own minimum never falls to a thousandth of where it
     * began. What the run is judged by is whether it can still lower Omega; the
     * gradient is reported beside the answer as the diagnostic it is.
     */
    Real relative_gradient_tolerance = 1.0e-3;
    /**
     * Conjugate-gradient restarts. Steepest descent on this functional creeps:
     * a three-function chain on eight k-points takes five hundred iterations to
     * do what conjugation does in a few dozen, and on a real system the crawl
     * is the difference between converging and exhausting the budget. The
     * search direction is conjugated against the previous one and reset every
     * `restart_interval` iterations, because the manifold is curved and a
     * direction carried too far along it stops being conjugate to anything.
     */
    int restart_interval = 20;
};

} // namespace q3
