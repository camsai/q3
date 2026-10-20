#pragma once

/**
 * SpreadMinimizer — rotate the gauge at every k-point until the Wannier
 * functions are as localised as this mesh allows.
 *
 * The subspace is already chosen when this runs; what is left is the
 * orientation inside it, one N x N unitary per k-point. Omega_I cannot move,
 * so what is descended is Omega_D + Omega_OD, and the descent is the
 * Marzari-Vanderbilt steepest descent: rotate by exp(-step G), where G is the
 * analytic gradient and the step is whatever length actually lowers Omega.
 *
 * The step is found by backtracking rather than fixed. A fixed step is what the
 * literature does and it is fine when a person is watching the spread history;
 * for a driver that must not be watched, a step that overshoots looks exactly
 * like a system that will not localise. Backtracking makes "Omega never
 * increases" an invariant of this class rather than a hope, and turns the two
 * failure modes into different, nameable outcomes: a gradient that reaches the
 * tolerance is converged, a step that collapses without lowering Omega is a
 * stationary point, and neither is silently reported as the other.
 *
 * What is descended is the functional the caller hands in, branch and all:
 * which centres guide its phases is the caller's decision, made once, because
 * a functional whose branch choice moved with the gauge would be a different
 * functional at every iteration, and neither a line search nor a conjugate
 * direction means anything against a moving target.
 *
 * The rotation is applied to the ORIGINAL overlaps every iteration, not
 * composed onto the previous iteration's rotated ones. Composing accumulates a
 * unitary drift that shows up first as an Omega_I that moves -- the one
 * quantity guaranteed not to -- which is how the error is caught, and by then
 * the centres are already wrong. Recomputing costs one N x N product per (k,b)
 * per iteration, which is nothing beside the eigenproblems.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/convergence_window.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/math/linear_algebra/unitary_exponential.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/polak_ribiere_direction.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_functional.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_minimization_result.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_terms.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/spread_minimization_settings.hpp"

namespace q3 {

class SpreadMinimizer {
  public:
    /// Settings that describe no minimisation are refused here rather than mid-run.
    SpreadMinimizer(const KPointStencil& stencil, SpreadMinimizationSettings settings)
        : stencil_(stencil), settings_(settings) {
        if (settings.maximum_iterations < 1) {
            throw std::invalid_argument(
                "SpreadMinimizer: maximum_iterations must be at least 1, got " +
                std::to_string(settings.maximum_iterations));
        }
        if (settings.maximum_step_reductions < 0) {
            throw std::invalid_argument(
                "SpreadMinimizer: maximum_step_reductions must not be negative, got " +
                std::to_string(settings.maximum_step_reductions));
        }
        if (settings.restart_interval < 1) {
            throw std::invalid_argument(
                "SpreadMinimizer: restart_interval must be at least 1, got " +
                std::to_string(settings.restart_interval));
        }
        if (settings.relative_gradient_tolerance < 0.0) {
            throw std::invalid_argument(
                "SpreadMinimizer: relative_gradient_tolerance must not be negative, got " +
                std::to_string(settings.relative_gradient_tolerance));
        }
        if (!(settings.step_reduction_factor > 0.0 && settings.step_reduction_factor < 1.0)) {
            throw std::invalid_argument(
                "SpreadMinimizer: step_reduction_factor must lie in (0, 1), got " +
                std::to_string(settings.step_reduction_factor));
        }
        if (settings.step_growth_factor < 1.0) {
            throw std::invalid_argument(
                "SpreadMinimizer: step_growth_factor must be at least 1, got " +
                std::to_string(settings.step_growth_factor));
        }
    }

    /**
     * `overlaps` are N x N at every (k, b), between the states the gauge has
     * already selected -- the composite V(k)U(k) applied on both sides.
     * `functional` is the spread to descend, over the same stencil.
     */
    [[nodiscard]] SpreadMinimizationResult run(const BlochOverlaps& overlaps,
                                               const SpreadFunctional& functional) const {
        GaugeState state = starting_gauge(overlaps, functional);
        const Real initial_minimizable = state.terms.minimizable_bohr_squared();

        /**
         * The natural scale of the problem: the gradient carries a factor of
         * four and a sum over the b-weights, so this makes the first trial
         * rotation an angle of order one radian's worth of the actual descent
         * direction rather than an arbitrary number of them.
         */
        Real step = 1.0 / (4.0 * stencil_.total_weight());

        ConvergenceWindow window(settings_.convergence_tolerance, settings_.convergence_window);
        PolakRibiereDirection conjugate;
        Real gradient_floor = 0.0;
        for (int iteration = 1; iteration <= settings_.maximum_iterations; iteration++) {
            const std::vector<DenseMatrix<Complex>> gradients =
                functional.gradient(state.rotated, state.terms);
            const Real gradient_norm = norm_per_kpoint(gradients);
            if (iteration == 1) {
                gradient_floor = settings_.relative_gradient_tolerance *
                                 std::max(gradient_norm, std::numeric_limits<Real>::min());
            }
            if (gradient_norm < gradient_floor) {
                return concluded(Ending::Stationary,
                                 std::move(state),
                                 iteration,
                                 gradient_norm,
                                 gradient_floor,
                                 initial_minimizable);
            }

            const std::vector<DenseMatrix<Complex>>& direction =
                conjugate.next(gradients, iteration % settings_.restart_interval == 0);
            std::optional<GaugeState> lowered =
                line_search(overlaps, functional, state, direction, step);
            if (!lowered) {
                /**
                 * The gauge is the one the gradient above was taken at, and
                 * that gradient did not pass the floor or the run would have
                 * ended there -- so this ending is never stationary.
                 */
                return concluded(Ending::StepCollapsed,
                                 std::move(state),
                                 iteration,
                                 gradient_norm,
                                 gradient_floor,
                                 initial_minimizable);
            }
            state = std::move(*lowered);
            window.record(state.terms.minimizable_bohr_squared());
            step *= settings_.step_growth_factor;

            if (window.has_converged()) {
                /**
                 * Omega stopped moving. Whether that is a minimum is settled
                 * by the gradient and not by the fact that it stopped: a line
                 * search whose steps have gone tiny produces a spread that
                 * changes by nothing, which satisfies any relative-change test
                 * while the gauge is nowhere near stationary. Copper stops here
                 * at a real minimum; a rattled four-atom supercell stops here
                 * with a gradient four orders of magnitude too large, and the
                 * two must not get the same word.
                 *
                 * The gradient is taken again, at the gauge being returned:
                 * the one above belongs to the gauge before this iteration's
                 * step.
                 */
                const Real settled_gradient_norm =
                    norm_per_kpoint(functional.gradient(state.rotated, state.terms));
                return concluded(Ending::SpreadSettled,
                                 std::move(state),
                                 iteration,
                                 settled_gradient_norm,
                                 gradient_floor,
                                 initial_minimizable);
            }
        }

        const Real gradient_norm = norm_per_kpoint(functional.gradient(state.rotated, state.terms));
        return concluded(Ending::BudgetExhausted,
                         std::move(state),
                         settings_.maximum_iterations,
                         gradient_norm,
                         gradient_floor,
                         initial_minimizable);
    }

  private:
    /// A gauge at every k-point, the overlaps rotated into it, and the spread there.
    struct GaugeState {
        std::vector<DenseMatrix<Complex>> gauges;
        BlochOverlaps rotated;
        SpreadTerms terms;
    };

    /// The four ways a run ends; only the first has found a stationary gauge for certain.
    enum class Ending { Stationary, StepCollapsed, SpreadSettled, BudgetExhausted };

    /// The identity at every k-point: the gauge the subspace selection handed over.
    [[nodiscard]] GaugeState starting_gauge(const BlochOverlaps& overlaps,
                                            const SpreadFunctional& functional) const {
        std::vector<DenseMatrix<Complex>> gauges(
            static_cast<std::size_t>(stencil_.number_of_kpoints()),
            DenseMatrix<Complex>::identity(functional.number_of_wannier_functions()));
        BlochOverlaps rotated = overlaps.transformed_by(gauges, gauges, stencil_);
        SpreadTerms terms = functional.evaluate(rotated);
        return GaugeState{std::move(gauges), std::move(rotated), std::move(terms)};
    }

    /**
     * Backtracking along `direction` from `state`: the first gauge that
     * lowers Omega, or nothing when every trial step down to the smallest
     * raised it. `step` is shortened in place by each failure and stays
     * shortened, which is what makes the next iteration start from a length
     * that last worked. "Omega never increases" is an invariant of this
     * class because nothing else is ever accepted.
     */
    [[nodiscard]] std::optional<GaugeState>
    line_search(const BlochOverlaps& overlaps,
                const SpreadFunctional& functional,
                const GaugeState& state,
                const std::vector<DenseMatrix<Complex>>& direction,
                Real& step) const {
        for (int reduction = 0; reduction <= settings_.maximum_step_reductions; reduction++) {
            std::vector<DenseMatrix<Complex>> gauges = advanced(state.gauges, direction, step);
            BlochOverlaps rotated = overlaps.transformed_by(gauges, gauges, stencil_);
            SpreadTerms terms = functional.evaluate(rotated);
            if (terms.minimizable_bohr_squared() < state.terms.minimizable_bohr_squared()) {
                return GaugeState{std::move(gauges), std::move(rotated), std::move(terms)};
            }
            step *= settings_.step_reduction_factor;
        }
        return std::nullopt;
    }

    /// U(k) -> U(k) exp(step D(k)), one k-point at a time.
    [[nodiscard]] static std::vector<DenseMatrix<Complex>>
    advanced(const std::vector<DenseMatrix<Complex>>& gauges,
             const std::vector<DenseMatrix<Complex>>& directions,
             Real step) {
        std::vector<DenseMatrix<Complex>> result;
        result.reserve(gauges.size());
        for (std::size_t kpoint = 0; kpoint < gauges.size(); kpoint++) {
            result.push_back(
                multiply(MatrixOperation::None,
                         gauges[kpoint],
                         MatrixOperation::None,
                         exponential_of_anti_hermitian(directions[kpoint].scaled(step))));
        }
        return result;
    }

    /// Root-mean-square gradient per k-point: the size every ending is judged and reported by.
    [[nodiscard]] static Real norm_per_kpoint(const std::vector<DenseMatrix<Complex>>& gradients) {
        return std::sqrt(SpreadFunctional::square_norm(gradients) /
                         static_cast<Real>(gradients.size()));
    }

    /**
     * The result of a run that ended `ending`, said in the log in the words
     * that ending has earned. A gauge that reached stationarity is reported
     * quietly; one that stopped short is warned about with the numbers that
     * show it, and one cut off by the budget says so -- it is returned all
     * the same, because every accepted step lowered Omega, so the gauge in
     * hand is never worse than the projection gauge it started from, and
     * throwing it away would leave the caller with no model at all. What
     * matters is that it is not called converged, and it is not.
     */
    [[nodiscard]] SpreadMinimizationResult concluded(Ending ending,
                                                     GaugeState state,
                                                     int iterations,
                                                     Real gradient_norm,
                                                     Real gradient_floor,
                                                     Real initial_minimizable) const {
        SpreadMinimizationResult result;
        result.iterations = iterations;
        result.initial_minimizable_bohr_squared = initial_minimizable;
        result.gradient_norm = gradient_norm;
        result.converged = ending != Ending::BudgetExhausted;
        result.stationary = ending == Ending::Stationary ||
                            (ending == Ending::SpreadSettled && gradient_norm < gradient_floor);
        report(ending, result, state.terms, gradient_floor);
        result.gauges = std::move(state.gauges);
        result.terms = std::move(state.terms);
        return result;
    }

    void report(Ending ending,
                const SpreadMinimizationResult& result,
                const SpreadTerms& terms,
                Real gradient_floor) const {
        if (ending == Ending::Stationary) {
            log::info("localization: stationary after {} iterations, Omega = {:.9f} bohr^2",
                      result.iterations,
                      terms.total_bohr_squared());
            return;
        }
        if (ending == Ending::BudgetExhausted) {
            log::warn(
                "localization ran out of iterations after {}: Omega = {:.9f} bohr^2 (Omega_I "
                "{:.9f}, Omega_D {:.9f}, Omega_OD {:.9f}), gradient {:.3e}. The gauge returned "
                "is the best one found and is never worse than the projection gauge it started "
                "from, but the minimisation was cut off rather than finished.",
                settings_.maximum_iterations,
                terms.total_bohr_squared(),
                terms.gauge_invariant_bohr_squared,
                terms.diagonal_bohr_squared,
                terms.off_diagonal_bohr_squared,
                result.gradient_norm);
            return;
        }
        if (result.stationary) {
            log::info("localization: converged in {} iterations, Omega = {:.9f} bohr^2 (Omega_I "
                      "{:.9f}, Omega_D {:.9f}, Omega_OD {:.9f})",
                      result.iterations,
                      terms.total_bohr_squared(),
                      terms.gauge_invariant_bohr_squared,
                      terms.diagonal_bohr_squared,
                      terms.off_diagonal_bohr_squared);
            return;
        }
        log::warn("localization stopped short after {} iterations: Omega = {:.9f} bohr^2 (Omega_I "
                  "{:.9f}, Omega_D {:.9f}, Omega_OD {:.9f}), down from {:.9f} minimisable at the "
                  "projection gauge, but the gradient is {:.3e} against the {:.3e} that would "
                  "make it stationary. {}, so this is as far as the minimisation goes; the gauge "
                  "is better than the projection gauge it started from and is not maximally "
                  "localised. Treat the model accordingly.",
                  result.iterations,
                  terms.total_bohr_squared(),
                  terms.gauge_invariant_bohr_squared,
                  terms.diagonal_bohr_squared,
                  terms.off_diagonal_bohr_squared,
                  result.initial_minimizable_bohr_squared,
                  result.gradient_norm,
                  gradient_floor,
                  ending == Ending::StepCollapsed
                      ? "No further step lowered Omega"
                      : "Omega stopped moving while steps were still being accepted");
    }

    const KPointStencil& stencil_;
    SpreadMinimizationSettings settings_;
};

} // namespace q3
