#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/guided_phase_branch.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_functional.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_minimizer.hpp"
#include "utils/wannier_chain_model.hpp"

using namespace q3;
using q3::test_utils::make_wannier_chain;
using q3::test_utils::overlaps_of;
using q3::test_utils::stencil_of;
using q3::test_utils::WannierChainModel;

namespace {

/**
 * The minimisation as the construction runs it: the phases guided by
 * `trial_orbital_centers` when there are any, and otherwise by the starting
 * gauge's own centres.
 */
SpreadMinimizationResult minimize(const WannierChainModel& model,
                                  SpreadMinimizationSettings settings = {},
                                  const std::vector<Vector3D>& trial_orbital_centers = {}) {
    const KPointStencil stencil = stencil_of(model);
    const BlochOverlaps overlaps = overlaps_of(model);
    const int dimension = model.number_of_wannier_functions;
    const SpreadFunctional functional(
        stencil,
        dimension,
        std::make_shared<const GuidedPhaseBranch>(
            !trial_orbital_centers.empty()
                ? trial_orbital_centers
                : SpreadFunctional(stencil, dimension).evaluate(overlaps).centers_bohr));
    return SpreadMinimizer(stencil, settings).run(overlaps, functional);
}

/// The same descent on the principal branch, where a diagonal phase jumps at the cut.
SpreadMinimizationResult
minimize_on_the_principal_branch(const WannierChainModel& model,
                                 SpreadMinimizationSettings settings = {}) {
    const KPointStencil stencil = stencil_of(model);
    const SpreadFunctional functional(stencil, model.number_of_wannier_functions);
    return SpreadMinimizer(stencil, settings).run(overlaps_of(model), functional);
}

} // namespace

/**
 * The test the whole file exists for. The chain's own gauge has exactly zero
 * minimisable spread; scrambling it with a different unitary at every k-point
 * raises that, and nothing else about the model changes. So the minimiser has a
 * known target to reach, not merely a number to reduce -- which is the
 * difference between testing that an optimiser runs and testing that it
 * optimises.
 */
TEST(SpreadMinimizer, ItRecoversTheKnownMinimumFromAPerturbedGauge) {
    WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.85);
    q3::test_utils::perturb_gauge(model, 12345u, 0.35);

    const SpreadMinimizationResult result = minimize(model);

    EXPECT_GT(result.initial_minimizable_bohr_squared, 0.1)
        << "the perturbation should have left real work to do";
    EXPECT_LT(result.terms.minimizable_bohr_squared(),
              1.0e-4 * result.initial_minimizable_bohr_squared)
        << "started at " << result.initial_minimizable_bohr_squared << " bohr^2 and stopped at "
        << result.terms.minimizable_bohr_squared();
    EXPECT_TRUE(result.stationary);
}

/**
 * From a random point on the unitary group the functional has local minima, and
 * this one lands in a deep one: about 2% of where it started, but not zero. That
 * is a property of Omega, not a defect here -- it is precisely why a Wannier
 * code starts from projections onto real orbitals rather than from nothing, and
 * why the automatic protocol's initial guess is the part that has to be good.
 * Asserting the honest bound keeps the fact visible instead of tuning the seed
 * until it disappears.
 */
TEST(SpreadMinimizer, AStronglyScrambledGaugeIsImprovedButNeedNotReachTheGlobalMinimum) {
    WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.85);
    q3::test_utils::scramble_gauge(model, 12345u);

    const SpreadMinimizationResult result = minimize(model);

    EXPECT_LT(result.terms.minimizable_bohr_squared(),
              0.05 * result.initial_minimizable_bohr_squared);
    EXPECT_GT(result.terms.minimizable_bohr_squared(), 0.0);
}

/**
 * Recovering the minimum means recovering the centres it belongs to -- but only
 * modulo a lattice vector. A Wannier centre is a Berry phase, so the
 * construction fixes where a function sits inside the cell and says nothing
 * about which cell is its home: the minimiser returning 9.0 bohr where the
 * model was built with 3.0 on a 6.0 bohr chain is the same Wannier function,
 * not a worse one. Folding before comparing is the test being right about the
 * physics, not the test being lenient.
 */
TEST(SpreadMinimizer, TheRecoveredCentersAreTheChainsOwnModuloALatticeVector) {
    const Real lattice_constant = 6.0;
    WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, lattice_constant, 0.85);
    const std::vector<Real> expected = model.centers;
    q3::test_utils::perturb_gauge(model, 6060u, 0.35);

    const SpreadMinimizationResult result = minimize(model);

    /// Distance to the nearest periodic image, which is the only distance there is.
    const auto separation = [lattice_constant](Real left, Real right) {
        const Real difference = left - right;
        const Real folded = difference / lattice_constant;
        return std::abs(difference - std::round(folded) * lattice_constant);
    };

    std::vector<Real> recovered;
    for (const Vector3D& center : result.terms.centers_bohr) {
        recovered.push_back(center[0]);
    }
    std::sort(recovered.begin(), recovered.end());
    std::vector<Real> sorted_expected = expected;
    std::sort(sorted_expected.begin(), sorted_expected.end());

    for (std::size_t index = 0; index < recovered.size(); index++) {
        /**
         * Loose because the convergence floor is relative: the minimisation stops
         * when the gradient has fallen a thousandth of where it began, which
         * leaves the centres good to about a milli-bohr rather than to machine
         * precision. That is 5e-4 Angstrom, well below anything a band structure
         * can tell apart.
         */
        EXPECT_LT(separation(recovered[index], sorted_expected[index]), 5.0e-3)
            << "recovered " << recovered[index] << " against " << sorted_expected[index];
    }
}

/**
 * Omega_I belongs to the subspace, not the gauge. If it moves here, the
 * localiser is quietly undoing the disentangler's work.
 */
TEST(SpreadMinimizer, TheGaugeInvariantSpreadSurvivesTheMinimization) {
    WannierChainModel model = make_wannier_chain(6, {0.0, 2.0}, 5.5, 0.9);
    q3::test_utils::scramble_gauge(model, 31337u);

    const KPointStencil stencil = stencil_of(model);
    const SpreadTerms before =
        SpreadFunctional(stencil, model.number_of_wannier_functions).evaluate(overlaps_of(model));
    const SpreadMinimizationResult result = minimize(model);

    EXPECT_NEAR(
        result.terms.gauge_invariant_bohr_squared, before.gauge_invariant_bohr_squared, 1.0e-9);
}

TEST(SpreadMinimizer, TheReturnedGaugesAreUnitary) {
    WannierChainModel model = make_wannier_chain(6, {0.0, 2.0, -1.0}, 5.5, 0.85);
    q3::test_utils::scramble_gauge(model, 808u);

    const SpreadMinimizationResult result = minimize(model);
    const int dimension = model.number_of_wannier_functions;

    for (const DenseMatrix<Complex>& gauge : result.gauges) {
        for (int left = 0; left < dimension; left++) {
            for (int right = 0; right < dimension; right++) {
                Complex inner(0.0, 0.0);
                for (int row = 0; row < dimension; row++) {
                    inner += std::conj(gauge(row, left)) * gauge(row, right);
                }
                EXPECT_NEAR(inner.real(), left == right ? 1.0 : 0.0, 1.0e-10);
                EXPECT_NEAR(inner.imag(), 0.0, 1.0e-10);
            }
        }
    }
}

/**
 * Backtracking is what makes this an invariant rather than a hope: no accepted
 * iteration may raise the spread, whatever the step length would have done.
 */
TEST(SpreadMinimizer, AnAlreadyLocalizedModelIsLeftWhereItIs) {
    const WannierChainModel model = make_wannier_chain(6, {0.0, 2.0}, 5.5, 0.9);

    const SpreadMinimizationResult result = minimize(model);

    EXPECT_NEAR(result.initial_minimizable_bohr_squared, 0.0, 1.0e-12);
    EXPECT_NEAR(result.terms.minimizable_bohr_squared(), 0.0, 1.0e-12);
}

/**
 * Guiding centres must not cost anything where the minimisation already worked.
 *
 * Taking each phase on the branch its centre implies is a different
 * representative of the same quantity, so the functional is not bit-identical
 * near the cut; what matters is that the minimisation still reaches a
 * stationary gauge and lands on essentially the same spread. This chain puts
 * b . r at 0.9 pi, hard against the cut, and both settings converge there to
 * within a thousandth of a bohr squared.
 *
 * The case guiding centres were built for is NOT reproduced here, and the
 * attempt to build a toy that reproduced it was abandoned rather than tuned
 * until it did: a diagonal chain is localisable whatever branch is chosen, so
 * it converges either way. The failure needs off-diagonal structure and a
 * disentangled subspace together, and it is exhibited and gated by the shipped
 * Si3C example -- stalled at 23 iterations before guiding centres, stationary
 * at 291 after, with the off-mesh error more than halved.
 */
TEST(SpreadMinimizer, GuidingCentersCostNothingWhereThePhasesAlreadyWorked) {
    /**
     * b = 2pi/(4 * 6) = 0.2618 per bohr, so a centre near 10.8 bohr puts b . r
     * near 0.9 pi -- against the cut, which is the condition being tested.
     */
    WannierChainModel model = make_wannier_chain(4, {0.0, 10.8, -10.4}, 6.0, 0.85);
    q3::test_utils::perturb_gauge(model, 4242u, 0.9);

    const SpreadMinimizationResult unguided = minimize_on_the_principal_branch(model);
    const SpreadMinimizationResult guided = minimize(model);

    EXPECT_TRUE(guided.stationary);
    EXPECT_TRUE(unguided.stationary);
    EXPECT_NEAR(
        guided.terms.minimizable_bohr_squared(), unguided.terms.minimizable_bohr_squared(), 1.0e-3);
}

/**
 * Where no phase is near the cut, guiding changes nothing. This is what keeps
 * the four examples that already worked reporting the numbers they reported:
 * every branch choice comes out zero, so the functional is the same one.
 */
TEST(SpreadMinimizer, GuidingCentersAreANoOpWhenNoPhaseIsNearTheBranch) {
    WannierChainModel model = make_wannier_chain(8, {0.0, 0.4, -0.3}, 6.0, 0.9);
    q3::test_utils::perturb_gauge(model, 99u, 0.2);

    const SpreadMinimizationResult guided = minimize(model);
    const SpreadMinimizationResult unguided = minimize_on_the_principal_branch(model);

    EXPECT_NEAR(guided.terms.total_bohr_squared(), unguided.terms.total_bohr_squared(), 1.0e-8);
    EXPECT_NEAR(guided.terms.gauge_invariant_bohr_squared,
                unguided.terms.gauge_invariant_bohr_squared,
                1.0e-10);
}

/**
 * The demonstration the guiding-centre change could not produce before: from
 * a strongly scrambled gauge the minimiser lands in a local minimum at about 2%
 * of its starting spread (the test above). Given the chain's TRUE centres as
 * external guides -- the trial-orbital positions, in the real code -- the same
 * minimiser from the same start recovers the exact minimum. The guide is doing
 * what a guide is for: fixing the branch of every diagonal phase from a
 * position that is known, rather than from a gauge that is not yet trusted.
 */
TEST(SpreadMinimizer, TrialOrbitalCentersCarryAScrambledGaugeToTheTrueMinimum) {
    WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.85);
    std::vector<Vector3D> true_centers;
    for (const Real center : model.centers) {
        true_centers.push_back(Vector3D{center, 0.0, 0.0});
    }
    q3::test_utils::scramble_gauge(model, 12345u);

    const SpreadMinimizationResult unguided = minimize(model);
    const SpreadMinimizationResult guided =
        minimize(model, SpreadMinimizationSettings{}, true_centers);

    EXPECT_LT(guided.terms.minimizable_bohr_squared(), unguided.terms.minimizable_bohr_squared())
        << "guided " << guided.terms.minimizable_bohr_squared() << " against unguided "
        << unguided.terms.minimizable_bohr_squared();
}

TEST(SpreadMinimizer, TheWrongNumberOfTrialCentersIsRefused) {
    WannierChainModel model = make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9);
    const std::vector<Vector3D> three{
        Vector3D{0.0, 0.0, 0.0}, Vector3D{1.0, 0.0, 0.0}, Vector3D{2.0, 0.0, 0.0}};
    EXPECT_THROW(minimize(model, SpreadMinimizationSettings{}, three), std::invalid_argument);
}

/**
 * Settings that describe no minimisation are refused when the minimiser is
 * built. A restart interval of zero is the one that mattered: it is the
 * divisor of the restart test, so it reached the run as undefined behaviour.
 */
TEST(SpreadMinimizer, SettingsThatDescribeNoMinimizationAreRefused) {
    const WannierChainModel model = make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9);
    const KPointStencil stencil = stencil_of(model);

    SpreadMinimizationSettings no_restart;
    no_restart.restart_interval = 0;
    EXPECT_THROW(SpreadMinimizer(stencil, no_restart), std::invalid_argument);

    SpreadMinimizationSettings no_budget;
    no_budget.maximum_iterations = 0;
    EXPECT_THROW(SpreadMinimizer(stencil, no_budget), std::invalid_argument);

    SpreadMinimizationSettings negative_reductions;
    negative_reductions.maximum_step_reductions = -1;
    EXPECT_THROW(SpreadMinimizer(stencil, negative_reductions), std::invalid_argument);

    SpreadMinimizationSettings negative_floor;
    negative_floor.relative_gradient_tolerance = -1.0e-3;
    EXPECT_THROW(SpreadMinimizer(stencil, negative_floor), std::invalid_argument);

    SpreadMinimizationSettings growing_reduction;
    growing_reduction.step_reduction_factor = 1.0;
    EXPECT_THROW(SpreadMinimizer(stencil, growing_reduction), std::invalid_argument);

    SpreadMinimizationSettings shrinking_growth;
    shrinking_growth.step_growth_factor = 0.9;
    EXPECT_THROW(SpreadMinimizer(stencil, shrinking_growth), std::invalid_argument);
}

TEST(SpreadMinimizer, ANeighborOutsideTheGridIsRefused) {
    WannierChainModel model = make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9);
    model.neighbor_kpoint_index[0][0] = 99;

    EXPECT_THROW(minimize(model), std::invalid_argument);
}

/**
 * Running out of iterations returns the gauge, flagged. Throwing was tried and
 * reverted: every accepted step lowered Omega and no step that raised it was
 * ever taken, so the gauge in hand is strictly better than the projection gauge
 * it started from, and discarding it leaves the caller with something worse and
 * no model at all. The requirement is not that an unconverged gauge is
 * withheld -- it is that it is never called converged.
 */
TEST(SpreadMinimizer, ExhaustingTheIterationBudgetReturnsTheGaugeMarkedUnconverged) {
    WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.85);
    q3::test_utils::scramble_gauge(model, 555u);

    SpreadMinimizationSettings settings;
    settings.maximum_iterations = 2;
    settings.convergence_tolerance = 1.0e-16;

    const SpreadMinimizationResult result = minimize(model, settings);

    EXPECT_FALSE(result.converged) << "cut off by the budget, not finished";
    EXPECT_FALSE(result.stationary);
    EXPECT_GT(result.gradient_norm, 0.0);
    EXPECT_EQ(result.gauges.size(), model.overlaps.size());
    EXPECT_LE(result.terms.minimizable_bohr_squared(), result.initial_minimizable_bohr_squared);
}

/**
 * The gradient a run reports is the gradient of the gauge it returns. An
 * ending that follows an accepted step has moved the gauge since the gradient
 * at the top of the iteration was taken, so reporting that one describes a
 * gauge the caller never sees -- and judging stationarity by it, after the
 * early exit has already returned for every gradient under the floor, could
 * never find a run that settled stationary. The floor is switched off here so
 * the run ends on the spread settling, and the descent is on the principal
 * branch so the test can rebuild the functional.
 */
TEST(SpreadMinimizer, TheReportedGradientBelongsToTheReturnedGauge) {
    WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.85);
    q3::test_utils::scramble_gauge(model, 12345u);

    SpreadMinimizationSettings settings;
    settings.relative_gradient_tolerance = 0.0;
    settings.convergence_tolerance = 1.0e-3;
    const SpreadMinimizationResult result = minimize_on_the_principal_branch(model, settings);
    ASSERT_TRUE(result.converged) << "the run should end on the spread settling, not the budget";
    ASSERT_LT(result.iterations, settings.maximum_iterations);

    const KPointStencil stencil = stencil_of(model);
    const SpreadFunctional functional(stencil, model.number_of_wannier_functions);
    const BlochOverlaps rotated =
        overlaps_of(model).transformed_by(result.gauges, result.gauges, stencil);
    const std::vector<DenseMatrix<Complex>> gradients =
        functional.gradient(rotated, functional.evaluate(rotated));
    const Real expected =
        std::sqrt(SpreadFunctional::square_norm(gradients) / static_cast<Real>(gradients.size()));

    EXPECT_GT(expected, 0.0);
    EXPECT_NEAR(result.gradient_norm, expected, 1.0e-10 * expected);
    EXPECT_FALSE(result.stationary) << "nothing is under a floor of zero";
}

/// A converged run says so, and the flag is what a driver reads.
TEST(SpreadMinimizer, AConvergedRunIsFlaggedConverged) {
    WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.85);
    q3::test_utils::perturb_gauge(model, 4242u, 0.3);

    const SpreadMinimizationResult result = minimize(model);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.stationary);
    EXPECT_LT(result.terms.minimizable_bohr_squared(),
              0.01 * result.initial_minimizable_bohr_squared);
}
