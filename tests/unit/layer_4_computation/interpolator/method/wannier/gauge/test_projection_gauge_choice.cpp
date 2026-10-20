#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/projection_gauge_choice.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/isolated_manifold_selection.hpp"
#include "utils/wannier_chain_model.hpp"

using namespace q3;
using q3::test_utils::inputs_of;
using q3::test_utils::make_wannier_chain;
using q3::test_utils::WannierChainModel;

namespace {

/// Omega_I of the chain in closed form: sum_b w_b N (1 - rho^2) over both directions.
Real chain_gauge_invariant_spread(const WannierChainModel& model, Real diagonal_modulus) {
    Real total = 0.0;
    for (const Real weight : model.weights) {
        total += weight * static_cast<Real>(model.number_of_wannier_functions) *
                 (1.0 - diagonal_modulus * diagonal_modulus);
    }
    return total;
}

} // namespace

TEST(ProjectionGaugeChoice, WithoutOverlapsTheFramesPassThroughAndNoSpreadIsClaimed) {
    WannierInputs record = inputs_of(make_wannier_chain(6, {0.0, 2.0}, 5.5, 0.9));
    record.overlap_matrices.clear();
    const WannierConstructionInputs inputs(record);
    const WannierSubspace subspace = IsolatedManifoldSelection().choose(inputs);
    const WannierGauge gauge = ProjectionGaugeChoice().choose(inputs, subspace);
    EXPECT_EQ(gauge.frames, subspace.frames);
    EXPECT_FALSE(gauge.spread.evaluated);
    EXPECT_FALSE(gauge.spread.localized);
    EXPECT_DOUBLE_EQ(gauge.spread.gauge_invariant_bohr_squared, 0.0);
}

/**
 * The chain's own gauge, read through a record: the projection gauge of an
 * identity projection is the identity, so the spread the choice reports is
 * the chain's -- Omega_I in closed form, nothing minimisable, and the
 * centres where the chain put them. What this pins beyond
 * test_spread_functional is the route: record to stencil to frames to
 * report.
 */
TEST(ProjectionGaugeChoice, TheChainsOwnGaugeReportsItsSpreadAndCenters) {
    const WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.8);
    const WannierInputs record = inputs_of(model);
    const WannierConstructionInputs inputs(record);
    const WannierSubspace subspace = IsolatedManifoldSelection().choose(inputs);
    const WannierGauge gauge = ProjectionGaugeChoice().choose(inputs, subspace);

    const SpreadReport& spread = gauge.spread;
    EXPECT_TRUE(spread.evaluated);
    EXPECT_FALSE(spread.localized);
    EXPECT_FALSE(spread.guided_by_trial_orbitals);
    EXPECT_TRUE(spread.trial_orbital_centers_bohr.empty());
    EXPECT_NEAR(
        spread.gauge_invariant_bohr_squared, chain_gauge_invariant_spread(model, 0.8), 1.0e-10);
    EXPECT_NEAR(spread.total_bohr_squared, spread.gauge_invariant_bohr_squared, 1.0e-12);
    EXPECT_NEAR(spread.initial_minimizable_bohr_squared, 0.0, 1.0e-12);
    ASSERT_EQ(spread.centers_bohr.size(), 3u);
    ASSERT_EQ(spread.spreads_bohr_squared.size(), 3u);
    for (std::size_t wannier = 0; wannier < 3; wannier++) {
        EXPECT_NEAR(spread.centers_bohr[wannier][0], model.centers[wannier], 1.0e-10);
        EXPECT_NEAR(spread.centers_bohr[wannier][1], 0.0, 1.0e-12);
        EXPECT_NEAR(spread.centers_bohr[wannier][2], 0.0, 1.0e-12);
    }
}

TEST(ProjectionGaugeChoice, TrialOrbitalCentersInTheRecordGuideTheBranches) {
    const WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.8);
    WannierInputs record = inputs_of(model);
    record.trial_orbital_centers_bohr = {
        Vector3D(0.0, 0.0, 0.0), Vector3D(3.0, 0.0, 0.0), Vector3D(-1.5, 0.0, 0.0)};
    const WannierConstructionInputs inputs(record);
    const WannierSubspace subspace = IsolatedManifoldSelection().choose(inputs);
    const WannierGauge gauge = ProjectionGaugeChoice().choose(inputs, subspace);

    EXPECT_TRUE(gauge.spread.guided_by_trial_orbitals);
    ASSERT_EQ(gauge.spread.trial_orbital_centers_bohr.size(), 3u);
    ASSERT_EQ(gauge.spread.centers_bohr.size(), 3u);
    for (std::size_t wannier = 0; wannier < 3; wannier++) {
        EXPECT_NEAR(gauge.spread.centers_bohr[wannier][0], model.centers[wannier], 1.0e-10);
    }
}
