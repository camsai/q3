#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/maximally_localized_gauge_choice.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_minimizer.hpp"
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

WannierGauge localize(const WannierConstructionInputs& inputs) {
    const WannierSubspace subspace = IsolatedManifoldSelection().choose(inputs);
    return MaximallyLocalizedGaugeChoice(SpreadMinimizationSettings{}).choose(inputs, subspace);
}

} // namespace

/**
 * The same perturbation test_spread_minimizer recovers from, read through a
 * record: the projection gauge of an identity projection is the identity,
 * so the choice starts where the scrambled overlaps put it and must carry
 * them back to the chain's minimum, keeping Omega_I and the frames unitary
 * on the way.
 */
TEST(MaximallyLocalizedGaugeChoice, APerturbedGaugeIsCarriedBackToTheChainsMinimum) {
    WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.85);
    q3::test_utils::perturb_gauge(model, 12345u, 0.35);
    const WannierInputs record = inputs_of(model);
    const WannierConstructionInputs inputs(record);
    const WannierGauge gauge = localize(inputs);

    const SpreadReport& spread = gauge.spread;
    EXPECT_TRUE(spread.evaluated);
    EXPECT_TRUE(spread.localized);
    EXPECT_TRUE(spread.localization_converged);
    EXPECT_GT(spread.localization_iterations, 0);
    EXPECT_FALSE(spread.guided_by_trial_orbitals);
    EXPECT_GT(spread.initial_minimizable_bohr_squared, 0.1)
        << "the perturbation should have left real work to do";
    const Real minimizable = spread.total_bohr_squared - spread.gauge_invariant_bohr_squared;
    EXPECT_LT(minimizable, 1.0e-4 * spread.initial_minimizable_bohr_squared)
        << "started at " << spread.initial_minimizable_bohr_squared << " bohr^2 and stopped at "
        << minimizable;
    EXPECT_NEAR(
        spread.gauge_invariant_bohr_squared, chain_gauge_invariant_spread(model, 0.85), 1.0e-10);

    ASSERT_EQ(gauge.frames.size(), 8u);
    for (const DenseMatrix<Complex>& frame : gauge.frames) {
        const DenseMatrix<Complex> gram =
            multiply(MatrixOperation::ConjugateTranspose, frame, MatrixOperation::None, frame);
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                const Complex expected = row == column ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
                EXPECT_NEAR(std::abs(gram(row, column) - expected), 0.0, 1.0e-10);
            }
        }
    }
}

TEST(MaximallyLocalizedGaugeChoice, RefusesARecordWithoutOverlaps) {
    WannierInputs record = inputs_of(make_wannier_chain(6, {0.0, 2.0}, 5.5, 0.9));
    record.overlap_matrices.clear();
    const WannierConstructionInputs inputs(record);
    try {
        const WannierGauge gauge = localize(inputs);
        FAIL() << "there is nothing to minimise without the overlaps";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::localization_unsupported);
        const std::string message = error.what();
        EXPECT_NE(message.find("nothing to minimise"), std::string::npos) << message;
    }
}

TEST(MaximallyLocalizedGaugeChoice, RefusesAMeshWithATwoPointAxis) {
    const WannierInputs record = inputs_of(make_wannier_chain(2, {0.0, 2.0}, 5.5, 0.9));
    const WannierConstructionInputs inputs(record);
    try {
        const WannierGauge gauge = localize(inputs);
        FAIL() << "the centres are Berry phases, undefined on a two-point axis";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::two_point_axis);
        const std::string message = error.what();
        EXPECT_NE(message.find("reaches the same k-point as its opposite"), std::string::npos)
            << message;
    }
}
