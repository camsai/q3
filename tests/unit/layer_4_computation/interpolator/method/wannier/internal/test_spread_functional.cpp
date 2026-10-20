#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <string>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/unitary_exponential.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_functional.hpp"
#include "utils/wannier_chain_model.hpp"

using namespace q3;
using q3::test_utils::make_wannier_chain;
using q3::test_utils::overlaps_of;
using q3::test_utils::rotate_block;
using q3::test_utils::stencil_of;
using q3::test_utils::WannierChainModel;

namespace {

/// Omega_D + Omega_OD of the model rotated by exp(t D) at every k-point.
Real minimizable_spread_along(const WannierChainModel& model,
                              const std::vector<std::vector<Complex>>& directions,
                              Real step) {
    const int dimension = model.number_of_wannier_functions;
    std::vector<std::vector<Complex>> rotations;
    rotations.reserve(directions.size());
    for (const std::vector<Complex>& direction : directions) {
        const DenseMatrix<Complex> generator(dimension, dimension, direction);
        rotations.push_back(exponential_of_anti_hermitian(generator.scaled(step)).raw());
    }

    std::vector<std::vector<std::vector<Complex>>> rotated(model.overlaps.size());
    for (std::size_t kpoint = 0; kpoint < model.overlaps.size(); kpoint++) {
        for (std::size_t direction = 0; direction < model.overlaps[kpoint].size(); direction++) {
            const std::size_t neighbor =
                static_cast<std::size_t>(model.neighbor_kpoint_index[kpoint][direction]);
            rotated[kpoint].push_back(rotate_block(model.overlaps[kpoint][direction],
                                                   rotations[kpoint],
                                                   rotations[neighbor],
                                                   dimension));
        }
    }
    const KPointStencil stencil = stencil_of(model);
    return SpreadFunctional(stencil, dimension)
        .evaluate(BlochOverlaps::square(rotated, dimension))
        .minimizable_bohr_squared();
}

/// A seeded anti-Hermitian matrix per k-point, to probe the gradient along.
std::vector<std::vector<Complex>>
random_anti_hermitian_directions(int number_of_kpoints, int dimension, unsigned int seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<Real> gaussian(0.0, 1.0);
    std::vector<std::vector<Complex>> directions;
    for (int kpoint = 0; kpoint < number_of_kpoints; kpoint++) {
        std::vector<Complex> matrix(static_cast<std::size_t>(dimension) * dimension);
        for (Complex& element : matrix) {
            element = Complex(gaussian(generator), gaussian(generator));
        }
        std::vector<Complex> anti(matrix.size());
        for (int column = 0; column < dimension; column++) {
            for (int row = 0; row < dimension; row++) {
                const std::size_t index = static_cast<std::size_t>(column) * dimension + row;
                const std::size_t transposed = static_cast<std::size_t>(row) * dimension + column;
                anti[index] = 0.5 * (matrix[index] - std::conj(matrix[transposed]));
            }
        }
        directions.push_back(std::move(anti));
    }
    return directions;
}

/// <G, D> = sum_k Re tr(G_k^dagger D_k): the change of the functional per unit step along D.
Real directional_derivative(const std::vector<DenseMatrix<Complex>>& gradients,
                            const std::vector<std::vector<Complex>>& directions) {
    Real predicted = 0.0;
    for (std::size_t kpoint = 0; kpoint < gradients.size(); kpoint++) {
        const std::vector<Complex>& gradient = gradients[kpoint].raw();
        for (std::size_t index = 0; index < gradient.size(); index++) {
            predicted += (std::conj(gradient[index]) * directions[kpoint][index]).real();
        }
    }
    return predicted;
}

WannierChainModel scrambled_chain(unsigned int seed) {
    WannierChainModel model = make_wannier_chain(6, {0.0, 2.5, -2.0}, 7.5, 0.85);
    q3::test_utils::scramble_gauge(model, seed);
    return model;
}

} // namespace

/**
 * The gate on the whole construction: if the analytic gradient is not the
 * derivative of the spread, every step the minimiser takes is a step along
 * something else. A wrong sign, a wrong factor and a transposed index all
 * survive a "does it converge" test and all fail here.
 */
TEST(SpreadFunctional, TheAnalyticGradientIsTheDerivativeOfTheSpread) {
    const WannierChainModel model = scrambled_chain(20260905u);
    const int dimension = model.number_of_wannier_functions;
    const KPointStencil stencil = stencil_of(model);
    const SpreadFunctional functional(stencil, dimension);

    const SpreadTerms terms = functional.evaluate(overlaps_of(model));
    const std::vector<DenseMatrix<Complex>> gradients =
        functional.gradient(overlaps_of(model), terms);

    const std::vector<std::vector<Complex>> directions =
        random_anti_hermitian_directions(static_cast<int>(model.overlaps.size()), dimension, 4242u);
    const Real predicted = directional_derivative(gradients, directions);

    const Real delta = 1.0e-5;
    const Real measured = (minimizable_spread_along(model, directions, delta) -
                           minimizable_spread_along(model, directions, -delta)) /
                          (2.0 * delta);

    EXPECT_NEAR(measured, predicted, 1.0e-4 * std::max(1.0, std::abs(predicted)))
        << "central difference " << measured << " against analytic " << predicted;
}

/**
 * The same check where the diagonal phases are LARGE. Im ln M_nn lives on
 * (-pi, pi], so a centre far from the origin on a coarse mesh drives the
 * residual toward the branch cut, and a gradient that is right for small
 * phases can be wrong there -- which is exactly the regime a real supercell
 * lands in, and where the minimiser has been seen to stall.
 */
TEST(SpreadFunctional, TheGradientSurvivesLargeDiagonalPhases) {
    // Four k-points and centres near half the cell: b . r approaches pi/2.
    WannierChainModel model = make_wannier_chain(4, {0.0, 2.6, -2.4}, 6.0, 0.85);
    q3::test_utils::scramble_gauge(model, 606u);
    const int dimension = model.number_of_wannier_functions;
    const KPointStencil stencil = stencil_of(model);
    const SpreadFunctional functional(stencil, dimension);

    const SpreadTerms terms = functional.evaluate(overlaps_of(model));
    const std::vector<DenseMatrix<Complex>> gradients =
        functional.gradient(overlaps_of(model), terms);
    const std::vector<std::vector<Complex>> directions =
        random_anti_hermitian_directions(static_cast<int>(model.overlaps.size()), dimension, 171u);
    const Real predicted = directional_derivative(gradients, directions);

    const Real delta = 1.0e-5;
    const Real measured = (minimizable_spread_along(model, directions, delta) -
                           minimizable_spread_along(model, directions, -delta)) /
                          (2.0 * delta);

    EXPECT_NEAR(measured, predicted, 1.0e-4 * std::max(1.0, std::abs(predicted)))
        << "central difference " << measured << " against analytic " << predicted;
}

/// The descent direction has to actually descend, not merely be a derivative.
TEST(SpreadFunctional, MovingAgainstTheGradientLowersTheSpread) {
    const WannierChainModel model = scrambled_chain(77u);
    const int dimension = model.number_of_wannier_functions;
    const KPointStencil stencil = stencil_of(model);
    const SpreadFunctional functional(stencil, dimension);

    const SpreadTerms terms = functional.evaluate(overlaps_of(model));
    std::vector<std::vector<Complex>> descent;
    for (const DenseMatrix<Complex>& gradient : functional.gradient(overlaps_of(model), terms)) {
        std::vector<Complex> negated = gradient.raw();
        for (Complex& element : negated) {
            element = -element;
        }
        descent.push_back(std::move(negated));
    }

    const Real before = terms.minimizable_bohr_squared();
    const Real after = minimizable_spread_along(model, descent, 1.0e-4);
    EXPECT_LT(after, before);
}

/**
 * Omega_I is the part no gauge can touch. It is also the part the disentangler
 * spent its whole iteration minimising, so a localiser that moved it would be
 * undoing that work invisibly -- the total spread would fall while the model
 * got worse.
 */
TEST(SpreadFunctional, TheGaugeInvariantSpreadIsInvariantUnderAGaugeRotation) {
    const WannierChainModel plain = make_wannier_chain(6, {0.0, 2.5, -2.0}, 7.5, 0.85);
    const WannierChainModel scrambled = scrambled_chain(9001u);
    const int dimension = plain.number_of_wannier_functions;

    const KPointStencil stencil = stencil_of(plain);
    const SpreadTerms before = SpreadFunctional(stencil, dimension).evaluate(overlaps_of(plain));
    const SpreadTerms after = SpreadFunctional(stencil, dimension).evaluate(overlaps_of(scrambled));

    EXPECT_NEAR(after.gauge_invariant_bohr_squared, before.gauge_invariant_bohr_squared, 1.0e-10);
    EXPECT_GT(after.minimizable_bohr_squared(), before.minimizable_bohr_squared())
        << "scrambling the gauge must raise the part a gauge can change";
}

/**
 * The chain is built so that its unscrambled form is exactly localised: no
 * off-diagonal weight, and every diagonal phase already the one its centre
 * implies.
 */
TEST(SpreadFunctional, TheChainsOwnGaugeHasNoMinimizableSpreadAndRecoversItsCenters) {
    const WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.8);
    const int dimension = model.number_of_wannier_functions;
    const KPointStencil stencil = stencil_of(model);

    const SpreadTerms terms = SpreadFunctional(stencil, dimension).evaluate(overlaps_of(model));

    EXPECT_NEAR(terms.minimizable_bohr_squared(), 0.0, 1.0e-12);
    EXPECT_GT(terms.gauge_invariant_bohr_squared, 0.0);
    for (int wannier = 0; wannier < dimension; wannier++) {
        EXPECT_NEAR(terms.centers_bohr[static_cast<std::size_t>(wannier)][0],
                    model.centers[static_cast<std::size_t>(wannier)],
                    1.0e-10);
        EXPECT_NEAR(terms.centers_bohr[static_cast<std::size_t>(wannier)][1], 0.0, 1.0e-12);
    }
}

/**
 * Omega_n = <r^2>_n - |<r>_n|^2 per function. On the chain's own gauge every
 * function has the same diagonal modulus rho and no off-diagonal weight, so
 * each one carries an equal share (1 - rho^2) sum_b w_b of Omega_I and nothing
 * else; on a scrambled gauge the shares differ, but they still add up to the
 * total, which is the identity that ties the two decompositions together.
 */
TEST(SpreadFunctional, PerFunctionSpreadsShareTheTotalAndSumToIt) {
    const WannierChainModel model = make_wannier_chain(8, {0.0, 3.0, -1.5}, 6.0, 0.8);
    const int dimension = model.number_of_wannier_functions;
    const KPointStencil stencil = stencil_of(model);
    const SpreadTerms terms = SpreadFunctional(stencil, dimension).evaluate(overlaps_of(model));

    ASSERT_EQ(terms.spreads_bohr_squared.size(), static_cast<std::size_t>(dimension));
    Real sum = 0.0;
    for (const Real spread : terms.spreads_bohr_squared) {
        EXPECT_NEAR(spread, terms.total_bohr_squared() / dimension, 1.0e-10);
        sum += spread;
    }
    EXPECT_NEAR(sum, terms.total_bohr_squared(), 1.0e-10);

    const WannierChainModel scrambled = scrambled_chain(4242u);
    const KPointStencil scrambled_stencil = stencil_of(scrambled);
    const SpreadTerms mixed =
        SpreadFunctional(scrambled_stencil, dimension).evaluate(overlaps_of(scrambled));
    Real mixed_sum = 0.0;
    Real spread_range = 0.0;
    for (const Real spread : mixed.spreads_bohr_squared) {
        mixed_sum += spread;
        spread_range =
            std::max(spread_range, std::abs(spread - mixed.total_bohr_squared() / dimension));
    }
    EXPECT_NEAR(mixed_sum, mixed.total_bohr_squared(), 1.0e-9 * mixed.total_bohr_squared());
    EXPECT_GT(spread_range, 1.0e-6) << "a scrambled gauge spreads the functions unequally";
}

TEST(SpreadFunctional, ACollapsedGaugeIsRefusedRatherThanReturningNoise) {
    WannierChainModel model = make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9);
    model.overlaps[1][0][0] = Complex(0.0, 0.0);
    const KPointStencil stencil = stencil_of(model);
    const SpreadFunctional functional(stencil, 2);

    try {
        const std::vector<DenseMatrix<Complex>> gradient =
            functional.gradient(overlaps_of(model), functional.evaluate(overlaps_of(model)));
        FAIL() << "a vanishing diagonal overlap has no Berry phase";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::gauge_collapsed);
    }
}

TEST(SpreadFunctional, AMisshapenOverlapBlockIsRefused) {
    WannierChainModel model = make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9);
    model.overlaps[0][0].pop_back();
    const KPointStencil stencil = stencil_of(model);

    EXPECT_THROW(SpreadFunctional(stencil, 2).evaluate(overlaps_of(model)), std::invalid_argument);
}

TEST(SpreadFunctional, AStencilWithoutBVectorsCannotCarryASpread) {
    const WannierChainModel model = make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9);
    const KPointStencil without_b =
        KPointStencil::from_table(model.neighbor_kpoint_index, model.weights);
    EXPECT_THROW(SpreadFunctional(without_b, 2), std::invalid_argument);
}
