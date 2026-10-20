#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/reciprocal_space/broyden.hpp"

using namespace q3;

// ── Construction and accessors ──────────────────────────────────────

TEST(BroydenMixer, DefaultConstructionSetsDefaults) {
    BroydenMixer mixer;
    EXPECT_DOUBLE_EQ(mixer.mixing_parameter(), 0.7);
    EXPECT_EQ(mixer.history_depth(), 8);
}

TEST(BroydenMixer, CustomParametersAreStored) {
    BroydenMixer mixer(0.3, 4);
    EXPECT_DOUBLE_EQ(mixer.mixing_parameter(), 0.3);
    EXPECT_EQ(mixer.history_depth(), 4);
}

TEST(BroydenMixer, NameReturnsBroyden) {
    BroydenMixer mixer;
    EXPECT_EQ(mixer.name(), "broyden");
}

TEST(BroydenMixer, LastResidualNormInitiallyNegative) {
    BroydenMixer mixer;
    EXPECT_LT(mixer.last_residual_norm(), 0.0);
}

// ── First iteration: linear mixing ─────────────────────────────────

/**
 * On the very first call, the Broyden mixer has no history and must
 * fall back to linear mixing: rho_new = rho_in + alpha * (rho_out - rho_in).
 *
 * We set up a trivial single-spin, single-G-vector scenario with no metric
 * complications to verify the linear mixing formula.
 */
TEST(BroydenMixer, FirstIterationAppliesLinearMixing) {
    Real mixing_parameter = 0.5;
    BroydenMixer mixer(mixing_parameter, 4);

    // Minimal environment: 1 G-vector (G=0 excluded), with unit metric.
    MixerEnvironment environment;
    environment.charge_metric_weights = {0.0, 1.0};
    environment.first_nonzero_g_index = 1;
    environment.is_gamma_only = false;
    environment.number_of_smooth_g_vectors = 0; // 0 means use all
    mixer.initialize(environment);

    // Single spin, 2 G-vectors (G=0 is zeroed out by metric)
    Complex rho_in_g0(0.5, 0.0);
    Complex rho_in_g1(1.0, 0.0);
    Complex rho_out_g0(0.5, 0.0);
    Complex rho_out_g1(3.0, 0.0);

    std::vector<std::vector<Complex>> rho_in_g = {{rho_in_g0, rho_in_g1}};
    std::vector<std::vector<Complex>> rho_out_g = {{rho_out_g0, rho_out_g1}};

    mixer.mix_in_reciprocal_space(rho_in_g, rho_out_g, 1);

    // For G=0: residual = 0, so result = rho_in + alpha * 0 = 0.5
    EXPECT_NEAR(rho_in_g[0][0].real(), 0.5, 1e-12);
    // For G=1: rho_in + alpha * (rho_out - rho_in) = 1.0 + 0.5 * (3.0 - 1.0) = 2.0
    EXPECT_NEAR(rho_in_g[0][1].real(), 2.0, 1e-12);
    EXPECT_NEAR(rho_in_g[0][1].imag(), 0.0, 1e-12);
}

TEST(BroydenMixer, FirstIterationSetsResidualNorm) {
    BroydenMixer mixer(0.7, 4);

    MixerEnvironment environment;
    environment.charge_metric_weights = {0.0, 1.0};
    environment.first_nonzero_g_index = 1;
    environment.is_gamma_only = false;
    environment.number_of_smooth_g_vectors = 0;
    mixer.initialize(environment);

    std::vector<std::vector<Complex>> rho_in_g = {{Complex(0.0, 0.0), Complex(1.0, 0.0)}};
    std::vector<std::vector<Complex>> rho_out_g = {{Complex(0.0, 0.0), Complex(2.0, 0.0)}};

    mixer.mix_in_reciprocal_space(rho_in_g, rho_out_g, 1);

    // Residual norm should be positive after mixing with a non-zero residual
    EXPECT_GT(mixer.last_residual_norm(), 0.0);
}

// ── Reset ───────────────────────────────────────────────────────────

/**
 * Verify that reset() allows the mixer to start fresh. After reset,
 * the next call should behave as a first iteration (linear mixing only).
 */
TEST(BroydenMixer, ResetAllowsFreshStart) {
    BroydenMixer mixer(0.5, 4);

    MixerEnvironment environment;
    environment.charge_metric_weights = {0.0, 1.0};
    environment.first_nonzero_g_index = 1;
    environment.is_gamma_only = false;
    environment.number_of_smooth_g_vectors = 0;
    mixer.initialize(environment);

    // Run two iterations to build up history
    std::vector<std::vector<Complex>> rho_in_g = {{Complex(0.0, 0.0), Complex(1.0, 0.0)}};
    std::vector<std::vector<Complex>> rho_out_g = {{Complex(0.0, 0.0), Complex(2.0, 0.0)}};
    mixer.mix_in_reciprocal_space(rho_in_g, rho_out_g, 1);

    rho_out_g = {{Complex(0.0, 0.0), Complex(2.5, 0.0)}};
    mixer.mix_in_reciprocal_space(rho_in_g, rho_out_g, 1);

    mixer.reset();

    // After reset, perform a first iteration that should be simple linear mixing
    std::vector<std::vector<Complex>> rho_in_fresh = {{Complex(0.0, 0.0), Complex(1.0, 0.0)}};
    std::vector<std::vector<Complex>> rho_out_fresh = {{Complex(0.0, 0.0), Complex(3.0, 0.0)}};
    mixer.mix_in_reciprocal_space(rho_in_fresh, rho_out_fresh, 1);

    // Linear mixing: 1.0 + 0.5 * (3.0 - 1.0) = 2.0
    EXPECT_NEAR(rho_in_fresh[0][1].real(), 2.0, 1e-12);
}

// ── Convergence test ────────────────────────────────────────────────

/**
 * Verify that repeated Broyden mixing steps converge toward a fixed-point
 * density. With linear mixing (alpha=0.5), after many iterations the
 * density should approach the target.
 */
TEST(BroydenMixer, ConvergesTowardFixedPoint) {
    Real mixing_parameter = 0.4;
    int history_depth = 4;
    BroydenMixer mixer(mixing_parameter, history_depth);

    int number_of_g_vectors = 5;

    // G=0 is excluded from the metric (set weight=0, first_nonzero_g_index=1)
    MixerEnvironment environment;
    environment.charge_metric_weights.resize(number_of_g_vectors, 1.0);
    environment.charge_metric_weights[0] = 0.0; // exclude G=0
    environment.first_nonzero_g_index = 1;
    environment.is_gamma_only = false;
    environment.number_of_smooth_g_vectors = 0;
    mixer.initialize(environment);

    // Target density (the fixed point), G=0 is neutral
    std::vector<Complex> target = {Complex(0.0, 0.0),
                                   Complex(1.0, 0.0),
                                   Complex(2.0, 0.0),
                                   Complex(3.0, 0.0),
                                   Complex(4.0, 0.0)};

    // Start with a poor initial guess for G>0
    std::vector<std::vector<Complex>> rho_in_g = {{Complex(0.0, 0.0),
                                                   Complex(0.1, 0.0),
                                                   Complex(0.2, 0.0),
                                                   Complex(0.3, 0.0),
                                                   Complex(0.4, 0.0)}};

    int number_of_iterations = 30;
    for (int iteration = 0; iteration < number_of_iterations; iteration++) {
        // rho_out is always the target (simulating a trivially linear problem)
        std::vector<std::vector<Complex>> rho_out_g = {target};
        mixer.mix_in_reciprocal_space(rho_in_g, rho_out_g, 1);
    }

    /**
     * After many iterations with alpha * (target - current) updates,
     * the density should converge close to the target.
     */
    for (int g_index = 1; g_index < number_of_g_vectors; g_index++) {
        EXPECT_NEAR(rho_in_g[0][g_index].real(), target[g_index].real(), 0.01)
            << "G-vector index " << g_index << " did not converge to target";
    }
}

// ── Gamma-only mode ─────────────────────────────────────────────────

TEST(BroydenMixer, GammaOnlyDoublesResidualNorm) {
    BroydenMixer mixer_gamma(0.5, 4);
    BroydenMixer mixer_no_gamma(0.5, 4);

    MixerEnvironment environment_gamma;
    environment_gamma.charge_metric_weights = {0.0, 1.0};
    environment_gamma.first_nonzero_g_index = 1;
    environment_gamma.is_gamma_only = true;
    environment_gamma.number_of_smooth_g_vectors = 0;
    mixer_gamma.initialize(environment_gamma);

    MixerEnvironment environment_no_gamma;
    environment_no_gamma.charge_metric_weights = {0.0, 1.0};
    environment_no_gamma.first_nonzero_g_index = 1;
    environment_no_gamma.is_gamma_only = false;
    environment_no_gamma.number_of_smooth_g_vectors = 0;
    mixer_no_gamma.initialize(environment_no_gamma);

    std::vector<std::vector<Complex>> rho_in_g1 = {{Complex(0.0, 0.0), Complex(1.0, 0.0)}};
    std::vector<std::vector<Complex>> rho_out_g1 = {{Complex(0.0, 0.0), Complex(3.0, 0.0)}};
    std::vector<std::vector<Complex>> rho_in_g2 = rho_in_g1;
    std::vector<std::vector<Complex>> rho_out_g2 = rho_out_g1;

    mixer_gamma.mix_in_reciprocal_space(rho_in_g1, rho_out_g1, 1);
    mixer_no_gamma.mix_in_reciprocal_space(rho_in_g2, rho_out_g2, 1);

    // Gamma-only residual norm should be 2x the non-gamma version
    EXPECT_NEAR(mixer_gamma.last_residual_norm(), 2.0 * mixer_no_gamma.last_residual_norm(), 1e-12);
}

/**
 * ── G-space BLAS acceleration regression (TODOs.md §2.1) ────────────
 *
 * BroydenMixer::mix_in_reciprocal_space precomputes a metric-scaled copy of
 * each history vector once, then reduces the Gram matrix / projection dot
 * products to plain complex conjugate dot products (BLAS Level-1 style) and
 * the gamma-correction / final mixing updates to BLAS ZAXPY calls, instead
 * of re-applying the per-G metric weight inside the O(history_depth^2) Gram
 * loop. This is mathematically equivalent to the original per-pair weighted
 * loop but NOT guaranteed bit-identical (floating-point multiplication does
 * not distribute exactly over addition for arbitrary, non-power-of-2 metric
 * weights) -- hence the tolerance-based (not exact-equality) checks below.
 *
 * The following test hand-computes the expected result of two mixing steps
 * with a non-uniform, non-unit metric (weights 2.0 and 5.0, deliberately
 * distinct from 1.0) so the metric_scale_factor()/build_metric_scaled_copy()
 * path is genuinely exercised, not trivially satisfied by weight==1.
 */

TEST(BroydenMixer, NonUniformMetricWeightsProduceHandVerifiedResult) {
    Real mixing_parameter = 0.5;
    int history_depth = 4;
    BroydenMixer mixer(mixing_parameter, history_depth);

    MixerEnvironment environment;
    environment.charge_metric_weights = {0.0, 2.0, 5.0};
    environment.first_nonzero_g_index = 1;
    environment.is_gamma_only = false;
    environment.number_of_smooth_g_vectors = 0;
    mixer.initialize(environment);

    std::vector<std::vector<Complex>> rho_in_g = {
        {Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)}};

    /**
     * Iteration 1: no history yet, falls back to linear mixing.
     * rho_in -> [0, 1 + 0.5*(3-1), 2 + 0.5*(5-2)] = [0, 2.0, 3.5]
     */
    std::vector<std::vector<Complex>> rho_out_g_1 = {
        {Complex(0.0, 0.0), Complex(3.0, 0.0), Complex(5.0, 0.0)}};
    mixer.mix_in_reciprocal_space(rho_in_g, rho_out_g_1, 1);
    EXPECT_NEAR(rho_in_g[0][1].real(), 2.0, 1e-9);
    EXPECT_NEAR(rho_in_g[0][2].real(), 3.5, 1e-9);

    /**
     * Iteration 2: exercises the full Gram-matrix/projection/correction path
     * (actual_history == 1, so the Gram "matrix" is the scalar 1.25, its
     * Bunch-Kaufman inverse is 0.8, gamma == -7.0) -- see the branch's PR
     * description / TODOs.md write-up for the full hand-derivation.
     */
    std::vector<std::vector<Complex>> rho_out_g_2 = {
        {Complex(0.0, 0.0), Complex(4.0, 0.0), Complex(7.0, 0.0)}};
    mixer.mix_in_reciprocal_space(rho_in_g, rho_out_g_2, 1);

    EXPECT_NEAR(rho_in_g[0][0].real(), 0.0, 1e-9);
    EXPECT_NEAR(rho_in_g[0][1].real(), -4.0, 1e-9);
    EXPECT_NEAR(rho_in_g[0][2].real(), -7.0, 1e-9);
    EXPECT_NEAR(rho_in_g[0][0].imag(), 0.0, 1e-9);
    EXPECT_NEAR(rho_in_g[0][1].imag(), 0.0, 1e-9);
    EXPECT_NEAR(rho_in_g[0][2].imag(), 0.0, 1e-9);
}

/**
 * ── Two-spin (magnetization channel) coverage ────────────────────────
 *
 * All prior tests in this file are single-spin, so the spin>=1 branch of
 * metric_scale_factor() (flat magnetization metric, no Coulomb 1/G^2 weight)
 * was previously untested. Verify convergence still holds once a real
 * magnetization channel with a nonzero metric weight is mixed alongside
 * the charge channel, through enough iterations to build up a full
 * (actual_history > 1) Gram matrix in the new precompute-and-dot-product path.
 */
TEST(BroydenMixer, TwoSpinChannelsConvergeWithMagnetizationMetric) {
    Real mixing_parameter = 0.4;
    int history_depth = 4;
    BroydenMixer mixer(mixing_parameter, history_depth);

    MixerEnvironment environment;
    environment.charge_metric_weights = {0.0, 1.0, 1.0, 1.0};
    environment.first_nonzero_g_index = 1;
    environment.is_gamma_only = false;
    environment.number_of_smooth_g_vectors = 0;
    environment.magnetization_metric_weight = 1.5;
    mixer.initialize(environment);

    std::vector<Complex> target_total = {
        Complex(0.0, 0.0), Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(0.5, 0.0)};
    std::vector<Complex> target_magnetization = {
        Complex(0.3, 0.0), Complex(0.2, 0.0), Complex(0.1, 0.0), Complex(0.05, 0.0)};

    std::vector<std::vector<Complex>> rho_in_g = {
        {Complex(0.0, 0.0), Complex(0.5, 0.0), Complex(0.2, 0.0), Complex(0.1, 0.0)},
        {Complex(0.1, 0.0), Complex(0.05, 0.0), Complex(0.02, 0.0), Complex(0.01, 0.0)}};

    int number_of_iterations = 40;
    for (int iteration = 0; iteration < number_of_iterations; iteration++) {
        std::vector<std::vector<Complex>> rho_out_g = {target_total, target_magnetization};
        mixer.mix_in_reciprocal_space(rho_in_g, rho_out_g, 2);
    }

    for (int g_index = 1; g_index < 4; g_index++) {
        EXPECT_NEAR(rho_in_g[0][g_index].real(), target_total[g_index].real(), 0.01)
            << "charge channel G-vector index " << g_index << " did not converge";
    }
    for (int g_index = 0; g_index < 4; g_index++) {
        EXPECT_NEAR(rho_in_g[1][g_index].real(), target_magnetization[g_index].real(), 0.01)
            << "magnetization channel G-vector index " << g_index << " did not converge";
    }
}
