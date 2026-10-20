#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/real_space/anderson.hpp"

using namespace q3;

// ── Construction and name ───────────────────────────────────────────

TEST(AndersonMixer, NameReturnsAnderson) {
    AndersonMixer mixer;
    EXPECT_EQ(mixer.name(), "anderson");
}

TEST(AndersonMixer, DefaultMixingParameterIsZeroPointSeven) {
    AndersonMixer mixer;
    EXPECT_DOUBLE_EQ(mixer.mixing_parameter(), 0.7);
}

TEST(AndersonMixer, DefaultHistoryDepthIsEight) {
    AndersonMixer mixer;
    EXPECT_EQ(mixer.history_depth(), 8);
}

TEST(AndersonMixer, CustomMixingParameterAndHistoryDepth) {
    AndersonMixer mixer(0.3, 4);
    EXPECT_DOUBLE_EQ(mixer.mixing_parameter(), 0.3);
    EXPECT_EQ(mixer.history_depth(), 4);
}

// ── Single-iteration linear mixing fallback ─────────────────────────

TEST(AndersonMixer, FirstIterationPerformsLinearMixing) {
    Real mixing_parameter = 0.5;
    AndersonMixer mixer(mixing_parameter, 8);

    /**
     * input = [1.0, 2.0, 3.0], output = [2.0, 3.0, 5.0]
     * residual = output - input = [1.0, 1.0, 2.0]
     * mixed = input + beta * residual = [1.5, 2.5, 4.0]
     */
    std::vector<Real> input_density = {1.0, 2.0, 3.0};
    std::vector<Real> output_density = {2.0, 3.0, 5.0};

    auto mixed = mixer.mix_one_component(input_density, output_density);

    ASSERT_EQ(mixed.size(), 3u);
    EXPECT_NEAR(mixed[0], 1.5, 1e-12);
    EXPECT_NEAR(mixed[1], 2.5, 1e-12);
    EXPECT_NEAR(mixed[2], 4.0, 1e-12);
}

// ── Two-iteration Anderson mixing ───────────────────────────────────

TEST(AndersonMixer, SecondIterationUsesAndersonFormula) {
    Real mixing_parameter = 0.7;
    AndersonMixer mixer(mixing_parameter, 8);

    // First iteration: linear mixing
    std::vector<Real> input_density_1 = {1.0, 2.0};
    std::vector<Real> output_density_1 = {1.2, 2.3};
    auto mixed_1 = mixer.mix_one_component(input_density_1, output_density_1);

    // Second iteration: Anderson mixing should kick in (2 history entries)
    std::vector<Real> input_density_2 = mixed_1;
    std::vector<Real> output_density_2 = {1.1, 2.15};
    auto mixed_2 = mixer.mix_one_component(input_density_2, output_density_2);

    // Just verify it produces a result of the right size and finite values
    ASSERT_EQ(mixed_2.size(), 2u);
    EXPECT_TRUE(std::isfinite(mixed_2[0]));
    EXPECT_TRUE(std::isfinite(mixed_2[1]));
}

// ── Convergence behavior ────────────────────────────────────────────

TEST(AndersonMixer, MixingConvergesForSimpleLinearSystem) {
    /**
     * Test that Anderson mixing converges when iteratively mixing
     * toward a fixed target density. We simulate: output = 0.5*(input + target),
     * which is a simple contraction map.
     */
    Real mixing_parameter = 0.7;
    int history_depth = 4;
    AndersonMixer mixer(mixing_parameter, history_depth);

    int number_of_points = 10;
    std::vector<Real> target_density(number_of_points);
    for (int point_index = 0; point_index < number_of_points; ++point_index) {
        target_density[point_index] = std::sin(0.5 * point_index);
    }

    std::vector<Real> current_density(number_of_points, 0.0);

    for (int iteration = 0; iteration < 50; ++iteration) {
        // Simulate a DFT-like output: output = 0.5*(input + target)
        std::vector<Real> output_density(number_of_points);
        for (int point_index = 0; point_index < number_of_points; ++point_index) {
            output_density[point_index] =
                0.5 * (current_density[point_index] + target_density[point_index]);
        }
        current_density = mixer.mix_one_component(current_density, output_density);
    }

    // After many iterations, should be close to the target
    for (int point_index = 0; point_index < number_of_points; ++point_index) {
        EXPECT_NEAR(current_density[point_index], target_density[point_index], 1e-6);
    }
}

// ── Reset clears history ────────────────────────────────────────────

TEST(AndersonMixer, ResetClearsHistory) {
    AndersonMixer mixer(0.7, 8);

    std::vector<Real> input_density = {1.0, 2.0, 3.0};
    std::vector<Real> output_density = {2.0, 3.0, 5.0};
    mixer.mix_one_component(input_density, output_density);
    mixer.mix_one_component(input_density, output_density);

    mixer.reset();

    // After reset, the next call should behave like the first iteration (linear mixing)
    auto mixed = mixer.mix_one_component(input_density, output_density);
    Real expected_0 = input_density[0] + 0.7 * (output_density[0] - input_density[0]);
    EXPECT_NEAR(mixed[0], expected_0, 1e-12);
}

// ── Mix via the base-class interface ────────────────────────────────

TEST(AndersonMixer, MixDispatchesPerSpinComponent) {
    AndersonMixer mixer(0.5, 8);

    int number_of_spin_components = 2;
    std::vector<std::vector<Real>> input_density = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<Real>> output_density = {{1.5, 2.5}, {3.5, 4.5}};

    mixer.mix(input_density, output_density, number_of_spin_components);

    /**
     * After first-iteration linear mixing: mixed = input + beta * (output_orig - input)
     * For spin 0: [1.0 + 0.5*0.5, 2.0 + 0.5*0.5] = [1.25, 2.25]
     * For spin 1: [3.0 + 0.5*0.5, 4.0 + 0.5*0.5] = [3.25, 4.25]
     */
    ASSERT_EQ(output_density.size(), 2u);
    EXPECT_NEAR(output_density[0][0], 1.25, 1e-12);
    EXPECT_NEAR(output_density[0][1], 2.25, 1e-12);
    EXPECT_NEAR(output_density[1][0], 3.25, 1e-12);
    EXPECT_NEAR(output_density[1][1], 4.25, 1e-12);
}

// ── Identical input and output gives back the same density ──────────

TEST(AndersonMixer, IdenticalInputAndOutputReturnsInput) {
    AndersonMixer mixer(0.7, 8);

    std::vector<Real> density = {1.0, 2.0, 3.0};
    auto mixed = mixer.mix_one_component(density, density);

    // Residual is zero, so mixed = input + beta * 0 = input
    ASSERT_EQ(mixed.size(), 3u);
    EXPECT_DOUBLE_EQ(mixed[0], 1.0);
    EXPECT_DOUBLE_EQ(mixed[1], 2.0);
    EXPECT_DOUBLE_EQ(mixed[2], 3.0);
}

// ── History trimming respects depth ─────────────────────────────────

TEST(AndersonMixer, HistoryIsTrimmedToConfiguredDepth) {
    int history_depth = 2;
    AndersonMixer mixer(0.5, history_depth);

    std::vector<Real> input = {1.0};
    std::vector<Real> output = {2.0};

    // Perform more iterations than the history depth
    for (int iteration = 0; iteration < 10; ++iteration) {
        auto mixed = mixer.mix_one_component(input, output);
        // Just verify finite results every time
        ASSERT_EQ(mixed.size(), 1u);
        EXPECT_TRUE(std::isfinite(mixed[0]));
        input = mixed;
    }
}

// ── Single-element density vectors ──────────────────────────────────

TEST(AndersonMixer, SingleElementDensityVectors) {
    AndersonMixer mixer(0.5, 4);

    std::vector<Real> input = {10.0};
    std::vector<Real> output = {20.0};

    auto mixed = mixer.mix_one_component(input, output);
    // Linear: 10.0 + 0.5 * (20.0 - 10.0) = 15.0
    ASSERT_EQ(mixed.size(), 1u);
    EXPECT_NEAR(mixed[0], 15.0, 1e-12);
}
