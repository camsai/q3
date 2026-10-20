#include <gtest/gtest.h>

#include "io/input/data.hpp"

using namespace q3;

// ─── InputData Default Value Tests ──────────────────────────────────────────

TEST(InputData, DefaultTheoryIsKohnSham) {
    InputData data;
    EXPECT_EQ(data.theory, "kohn_sham");
}

TEST(InputData, DefaultConvergenceThreshold) {
    InputData data;
    EXPECT_DOUBLE_EQ(data.convergence_threshold, 1e-6);
}

TEST(InputData, DefaultMaximumIterations) {
    InputData data;
    EXPECT_EQ(data.maximum_iterations, 100);
}

/**
 * No default of its own: an input that names no functional takes the one its
 * pseudopotentials state, which parse_input() fills in.
 */
TEST(InputData, DefaultExchangeCorrelationFunctionalIsUnset) {
    InputData data;
    EXPECT_TRUE(data.exchange_correlation_functional.empty());
}

TEST(InputData, DefaultNumberOfSpinComponents) {
    InputData data;
    EXPECT_EQ(data.number_of_spin_components, 1);
}

TEST(InputData, DefaultMixingMode) {
    InputData data;
    EXPECT_EQ(data.mixing_mode, "broyden");
}

TEST(InputData, DefaultMixingBeta) {
    InputData data;
    EXPECT_DOUBLE_EQ(data.mixing_beta, 0.7);
}

TEST(InputData, DefaultSmearing) {
    InputData data;
    EXPECT_EQ(data.smearing, "none");
}

TEST(InputData, DefaultWavefunctionEnergyCutoff) {
    InputData data;
    EXPECT_DOUBLE_EQ(data.wavefunction_energy_cutoff, 30.0);
}

TEST(InputData, DefaultDiagonalizationMethod) {
    InputData data;
    EXPECT_EQ(data.diagonalization_method, "davidson");
}

TEST(InputData, DefaultCalculation) {
    InputData data;
    EXPECT_EQ(data.calculation, "self_consistent_field");
}
