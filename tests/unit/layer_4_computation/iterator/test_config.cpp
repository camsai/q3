#include <gtest/gtest.h>

#include "layer_4_computation/iterator/config.hpp"

using namespace q3;

// ── Default values ──────────────────────────────────────────────────

TEST(SCFIteratorConfig, DefaultCrystalPointerIsNull) {
    SCFIteratorConfig config;
    EXPECT_EQ(config.crystal_structure, nullptr);
}

TEST(SCFIteratorConfig, DefaultNumberOfSpinComponentsIsOne) {
    SCFIteratorConfig config;
    EXPECT_EQ(config.number_of_spin_components, 1);
}

TEST(SCFIteratorConfig, DefaultConvergenceThreshold) {
    SCFIteratorConfig config;
    EXPECT_DOUBLE_EQ(config.convergence_threshold, 1e-6);
}

TEST(SCFIteratorConfig, DefaultMaximumIterations) {
    SCFIteratorConfig config;
    EXPECT_EQ(config.maximum_iterations, 100);
}
