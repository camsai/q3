#include <gtest/gtest.h>

#include "layer_4_computation/effective_potential/planewave/result.hpp"

using namespace q3;

// ── Default construction ────────────────────────────────────────────

TEST(EffectivePotentialResult, DefaultConstructionYieldsZeroEnergies) {
    EffectivePotentialResult result;
    EXPECT_DOUBLE_EQ(result.hartree_energy, 0.0);
    EXPECT_DOUBLE_EQ(result.exchange_correlation_energy, 0.0);
    EXPECT_DOUBLE_EQ(result.exchange_correlation_potential_energy, 0.0);
}
