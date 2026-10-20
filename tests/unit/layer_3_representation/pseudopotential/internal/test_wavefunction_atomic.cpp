#include <gtest/gtest.h>

#include "layer_3_representation/pseudopotential/internal/wavefunction_atomic.hpp"

using namespace q3;

// ── Default construction gives empty fields ──
TEST(WavefunctionAtomic, DefaultConstructionGivesEmptyFields) {
    WavefunctionAtomic wavefunction{};

    EXPECT_TRUE(wavefunction.label.empty());
    EXPECT_EQ(wavefunction.angular_momentum, 0);
    EXPECT_DOUBLE_EQ(wavefunction.occupations, 0.0);
    EXPECT_TRUE(wavefunction.radial_values.empty());
}
