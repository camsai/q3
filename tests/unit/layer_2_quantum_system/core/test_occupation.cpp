#include "layer_2_quantum_system/core/occupation.hpp"
#include <gtest/gtest.h>

using namespace q3;

// ─── Occupation ─────────────────────────────────────────────────────────────

TEST(Occupation, DefaultNumberOfElectronsIsZero) {
    Occupation occupation;
    EXPECT_DOUBLE_EQ(occupation.number_of_electrons, 0.0);
}

TEST(Occupation, DefaultSpinDegeneracyIsTwo) {
    Occupation occupation;
    EXPECT_DOUBLE_EQ(occupation.spin_degeneracy, 2.0);
}

TEST(Occupation, DefaultOccupiedBandsPerSpinIsZero) {
    Occupation occupation;
    EXPECT_EQ(occupation.occupied_bands_per_spin, 0);
}

TEST(Occupation, DefaultIsSmearedIsFalse) {
    Occupation occupation;
    EXPECT_FALSE(occupation.is_smeared);
}
