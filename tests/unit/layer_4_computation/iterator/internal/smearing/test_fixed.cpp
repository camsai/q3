#include <gtest/gtest.h>

#include "layer_4_computation/iterator/internal/smearing/fixed.hpp"

using namespace q3;

TEST(FixedOccupation, BelowFermiIsFullyOccupied) {
    FixedOccupation fixed;
    EXPECT_DOUBLE_EQ(fixed.occupation(-1.0), 1.0);
    EXPECT_DOUBLE_EQ(fixed.occupation(-0.001), 1.0);
    EXPECT_DOUBLE_EQ(fixed.occupation(-100.0), 1.0);
}

TEST(FixedOccupation, AboveFermiIsEmpty) {
    FixedOccupation fixed;
    EXPECT_DOUBLE_EQ(fixed.occupation(1.0), 0.0);
    EXPECT_DOUBLE_EQ(fixed.occupation(0.001), 0.0);
    EXPECT_DOUBLE_EQ(fixed.occupation(100.0), 0.0);
}

TEST(FixedOccupation, AtFermiIsEmptyByConvention) {
    FixedOccupation fixed;
    // x = 0 means exactly at the Fermi level; convention: f(0) = 0
    EXPECT_DOUBLE_EQ(fixed.occupation(0.0), 0.0);
}

TEST(FixedOccupation, EntropyWeightIsAlwaysZero) {
    FixedOccupation fixed;
    EXPECT_DOUBLE_EQ(fixed.entropy_weight(0.0), 0.0);
    EXPECT_DOUBLE_EQ(fixed.entropy_weight(-5.0), 0.0);
    EXPECT_DOUBLE_EQ(fixed.entropy_weight(5.0), 0.0);
}

TEST(FixedOccupation, NameIsFixed) {
    FixedOccupation fixed;
    EXPECT_EQ(fixed.name(), "fixed");
}

TEST(FixedOccupation, SmearingWidthIsZero) {
    FixedOccupation fixed;
    EXPECT_DOUBLE_EQ(fixed.smearing_width(), 0.0);
}

TEST(FixedOccupation, UsesSmearingIsFalse) {
    FixedOccupation fixed;
    EXPECT_FALSE(fixed.uses_smearing());
}
