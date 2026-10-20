#include <gtest/gtest.h>

#include "layer_1_structure/reciprocal_space/kpoint.hpp"

using namespace q3;

TEST(KPoint, DefaultConstruction) {
    KPoint kpoint{};
    EXPECT_DOUBLE_EQ(kpoint.coordinates.x, 0.0);
    EXPECT_DOUBLE_EQ(kpoint.coordinates.y, 0.0);
    EXPECT_DOUBLE_EQ(kpoint.coordinates.z, 0.0);
    EXPECT_DOUBLE_EQ(kpoint.weight, 1.0);
}
