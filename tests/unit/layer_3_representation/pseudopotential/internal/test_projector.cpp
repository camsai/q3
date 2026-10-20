#include <gtest/gtest.h>

#include <vector>

#include "layer_3_representation/pseudopotential/internal/projector.hpp"

using namespace q3;

// ── Default construction gives zero angular momentum and cutoff ──
TEST(Projector, DefaultConstructionGivesZeroFields) {
    Projector projector{};

    EXPECT_EQ(projector.angular_momentum, 0);
    EXPECT_EQ(projector.number_of_radial_cutoff_points, 0);
    EXPECT_TRUE(projector.radial_values.empty());
}
