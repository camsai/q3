#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"

using namespace q3;

// ── Default construction ────────────────────────────────────────────

TEST(UniformQGrid, DefaultConstructionHasZeroSpacing) {
    UniformQGrid grid;
    EXPECT_DOUBLE_EQ(grid.q_spacing, 0.0);
}

TEST(UniformQGrid, DefaultConstructionHasZeroPoints) {
    UniformQGrid grid;
    EXPECT_EQ(grid.number_of_q_points, 0);
}

// ── Factory construction ────────────────────────────────────────────

TEST(UniformQGrid, FromMaximumQSetsSpacing) {
    auto grid = UniformQGrid::from_maximum_q(10.0, 0.01);
    EXPECT_DOUBLE_EQ(grid.q_spacing, 0.01);
}

TEST(UniformQGrid, FromMaximumQComputesPointCount) {
    // number_of_q_points = int(maximum_q_value / q_spacing) + 4
    auto grid = UniformQGrid::from_maximum_q(10.0, 0.01);
    int expected_points = static_cast<int>(10.0 / 0.01) + 4;
    EXPECT_EQ(grid.number_of_q_points, expected_points);
}

TEST(UniformQGrid, FromMaximumQSmallSpacing) {
    auto grid = UniformQGrid::from_maximum_q(1.0, 0.1);
    int expected_points = static_cast<int>(1.0 / 0.1) + 4;
    EXPECT_EQ(grid.number_of_q_points, expected_points);
}

TEST(UniformQGrid, FromMaximumQLargeSpacing) {
    auto grid = UniformQGrid::from_maximum_q(5.0, 2.5);
    int expected_points = static_cast<int>(5.0 / 2.5) + 4;
    EXPECT_EQ(grid.number_of_q_points, expected_points);
}

// ── q_at_index ──────────────────────────────────────────────────────

TEST(UniformQGrid, QAtIndexZeroIsZero) {
    auto grid = UniformQGrid::from_maximum_q(10.0, 0.5);
    EXPECT_DOUBLE_EQ(grid.q_at_index(0), 0.0);
}

TEST(UniformQGrid, QAtIndexReturnsMultipleOfSpacing) {
    auto grid = UniformQGrid::from_maximum_q(10.0, 0.25);
    EXPECT_DOUBLE_EQ(grid.q_at_index(1), 0.25);
    EXPECT_DOUBLE_EQ(grid.q_at_index(2), 0.50);
    EXPECT_DOUBLE_EQ(grid.q_at_index(4), 1.00);
    EXPECT_DOUBLE_EQ(grid.q_at_index(10), 2.50);
}

// ── q_point_values ──────────────────────────────────────────────────

TEST(UniformQGrid, QPointValuesHasCorrectSize) {
    auto grid = UniformQGrid::from_maximum_q(1.0, 0.5);
    auto values = grid.q_point_values();
    EXPECT_EQ(static_cast<int>(values.size()), grid.number_of_q_points);
}

TEST(UniformQGrid, QPointValuesStartsAtZero) {
    auto grid = UniformQGrid::from_maximum_q(5.0, 1.0);
    auto values = grid.q_point_values();
    ASSERT_FALSE(values.empty());
    EXPECT_DOUBLE_EQ(values[0], 0.0);
}

TEST(UniformQGrid, QPointValuesAreUniformlySpaced) {
    auto grid = UniformQGrid::from_maximum_q(2.0, 0.25);
    auto values = grid.q_point_values();
    for (int index = 1; index < static_cast<int>(values.size()); index++) {
        EXPECT_NEAR(values[index] - values[index - 1], 0.25, 1e-15);
    }
}

TEST(UniformQGrid, QPointValuesConsistentWithQAtIndex) {
    auto grid = UniformQGrid::from_maximum_q(3.0, 0.1);
    auto values = grid.q_point_values();
    for (int index = 0; index < grid.number_of_q_points; index++) {
        EXPECT_DOUBLE_EQ(values[index], grid.q_at_index(index));
    }
}

TEST(UniformQGrid, DefaultGridQPointValuesIsEmpty) {
    UniformQGrid grid;
    auto values = grid.q_point_values();
    EXPECT_TRUE(values.empty());
}
