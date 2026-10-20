#include <gtest/gtest.h>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/math/interpolation/uniform_q_grid_table.hpp"

using namespace q3;

TEST(UniformQGrid, FromMaximumQ) {
    UniformQGrid grid = UniformQGrid::from_maximum_q(10.0, 0.01);
    EXPECT_GT(grid.number_of_q_points, 1000);
    EXPECT_DOUBLE_EQ(grid.q_spacing, 0.01);
    EXPECT_DOUBLE_EQ(grid.q_at_index(0), 0.0);
    EXPECT_NEAR(grid.q_at_index(100), 1.0, 1e-12);
}

TEST(UniformQGridTable, InterpolateLinearFunction) {
    UniformQGrid grid = UniformQGrid::from_maximum_q(1.0, 0.1);
    UniformQGridTable table;
    table.resize(grid);
    for (int index = 0; index < table.size(); index++) {
        table[index] = grid.q_at_index(index);
    }
    EXPECT_NEAR(table.interpolate(0.55), 0.55, 0.01);
    EXPECT_NEAR(table.interpolate_derivative(0.55), 1.0, 0.1);
}

TEST(UniformQGridTable, InterpolateAtZero) {
    UniformQGrid grid = UniformQGrid::from_maximum_q(1.0, 0.1);
    UniformQGridTable table;
    table.resize(grid);
    table[0] = 42.0;
    EXPECT_DOUBLE_EQ(table.interpolate(0.0), 42.0);
}
