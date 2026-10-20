#include <gtest/gtest.h>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/math/interpolation/uniform_q_grid_table.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/init.hpp"

using namespace q3;

/**
 * Verify that a default-constructed BetaFunctionTable has zero q-points
 * and zero spacing.
 */
TEST(BetaFunctionTable, DefaultConstruction) {
    BetaFunctionTable table;
    EXPECT_EQ(table.number_of_q_points(), 0);
    EXPECT_DOUBLE_EQ(table.q_spacing(), 0.0);
}
