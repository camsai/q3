#include <gtest/gtest.h>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/core_charge_density_table.hpp"

using namespace q3;

/**
 * Default-constructed CoreChargeDensityTable should have a default q_grid
 * with zero points and zero spacing.
 */
TEST(CoreChargeDensityTable, DefaultConstruction) {
    CoreChargeDensityTable table;
    const UniformQGrid& grid = table.q_grid();
    EXPECT_EQ(grid.number_of_q_points, 0);
    EXPECT_DOUBLE_EQ(grid.q_spacing, 0.0);
}

/**
 * Test initialization with a PseudopotentialSet where the pseudopotential
 * does NOT have nonlinear core correction. The table should report
 * no core charge and interpolation should return zero.
 */
TEST(CoreChargeDensityTable, InitializeWithoutNLCC) {
    // Build a minimal pseudopotential with no NLCC
    auto pseudopotential = std::make_unique<Pseudopotential>();
    pseudopotential->element = "Si";
    pseudopotential->has_nonlinear_core_correction = false;
    pseudopotential->mesh_size = 100;
    pseudopotential->radial_grid.resize(100, 0.01);
    pseudopotential->radial_grid_derivative.resize(100, 0.01);
    pseudopotential->number_of_projectors = 0;

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(std::move(pseudopotential));

    UniformQGrid grid = UniformQGrid::from_maximum_q(10.0, 0.1);
    std::vector<int> mesh_sizes = {99};

    CoreChargeDensityTable table;
    table.initialize(grid, pseudopotential_set, mesh_sizes, 1);

    EXPECT_FALSE(table.has_core_charge_for_species(0));
    EXPECT_DOUBLE_EQ(table.interpolate(0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(table.interpolate_derivative(0, 1.0), 0.0);
}

/**
 * Verify that the q_grid is correctly stored after initialization.
 */
TEST(CoreChargeDensityTable, QGridStoredAfterInitialize) {
    auto pseudopotential = std::make_unique<Pseudopotential>();
    pseudopotential->element = "He";
    pseudopotential->has_nonlinear_core_correction = false;
    pseudopotential->mesh_size = 50;
    pseudopotential->radial_grid.resize(50, 0.01);
    pseudopotential->radial_grid_derivative.resize(50, 0.01);
    pseudopotential->number_of_projectors = 0;

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(std::move(pseudopotential));

    UniformQGrid grid = UniformQGrid::from_maximum_q(5.0, 0.05);
    std::vector<int> mesh_sizes = {49};

    CoreChargeDensityTable table;
    table.initialize(grid, pseudopotential_set, mesh_sizes, 1);

    EXPECT_EQ(table.q_grid().number_of_q_points, grid.number_of_q_points);
    EXPECT_DOUBLE_EQ(table.q_grid().q_spacing, grid.q_spacing);
}
