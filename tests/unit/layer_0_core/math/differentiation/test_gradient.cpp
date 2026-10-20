#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "layer_0_core/math/differentiation/gradient.hpp"

using namespace q3;

TEST(GradientNonuniform, LinearFunctionOnUniformGrid) {
    // f(x) = 2x + 1, f'(x) = 2
    const int mesh_size = 5;
    std::vector<Real> grid = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<Real> function_values(mesh_size);
    for (int index = 0; index < mesh_size; index++) {
        function_values[index] = 2.0 * grid[index] + 1.0;
    }
    std::vector<Real> gradient_values(mesh_size, 0.0);
    gradient_nonuniform(function_values.data(), gradient_values.data(), grid.data(), mesh_size);
    // Interior points should have gradient = 2.0
    for (int index = 1; index < mesh_size - 1; index++) {
        EXPECT_NEAR(gradient_values[index], 2.0, 1e-12);
    }
    // Last point is set to 0.0 by convention
    EXPECT_DOUBLE_EQ(gradient_values[mesh_size - 1], 0.0);
}

TEST(GradientNonuniform, QuadraticFunctionOnUniformGrid) {
    // f(x) = x², f'(x) = 2x
    const int mesh_size = 5;
    std::vector<Real> grid = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<Real> function_values(mesh_size);
    for (int index = 0; index < mesh_size; index++) {
        function_values[index] = grid[index] * grid[index];
    }
    std::vector<Real> gradient_values(mesh_size, 0.0);
    gradient_nonuniform(function_values.data(), gradient_values.data(), grid.data(), mesh_size);
    // Interior points: f'(1)=2, f'(2)=4, f'(3)=6
    EXPECT_NEAR(gradient_values[1], 2.0, 1e-12);
    EXPECT_NEAR(gradient_values[2], 4.0, 1e-12);
    EXPECT_NEAR(gradient_values[3], 6.0, 1e-12);
}

TEST(GradientNonuniform, NonUniformGrid) {
    // f(x) = 3x + 5, f'(x) = 3
    const int mesh_size = 4;
    std::vector<Real> grid = {0.0, 0.5, 2.0, 5.0};
    std::vector<Real> function_values(mesh_size);
    for (int index = 0; index < mesh_size; index++) {
        function_values[index] = 3.0 * grid[index] + 5.0;
    }
    std::vector<Real> gradient_values(mesh_size, 0.0);
    gradient_nonuniform(function_values.data(), gradient_values.data(), grid.data(), mesh_size);
    // Interior points should give gradient = 3.0
    for (int index = 1; index < mesh_size - 1; index++) {
        EXPECT_NEAR(gradient_values[index], 3.0, 1e-10);
    }
}

TEST(GradientNonuniform, LastPointIsZero) {
    const int mesh_size = 4;
    std::vector<Real> grid = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> function_values = {1.0, 4.0, 9.0, 16.0};
    std::vector<Real> gradient_values(mesh_size, 999.0);
    gradient_nonuniform(function_values.data(), gradient_values.data(), grid.data(), mesh_size);
    EXPECT_DOUBLE_EQ(gradient_values[mesh_size - 1], 0.0);
}

TEST(GradientNonuniform, ExtrapolatedFirstPoint) {
    /**
     * For f(x)=x², f'(x)=2x. On grid [0,1,2,3,4]:
     * gradient[1] = 2, gradient[2] = 4
     * Extrapolation: gradient[0] = gradient[1] +
     * (gradient[2]-gradient[1])*(grid[0]-grid[1])/(grid[2]-grid[1])
     *              = 2 + (4-2)*(0-1)/(2-1) = 2 + 2*(-1) = 0
     */
    const int mesh_size = 5;
    std::vector<Real> grid = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<Real> function_values = {0.0, 1.0, 4.0, 9.0, 16.0};
    std::vector<Real> gradient_values(mesh_size, 0.0);
    gradient_nonuniform(function_values.data(), gradient_values.data(), grid.data(), mesh_size);
    // f'(0) should be exactly 0 via extrapolation for x²
    EXPECT_NEAR(gradient_values[0], 0.0, 1e-12);
}

TEST(GradientNonuniform, SmallMeshSizeThreePoints) {
    // Minimum valid mesh for 3-point formula
    const int mesh_size = 3;
    std::vector<Real> grid = {0.0, 1.0, 2.0};
    std::vector<Real> function_values = {0.0, 1.0, 4.0}; // x²
    std::vector<Real> gradient_values(mesh_size, 0.0);
    gradient_nonuniform(function_values.data(), gradient_values.data(), grid.data(), mesh_size);
    EXPECT_NEAR(gradient_values[1], 2.0, 1e-12); // f'(1) = 2
    EXPECT_DOUBLE_EQ(gradient_values[mesh_size - 1], 0.0);
}
