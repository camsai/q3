#include "layer_0_core/math/spherical/bessel.hpp"
#include "layer_0_core/math/transform/spherical_bessel.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace q3;

// ─── spherical_bessel (single-point) ────────────────────────────────────────

TEST(SphericalBessel, OrderZeroAtZeroIsOne) {
    // j_0(0) = 1
    EXPECT_NEAR(spherical_bessel(0, 0.0), 1.0, 1e-14);
}

TEST(SphericalBessel, OrderZeroSinXOverX) {
    // j_0(x) = sin(x)/x for x > 0
    Real argument = 2.5;
    Real expected = std::sin(argument) / argument;
    EXPECT_NEAR(spherical_bessel(0, argument), expected, 1e-12);
}

TEST(SphericalBessel, OrderOneAtZeroIsZero) {
    // j_1(0) = 0
    EXPECT_NEAR(spherical_bessel(1, 0.0), 0.0, 1e-14);
}

TEST(SphericalBessel, OrderOneAnalyticFormula) {
    // j_1(x) = sin(x)/x^2 - cos(x)/x
    Real argument = 1.5;
    Real expected = std::sin(argument) / (argument * argument) - std::cos(argument) / argument;
    EXPECT_NEAR(spherical_bessel(1, argument), expected, 1e-12);
}

TEST(SphericalBessel, OrderZeroAtPiIsNearZero) {
    // j_0(π) = sin(π)/π ≈ 0
    EXPECT_NEAR(spherical_bessel(0, constants::pi), 0.0, 1e-14);
}

TEST(SphericalBessel, OrderTwoAnalyticFormula) {
    // j_2(x) = (3/x^2 - 1)*sin(x)/x - 3*cos(x)/x^2
    Real argument = 3.0;
    Real expected = ((3.0 / (argument * argument) - 1.0) * std::sin(argument) -
                     3.0 * std::cos(argument) / argument) /
                    argument;
    EXPECT_NEAR(spherical_bessel(2, argument), expected, 1e-12);
}

TEST(SphericalBessel, HighOrderViaRecurrence) {
    /**
     * j_3(x) via recurrence should match the analytic formula:
     * j_3(x) = (15/x^3 - 6/x)*sin(x)/x - (15/x^2 - 1)*cos(x)/x
     */
    Real argument = 2.0;
    Real sin_x = std::sin(argument);
    Real cos_x = std::cos(argument);
    Real expected = ((15.0 / (argument * argument * argument) - 6.0 / argument) * sin_x -
                     (15.0 / (argument * argument) - 1.0) * cos_x) /
                    argument;
    EXPECT_NEAR(spherical_bessel(3, argument), expected, 1e-10);
}

// ─── spherical_bessel_batch ─────────────────────────────────────────────────

TEST(SphericalBesselBatch, OrderZeroMatchesSinglePoint) {
    std::vector<Real> radial_grid = {0.5, 1.0, 1.5, 2.0, 2.5};
    int mesh_size = static_cast<int>(radial_grid.size());
    std::vector<Real> result(mesh_size);
    Real q_value = 1.0;

    spherical_bessel_batch(0, q_value, radial_grid.data(), mesh_size, result.data());

    for (int index = 0; index < mesh_size; index++) {
        Real argument = q_value * radial_grid[index];
        Real expected = std::sin(argument) / argument;
        EXPECT_NEAR(result[index], expected, 1e-12);
    }
}

TEST(SphericalBesselBatch, OrderOneMatchesSinglePoint) {
    std::vector<Real> radial_grid = {0.5, 1.0, 2.0, 3.0};
    int mesh_size = static_cast<int>(radial_grid.size());
    std::vector<Real> result(mesh_size);
    Real q_value = 2.0;

    spherical_bessel_batch(1, q_value, radial_grid.data(), mesh_size, result.data());

    for (int index = 0; index < mesh_size; index++) {
        Real argument = q_value * radial_grid[index];
        Real expected = (std::sin(argument) / argument - std::cos(argument)) / argument;
        EXPECT_NEAR(result[index], expected, 1e-12);
    }
}

TEST(SphericalBesselBatch, ZeroQValueOrderZeroGivesAllOnes) {
    std::vector<Real> radial_grid = {0.1, 0.5, 1.0, 2.0};
    int mesh_size = static_cast<int>(radial_grid.size());
    std::vector<Real> result(mesh_size);

    spherical_bessel_batch(0, 0.0, radial_grid.data(), mesh_size, result.data());

    for (int index = 0; index < mesh_size; index++) {
        EXPECT_DOUBLE_EQ(result[index], 1.0);
    }
}

TEST(SphericalBesselBatch, ZeroQValueOrderOneGivesAllZeros) {
    std::vector<Real> radial_grid = {0.1, 0.5, 1.0, 2.0};
    int mesh_size = static_cast<int>(radial_grid.size());
    std::vector<Real> result(mesh_size);

    spherical_bessel_batch(1, 0.0, radial_grid.data(), mesh_size, result.data());

    for (int index = 0; index < mesh_size; index++) {
        EXPECT_DOUBLE_EQ(result[index], 0.0);
    }
}

// ─── transform_spherical_bessel ─────────────────────────────────────────────

TEST(SphericalBesselTransform, OutputLengthMatchesQPoints) {
    int mesh_size = 101;
    Real grid_spacing = 0.1;
    std::vector<Real> grid(mesh_size);
    std::vector<Real> integration_factors(mesh_size, grid_spacing);
    std::vector<Real> function_values(mesh_size, 0.0);
    std::vector<Real> q_points = {0.0, 0.5, 1.0, 1.5, 2.0};

    std::vector<Real> fourier_transform;
    transform_spherical_bessel(
        grid, integration_factors, function_values, 0, mesh_size, q_points, fourier_transform);

    EXPECT_EQ(fourier_transform.size(), q_points.size());
}

TEST(SphericalBesselTransform, ZeroFunctionProducesZeroTransform) {
    int mesh_size = 101;
    Real grid_spacing = 0.1;
    std::vector<Real> grid(mesh_size);
    std::vector<Real> integration_factors(mesh_size, grid_spacing);
    std::vector<Real> function_values(mesh_size, 0.0);
    std::vector<Real> q_points = {0.0, 0.5, 1.0, 2.0};

    for (int index = 0; index < mesh_size; index++) {
        grid[index] = index * grid_spacing;
    }

    std::vector<Real> fourier_transform;
    transform_spherical_bessel(
        grid, integration_factors, function_values, 0, mesh_size, q_points, fourier_transform);

    for (size_t index = 0; index < fourier_transform.size(); index++) {
        EXPECT_NEAR(fourier_transform[index], 0.0, 1e-14);
    }
}

TEST(SphericalBesselTransform, GaussianProducesSmoothResult) {
    // Transform of a Gaussian f(r) = exp(-r^2) should produce a smooth q-space result
    int mesh_size = 501;
    Real grid_spacing = 0.02;
    std::vector<Real> grid(mesh_size);
    std::vector<Real> integration_factors(mesh_size, grid_spacing);
    std::vector<Real> function_values(mesh_size);

    for (int index = 0; index < mesh_size; index++) {
        grid[index] = index * grid_spacing;
        function_values[index] = std::exp(-grid[index] * grid[index]);
    }

    std::vector<Real> q_points = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0};
    std::vector<Real> fourier_transform;

    transform_spherical_bessel(
        grid, integration_factors, function_values, 0, mesh_size, q_points, fourier_transform);

    ASSERT_EQ(fourier_transform.size(), q_points.size());

    // The transform of a Gaussian should be finite and decrease monotonically for l=0
    for (size_t index = 0; index < fourier_transform.size(); index++) {
        EXPECT_TRUE(std::isfinite(fourier_transform[index]));
    }

    // For l=0 transform of exp(-r^2), result should be positive and decreasing
    for (size_t index = 1; index < fourier_transform.size(); index++) {
        EXPECT_GT(fourier_transform[index], 0.0);
        EXPECT_LT(fourier_transform[index], fourier_transform[index - 1]);
    }
}
