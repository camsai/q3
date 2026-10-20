#include "layer_0_core/math/interpolation/lagrange.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace q3;

// ─── lagrange_interpolation_4point ──────────────────────────────────────────

TEST(LagrangeInterpolation, LinearFunctionAtGridPoints) {
    // f(q) = 2*q + 1, tabulated at q = 0, 1, 2, 3, 4, 5
    Real q_spacing = 1.0;
    std::vector<Real> table = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0};
    int table_size = static_cast<int>(table.size());

    // Interpolation at grid points should be exact
    EXPECT_NEAR(lagrange_interpolation_4point(table, 0.0, q_spacing, table_size), 1.0, 1e-12);
    EXPECT_NEAR(lagrange_interpolation_4point(table, 1.0, q_spacing, table_size), 3.0, 1e-12);
    EXPECT_NEAR(lagrange_interpolation_4point(table, 2.0, q_spacing, table_size), 5.0, 1e-12);
}

TEST(LagrangeInterpolation, LinearFunctionBetweenGridPoints) {
    // f(q) = 2*q + 1 — Lagrange interpolation is exact for polynomials up to degree 3
    Real q_spacing = 1.0;
    std::vector<Real> table = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0};
    int table_size = static_cast<int>(table.size());

    // Midpoints
    EXPECT_NEAR(lagrange_interpolation_4point(table, 0.5, q_spacing, table_size), 2.0, 1e-12);
    EXPECT_NEAR(lagrange_interpolation_4point(table, 1.5, q_spacing, table_size), 4.0, 1e-12);
    EXPECT_NEAR(lagrange_interpolation_4point(table, 2.3, q_spacing, table_size), 5.6, 1e-12);
}

TEST(LagrangeInterpolation, QuadraticFunctionAtMidpoints) {
    // f(q) = q^2, tabulated at q = 0, 1, 2, 3, 4, 5
    Real q_spacing = 1.0;
    std::vector<Real> table = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0};
    int table_size = static_cast<int>(table.size());

    // 4-point Lagrange interpolation is exact for polynomials up to degree 3
    EXPECT_NEAR(lagrange_interpolation_4point(table, 0.5, q_spacing, table_size), 0.25, 1e-12);
    EXPECT_NEAR(lagrange_interpolation_4point(table, 1.5, q_spacing, table_size), 2.25, 1e-12);
    EXPECT_NEAR(lagrange_interpolation_4point(table, 2.5, q_spacing, table_size), 6.25, 1e-12);
}

TEST(LagrangeInterpolation, CubicFunctionExact) {
    // f(q) = q^3, tabulated at q = 0, 1, 2, 3, 4, 5
    Real q_spacing = 1.0;
    std::vector<Real> table = {0.0, 1.0, 8.0, 27.0, 64.0, 125.0};
    int table_size = static_cast<int>(table.size());

    // 4-point Lagrange interpolation is exact for cubics
    EXPECT_NEAR(lagrange_interpolation_4point(table, 0.5, q_spacing, table_size), 0.125, 1e-12);
    EXPECT_NEAR(lagrange_interpolation_4point(table, 1.5, q_spacing, table_size), 3.375, 1e-12);
    EXPECT_NEAR(lagrange_interpolation_4point(table, 2.7, q_spacing, table_size), 19.683, 1e-10);
}

TEST(LagrangeInterpolation, OutOfRangeReturnsZero) {
    Real q_spacing = 1.0;
    std::vector<Real> table = {0.0, 1.0, 4.0, 9.0};
    int table_size = static_cast<int>(table.size());

    // q_value = 1.0 → base_index = 1, needs base_index+3 = 4 < table_size=4 fails → returns 0
    EXPECT_DOUBLE_EQ(lagrange_interpolation_4point(table, 1.0, q_spacing, table_size), 0.0);
    // q_value = 5.0 → definitely out of range
    EXPECT_DOUBLE_EQ(lagrange_interpolation_4point(table, 5.0, q_spacing, table_size), 0.0);
}

TEST(LagrangeInterpolation, KnownValueQuadraticAtHalfPoint) {
    /**
     * table = {0, 1, 4, 9} representing q^2 at q=0,1,2,3 with spacing=1.0
     * Interpolate at q=1.5, expected result = 1.5^2 = 2.25
     */
    Real q_spacing = 1.0;
    std::vector<Real> table = {0.0, 1.0, 4.0, 9.0, 16.0};
    int table_size = static_cast<int>(table.size());

    EXPECT_NEAR(lagrange_interpolation_4point(table, 1.5, q_spacing, table_size), 2.25, 1e-12);
}

// ─── lagrange_interpolation_4point_derivative ───────────────────────────────

TEST(LagrangeInterpolationDerivative, LinearFunctionDerivativeIsConstant) {
    // f(q) = 3*q, tabulated at q = 0, 1, 2, 3, 4, 5
    Real q_spacing = 1.0;
    std::vector<Real> table = {0.0, 3.0, 6.0, 9.0, 12.0, 15.0};
    int table_size = static_cast<int>(table.size());

    // Derivative of 3*q is 3.0
    EXPECT_NEAR(
        lagrange_interpolation_4point_derivative(table, 0.5, q_spacing, table_size), 3.0, 1e-10);
    EXPECT_NEAR(
        lagrange_interpolation_4point_derivative(table, 1.0, q_spacing, table_size), 3.0, 1e-10);
    EXPECT_NEAR(
        lagrange_interpolation_4point_derivative(table, 1.5, q_spacing, table_size), 3.0, 1e-10);
}

TEST(LagrangeInterpolationDerivative, QuadraticFunctionDerivativeAtPoint) {
    // f(q) = q^2, derivative = 2*q
    Real q_spacing = 1.0;
    std::vector<Real> table = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0};
    int table_size = static_cast<int>(table.size());

    // At q=2.0, derivative should be 2*2.0 = 4.0
    EXPECT_NEAR(
        lagrange_interpolation_4point_derivative(table, 2.0, q_spacing, table_size), 4.0, 1e-10);
    // At q=1.5, derivative should be 2*1.5 = 3.0
    EXPECT_NEAR(
        lagrange_interpolation_4point_derivative(table, 1.5, q_spacing, table_size), 3.0, 1e-10);
}

TEST(LagrangeInterpolationDerivative, OutOfRangeReturnsZero) {
    Real q_spacing = 1.0;
    std::vector<Real> table = {0.0, 1.0, 4.0, 9.0};
    int table_size = static_cast<int>(table.size());

    EXPECT_DOUBLE_EQ(lagrange_interpolation_4point_derivative(table, 5.0, q_spacing, table_size),
                     0.0);
}
