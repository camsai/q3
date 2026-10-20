#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace q3;

TEST(RadialUtilitiesTest, RadialSimpsonSC) {
    // Integrate f(r) = r^2 on a uniform grid
    int nr = 101;
    Real dr = 0.1;
    std::vector<Real> r(nr), rab(nr), f(nr);
    for (int i = 0; i < nr; i++) {
        r[i] = i * dr;
        rab[i] = dr; // uniform grid derivative
        f[i] = r[i] * r[i];
    }
    // Integral from 0 to 10 of r^2 dr is r^3/3 = 1000/3 = 333.3333
    Real integral = simpson_rule(f, rab, nr);
    EXPECT_NEAR(integral, 333.3333, 1e-3);
}

TEST(RadialUtilitiesTest, ComputeYlmAtPoint) {
    // Point along z-axis (0,0,1)
    Vector3D z_axis{0.0, 0.0, 1.0};

    // Y00 = 1/sqrt(4pi)
    EXPECT_NEAR(compute_real_spherical_harmonic(0, 0, z_axis), 1.0 / std::sqrt(4.0 * M_PI), 1e-7);
    // Y1,0 = sqrt(3/4pi) * cos(theta) = sqrt(3/4pi) at z-axis
    EXPECT_NEAR(compute_real_spherical_harmonic(1, 0, z_axis), std::sqrt(3.0 / (4.0 * M_PI)), 1e-7);
    // Y1,1 and Y1,-1 should be 0 since x=y=0
    EXPECT_NEAR(compute_real_spherical_harmonic(1, 1, z_axis), 0.0, 1e-7);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, -1, z_axis), 0.0, 1e-7);
}
