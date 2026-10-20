#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "layer_0_core/math/spherical/bessel.hpp"

using namespace q3;

// ─── Single-point spherical_bessel tests ────────────────────────────────────

TEST(SphericalBessel, DegreeZeroAtZero) {
    // j_0(0) = 1
    EXPECT_DOUBLE_EQ(spherical_bessel(0, 0.0), 1.0);
}

TEST(SphericalBessel, NonZeroDegreeAtZero) {
    // j_l(0) = 0 for l > 0
    EXPECT_DOUBLE_EQ(spherical_bessel(1, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(spherical_bessel(2, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(spherical_bessel(5, 0.0), 0.0);
}

TEST(SphericalBessel, DegreeZeroAnalytic) {
    // j_0(x) = sin(x)/x
    double argument = 1.0;
    double expected = std::sin(argument) / argument;
    EXPECT_NEAR(spherical_bessel(0, argument), expected, 1e-14);

    argument = 3.14159;
    expected = std::sin(argument) / argument;
    EXPECT_NEAR(spherical_bessel(0, argument), expected, 1e-14);
}

TEST(SphericalBessel, DegreeOneAnalytic) {
    // j_1(x) = sin(x)/x² - cos(x)/x
    double argument = 2.0;
    double expected = (std::sin(argument) / argument - std::cos(argument)) / argument;
    EXPECT_NEAR(spherical_bessel(1, argument), expected, 1e-14);
}

TEST(SphericalBessel, DegreeTwoAnalytic) {
    // j_2(x) = (3/x² - 1)*sin(x)/x - 3*cos(x)/x²
    double argument = 2.5;
    double expected = ((3.0 / (argument * argument) - 1.0) * std::sin(argument) -
                       3.0 * std::cos(argument) / argument) /
                      argument;
    EXPECT_NEAR(spherical_bessel(2, argument), expected, 1e-14);
}

TEST(SphericalBessel, DegreeThreeByRecurrence) {
    // j_3(x) computed via recurrence: j_{l+1} = (2l-1)/x * j_l - j_{l-1}
    double argument = 3.0;
    double j0 = std::sin(argument) / argument;
    double j1 = (std::sin(argument) / argument - std::cos(argument)) / argument;
    double j2 = (3.0 * j1 / argument) - j0;
    double j3 = (5.0 * j2 / argument) - j1;
    EXPECT_NEAR(spherical_bessel(3, argument), j3, 1e-13);
}

TEST(SphericalBessel, SmallArgumentSeriesExpansion) {
    // For very small x, j_0(x) ≈ 1 - x²/6
    double argument = 0.01;
    double expected = std::sin(argument) / argument;
    EXPECT_NEAR(spherical_bessel(0, argument), expected, 1e-15);
}

TEST(SphericalBessel, SmallArgumentDegreeOne) {
    // For small x, j_1(x) ≈ x/3
    double argument = 0.001;
    double result = spherical_bessel(1, argument);
    double approximate = argument / 3.0;
    EXPECT_NEAR(result, approximate, 1e-10);
}

TEST(SphericalBessel, NegativeArgument) {
    // j_0(-x) = sin(-x)/(-x) = sin(x)/x = j_0(x)
    double argument = 2.0;
    EXPECT_NEAR(spherical_bessel(0, -argument), spherical_bessel(0, argument), 1e-14);
    // j_1(-x) = -j_1(x) (odd function for odd l)
    EXPECT_NEAR(spherical_bessel(1, -argument), -spherical_bessel(1, argument), 1e-14);
}

TEST(SphericalBessel, LargeArgument) {
    // j_0(100) = sin(100)/100
    double argument = 100.0;
    double expected = std::sin(argument) / argument;
    EXPECT_NEAR(spherical_bessel(0, argument), expected, 1e-14);
}

// ─── Batch spherical_bessel_batch tests ─────────────────────────────────────

TEST(SphericalBesselBatch, DegreeZeroMatchesSingle) {
    const int mesh_size = 5;
    std::vector<Real> radial_grid = {0.5, 1.0, 2.0, 3.0, 5.0};
    Real q_value = 1.5;
    std::vector<Real> result(mesh_size);
    spherical_bessel_batch(0, q_value, radial_grid.data(), mesh_size, result.data());
    for (int index = 0; index < mesh_size; index++) {
        Real expected = spherical_bessel(0, q_value * radial_grid[index]);
        EXPECT_NEAR(result[index], expected, 1e-14);
    }
}

TEST(SphericalBesselBatch, DegreeOneMatchesSingle) {
    const int mesh_size = 4;
    std::vector<Real> radial_grid = {1.0, 2.0, 3.0, 4.0};
    Real q_value = 2.0;
    std::vector<Real> result(mesh_size);
    spherical_bessel_batch(1, q_value, radial_grid.data(), mesh_size, result.data());
    for (int index = 0; index < mesh_size; index++) {
        Real expected = spherical_bessel(1, q_value * radial_grid[index]);
        EXPECT_NEAR(result[index], expected, 1e-14);
    }
}

TEST(SphericalBesselBatch, DegreeTwoMatchesSingle) {
    const int mesh_size = 4;
    std::vector<Real> radial_grid = {1.0, 2.0, 3.0, 4.0};
    Real q_value = 1.0;
    std::vector<Real> result(mesh_size);
    spherical_bessel_batch(2, q_value, radial_grid.data(), mesh_size, result.data());
    for (int index = 0; index < mesh_size; index++) {
        Real expected = spherical_bessel(2, q_value * radial_grid[index]);
        EXPECT_NEAR(result[index], expected, 1e-14);
    }
}

TEST(SphericalBesselBatch, HigherDegreeMatchesSingle) {
    const int mesh_size = 3;
    std::vector<Real> radial_grid = {1.0, 3.0, 5.0};
    Real q_value = 1.0;
    std::vector<Real> result(mesh_size);
    spherical_bessel_batch(4, q_value, radial_grid.data(), mesh_size, result.data());
    for (int index = 0; index < mesh_size; index++) {
        Real expected = spherical_bessel(4, q_value * radial_grid[index]);
        EXPECT_NEAR(result[index], expected, 1e-13);
    }
}

TEST(SphericalBesselBatch, ZeroQValue) {
    const int mesh_size = 3;
    std::vector<Real> radial_grid = {1.0, 2.0, 3.0};
    std::vector<Real> result(mesh_size);
    // q=0, l=0: all j_0(0) = 1
    spherical_bessel_batch(0, 0.0, radial_grid.data(), mesh_size, result.data());
    for (int index = 0; index < mesh_size; index++) {
        EXPECT_DOUBLE_EQ(result[index], 1.0);
    }
    // q=0, l=2: all j_2(0) = 0
    spherical_bessel_batch(2, 0.0, radial_grid.data(), mesh_size, result.data());
    for (int index = 0; index < mesh_size; index++) {
        EXPECT_DOUBLE_EQ(result[index], 0.0);
    }
}

TEST(SphericalBesselBatch, SmallQTimesRUsesSeriesExpansion) {
    // With q*r small enough, the batch function should use the series expansion
    const int mesh_size = 1;
    std::vector<Real> radial_grid = {0.001};
    Real q_value = 0.01;
    std::vector<Real> result(mesh_size);
    spherical_bessel_batch(0, q_value, radial_grid.data(), mesh_size, result.data());
    // j_0(0.00001) ≈ 1
    EXPECT_NEAR(result[0], 1.0, 1e-8);
}

// ─── Accuracy across the branch boundary ────────────────────────────────────

/**
 * Values of j_l(x) computed to 40 digits from the ascending series, spanning
 * both sides of the x = l+1 branch split and, deliberately, the region where
 * the upward recurrence used to be catastrophically wrong: at l = 5, x = 0.06
 * it returned −4.99e−08 for a quantity that is positive and of order 7e−11.
 *
 * Every entry is a place a hand-rolled evaluation can go wrong quietly. A
 * relative bound is what makes them useful — an absolute one is trivially met
 * by anything at all where j_l is small, which is exactly where the failures
 * were.
 */
struct BesselReference {
    int degree;
    Real argument;
    Real value;
};

TEST(SphericalBessel, MatchesHighPrecisionValuesOnBothSidesOfTheBranch) {
    const std::vector<BesselReference> reference = {
        {0, 0.06, 9.9940010799074330e-01},    {0, 3.1, 1.3413116913964703e-02},
        {1, 0.06, 1.9992800925652573e-02},    {1, 0.9, 2.7639251627640171e-01},
        {1, 4.48, 2.9216254300089167e-03},    {2, 0.06, 2.3993829188537766e-04},
        {2, 1.0, 6.2035052011373860e-02},     {2, 5.1, 1.1699826142925811e-01},
        {3, 0.06, 2.0567314622322125e-06},    {3, 0.3, 2.5585976969508186e-04},
        {3, 10.377, -3.7554557299644712e-03}, {4, 0.06, 1.3712041713799979e-08},
        {4, 0.1, 1.0577201502098732e-07},     {4, 2.0, 1.4079392762915321e-02},
        {5, 0.06, 7.4794837784273773e-11},    {5, 0.1, 9.6163102329164459e-10},
        {5, 0.2, 3.0736702297657188e-08},     {5, 2.0, 2.6351697702441173e-03},
        {6, 0.5, 1.1466510767409421e-07},     {6, 3.0, 3.9743825098192470e-03},
    };

    for (const auto& entry : reference) {
        EXPECT_NEAR(spherical_bessel(entry.degree, entry.argument),
                    entry.value,
                    1e-12 * std::abs(entry.value))
            << "l = " << entry.degree << " at x = " << entry.argument;
    }
}

TEST(SphericalBesselBatch, AgreesWithTheScalarRoutineIncludingAcrossBranches) {
    /**
     * The batch routine keeps its own copy of the branch logic for
     * vectorization, so the two can disagree; on a logarithmic grid every
     * q sweeps the whole small-to-large range of q·r, which is what makes this
     * a real cross-check rather than a spot check.
     */
    std::vector<Real> radial_grid;
    for (int index = 0; index < 300; index++)
        radial_grid.push_back(1e-5 * std::exp(0.04 * index));
    std::vector<Real> result(radial_grid.size());

    for (int degree = 0; degree <= 6; degree++) {
        for (Real q_value : {0.05, 0.5, 1.0, 3.0, 10.0}) {
            spherical_bessel_batch(
                degree, q_value, radial_grid.data(), radial_grid.size(), result.data());
            for (size_t index = 0; index < radial_grid.size(); index++) {
                Real expected = spherical_bessel(degree, q_value * radial_grid[index]);
                EXPECT_NEAR(result[index], expected, 1e-13 * std::abs(expected))
                    << "l = " << degree << " at x = " << q_value * radial_grid[index];
            }
        }
    }
}
