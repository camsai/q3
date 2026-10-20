#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_4_computation/iterator/internal/smearing/gaussian.hpp"

using namespace q3;

TEST(GaussianSmearing, OccupationAtZeroIsOneHalf) {
    // f(0) = erfc(0)/2 = 1/2
    GaussianSmearing gaussian(0.01);
    EXPECT_DOUBLE_EQ(gaussian.occupation(0.0), 0.5);
}

TEST(GaussianSmearing, OccupationLargeNegativeApproachesOne) {
    // For x << 0: erfc(x)/2 → 1
    GaussianSmearing gaussian(0.01);
    EXPECT_NEAR(gaussian.occupation(-10.0), 1.0, 1e-12);
}

TEST(GaussianSmearing, OccupationLargePositiveApproachesZero) {
    // For x >> 0: erfc(x)/2 → 0
    GaussianSmearing gaussian(0.01);
    EXPECT_NEAR(gaussian.occupation(10.0), 0.0, 1e-12);
}

TEST(GaussianSmearing, OccupationIsMonotonicallyDecreasing) {
    GaussianSmearing gaussian(0.01);
    const std::vector<Real> test_points = {-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0};
    for (size_t index = 1; index < test_points.size(); ++index) {
        EXPECT_GT(gaussian.occupation(test_points[index - 1]),
                  gaussian.occupation(test_points[index]))
            << "Monotonicity violated between x=" << test_points[index - 1]
            << " and x=" << test_points[index];
    }
}

TEST(GaussianSmearing, EntropyWeightAtZero) {
    // entropy_weight(0) = -1 / (2√π) * exp(0) = -1 / (2√π)
    GaussianSmearing gaussian(0.01);
    const Real expected = -0.5 / std::sqrt(constants::pi);
    EXPECT_NEAR(gaussian.entropy_weight(0.0), expected, 1e-15);
}

TEST(GaussianSmearing, EntropyWeightIsAlwaysNonPositive) {
    GaussianSmearing gaussian(0.01);
    const std::vector<Real> test_points = {-5.0, -1.0, 0.0, 1.0, 5.0};
    for (const auto& x : test_points) {
        EXPECT_LE(gaussian.entropy_weight(x), 0.0) << "Entropy weight must be <= 0 at x=" << x;
    }
}

TEST(GaussianSmearing, NameIsGaussian) {
    GaussianSmearing gaussian(0.01);
    EXPECT_EQ(gaussian.name(), "gaussian");
}

TEST(GaussianSmearing, SmearingWidthReturnsConstructorValue) {
    const Real width = 0.05;
    GaussianSmearing gaussian(width);
    EXPECT_DOUBLE_EQ(gaussian.smearing_width(), width);
}

TEST(GaussianSmearing, UsesSmearingIsTrueWhenWidthPositive) {
    GaussianSmearing gaussian(0.01);
    EXPECT_TRUE(gaussian.uses_smearing());
}

TEST(GaussianSmearing, UsesSmearingIsFalseWhenWidthZero) {
    GaussianSmearing gaussian(0.0);
    EXPECT_FALSE(gaussian.uses_smearing());
}
