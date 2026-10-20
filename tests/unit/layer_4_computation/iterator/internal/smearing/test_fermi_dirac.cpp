#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_4_computation/iterator/internal/smearing/fermi_dirac.hpp"

using namespace q3;

TEST(FermiDiracSmearing, OccupationAtZeroIsOneHalf) {
    // f(0) = 1 / (exp(0) + 1) = 1/2
    FermiDiracSmearing fd(0.01);
    EXPECT_DOUBLE_EQ(fd.occupation(0.0), 0.5);
}

TEST(FermiDiracSmearing, OccupationLargeNegativeApproachesOne) {
    FermiDiracSmearing fd(0.01);
    EXPECT_NEAR(fd.occupation(-40.0), 1.0, 1e-12);
}

TEST(FermiDiracSmearing, OccupationLargePositiveApproachesZero) {
    FermiDiracSmearing fd(0.01);
    EXPECT_NEAR(fd.occupation(40.0), 0.0, 1e-12);
}

TEST(FermiDiracSmearing, OccupationStaysWithinUnitInterval) {
    /**
     * Unlike Methfessel-Paxton, the true Fermi-Dirac distribution never
     * over/undershoots [0, 1] — this is the whole point of using it.
     */
    FermiDiracSmearing fd(0.01);
    for (Real x = -50.0; x <= 50.0; x += 0.1) {
        Real f = fd.occupation(x);
        EXPECT_GE(f, 0.0);
        EXPECT_LE(f, 1.0);
    }
}

TEST(FermiDiracSmearing, OccupationDoesNotOverflowAtExtremeX) {
    FermiDiracSmearing fd(0.01);
    EXPECT_EQ(fd.occupation(1000.0), 0.0);
    EXPECT_EQ(fd.occupation(-1000.0), 1.0);
}

TEST(FermiDiracSmearing, OccupationMatchesKnownValue) {
    // f(1) = 1/(e+1) ≈ 0.2689414213699951
    FermiDiracSmearing fd(0.01);
    EXPECT_NEAR(fd.occupation(1.0), 0.2689414213699951, 1e-14);
}

TEST(FermiDiracSmearing, OccupationIsMonotonicallyDecreasing) {
    FermiDiracSmearing fd(0.01);
    const std::vector<Real> test_points = {-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0};
    for (size_t index = 1; index < test_points.size(); ++index) {
        EXPECT_GT(fd.occupation(test_points[index - 1]), fd.occupation(test_points[index]));
    }
}

TEST(FermiDiracSmearing, EntropyWeightAtZero) {
    // At f=0.5: entropy_weight = 2 * 0.5 * ln(0.5) = ln(0.5)
    FermiDiracSmearing fd(0.01);
    EXPECT_NEAR(fd.entropy_weight(0.0), std::log(0.5), 1e-14);
}

TEST(FermiDiracSmearing, EntropyWeightIsAlwaysNonPositive) {
    /**
     * -[f ln f + (1-f) ln(1-f)] >= 0 always for 0<=f<=1, so with the sign
     * convention used here (already-negated for the -T*S total-energy term)
     * entropy_weight <= 0 everywhere.
     */
    FermiDiracSmearing fd(0.01);
    for (Real x = -20.0; x <= 20.0; x += 0.1) {
        EXPECT_LE(fd.entropy_weight(x), 1e-14) << "at x=" << x;
    }
}

TEST(FermiDiracSmearing, EntropyWeightVanishesAtLargeX) {
    FermiDiracSmearing fd(0.01);
    EXPECT_NEAR(fd.entropy_weight(40.0), 0.0, 1e-12);
    EXPECT_NEAR(fd.entropy_weight(-40.0), 0.0, 1e-12);
}

TEST(FermiDiracSmearing, OccupationIsContinuousAcrossX36) {
    /**
     * Guards the interval 36 < |x| < 40, which no other case reaches: it sits
     * past the last pinned point below and short of the 1e-12 checks above.
     * A saturation branch anywhere in it is invisible to every other test in
     * this file, because the value it returns (exactly 0 or 1) is what the
     * unbranched expression already converges to.
     */
    FermiDiracSmearing fd(0.01);
    // Strictly inside (0, 1): a clamp at any bound in here would return exactly 0 or 1.
    EXPECT_GT(fd.occupation(36.0001), 0.0);
    EXPECT_LT(fd.occupation(-36.0001), 1.0);
    // Still strictly monotonic across the interval, so no flat clamped region.
    EXPECT_GT(fd.occupation(36.5), fd.occupation(37.5));
    EXPECT_GT(fd.occupation(-37.5), fd.occupation(-36.5));
}

TEST(FermiDiracSmearing, EntropyWeightIsContinuousAcrossX36) {
    /**
     * The entropy term reaches f through occupation(), so a clamp there
     * shows up here as a step: entropy_weight jumps to exactly 0 at the
     * bound instead of decaying smoothly. Named in fermi_dirac.hpp.
     */
    FermiDiracSmearing fd(0.01);
    EXPECT_LT(fd.entropy_weight(36.0001), 0.0);
    EXPECT_LT(fd.entropy_weight(-36.0001), 0.0);
    // Magnitude decays rather than stepping to zero.
    EXPECT_GT(std::abs(fd.entropy_weight(36.5)), std::abs(fd.entropy_weight(37.5)));
}

TEST(FermiDiracSmearing, EntropyWeightIsFiniteInTheSaturatedTails) {
    /**
     * probability_floor is what keeps f*log(f) from evaluating 0*(-inf)
     * once f reaches exactly 0.0 or 1.0. Without it these are NaN.
     */
    FermiDiracSmearing fd(0.01);
    EXPECT_TRUE(std::isfinite(fd.entropy_weight(1000.0)));
    EXPECT_TRUE(std::isfinite(fd.entropy_weight(-1000.0)));
    EXPECT_DOUBLE_EQ(fd.entropy_weight(1000.0), 0.0);
    EXPECT_DOUBLE_EQ(fd.entropy_weight(-1000.0), 0.0);
}

TEST(FermiDiracSmearing, EntropyWeightIsEvenInX) {
    FermiDiracSmearing fd(0.01);
    for (Real x : {0.3, 0.7, 1.5, 2.5}) {
        EXPECT_NEAR(fd.entropy_weight(x), fd.entropy_weight(-x), 1e-14);
    }
}

TEST(FermiDiracSmearing, NameIsFermiDirac) {
    FermiDiracSmearing fd(0.01);
    EXPECT_EQ(fd.name(), "fermi-dirac");
}

TEST(FermiDiracSmearing, SmearingWidthReturnsConstructorValue) {
    const Real width = 0.05;
    FermiDiracSmearing fd(width);
    EXPECT_DOUBLE_EQ(fd.smearing_width(), width);
}

TEST(FermiDiracSmearing, UsesSmearingIsTrueWhenWidthPositive) {
    FermiDiracSmearing fd(0.01);
    EXPECT_TRUE(fd.uses_smearing());
}

TEST(FermiDiracSmearing, UsesSmearingIsFalseWhenWidthZero) {
    FermiDiracSmearing fd(0.0);
    EXPECT_FALSE(fd.uses_smearing());
}
