#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_4_computation/iterator/internal/smearing/marzari_vanderbilt.hpp"

using namespace q3;

/**
 * Reference values generated from the published closed-form cold-smearing
 * expressions of Marzari & Vanderbilt, PRL 82, 3296 (1999) (the n=-1 member of
 * the Methfessel-Paxton family), evaluated in this codebase's x=(E-E_F)/sigma
 * sign convention (Quantum ESPRESSO's internal x is negated relative to this
 * one — verified via the already-QE-validated Gaussian and Fermi-Dirac schemes
 * matching exactly under that same substitution) and cross-checked numerically
 * against a real QE run (Cu FCC, matched degauss): see
 * tests/validation/test_qe_energy_comparison.py (Cu-NC-MarzariVanderbilt).
 */

TEST(MarzariVanderbiltSmearing, OccupationLargeNegativeApproachesOne) {
    MarzariVanderbiltSmearing mv(0.01);
    EXPECT_NEAR(mv.occupation(-10.0), 1.0, 1e-12);
}

TEST(MarzariVanderbiltSmearing, OccupationLargePositiveApproachesZero) {
    MarzariVanderbiltSmearing mv(0.01);
    EXPECT_NEAR(mv.occupation(10.0), 0.0, 1e-12);
}

TEST(MarzariVanderbiltSmearing, OccupationIsNotSymmetricAboutZero) {
    /**
     * Unlike Gaussian/Methfessel-Paxton/Fermi-Dirac, cold smearing's defining
     * asymmetric shift means f(0) != 0.5 — this is expected, not a bug.
     */
    MarzariVanderbiltSmearing mv(0.01);
    EXPECT_NEAR(mv.occupation(0.0),
                0.5 * std::erfc(1.0 / std::sqrt(2.0)) +
                    std::exp(-0.5) / std::sqrt(2.0 * constants::pi),
                1e-14);
    EXPECT_NE(mv.occupation(0.0), 0.5);
}

TEST(MarzariVanderbiltSmearing, OccupationOvershootsUnityNearMinusOnePointFour) {
    /**
     * The known, accepted trade-off of cold smearing: a single smooth ~8%
     * overshoot above 1 around x ~ -1.4 (well below the Fermi level, not near
     * the transition). Verifying this guards against silently "fixing" the
     * formula later based on an incorrect assumption that cold smearing must
     * be strictly bounded in [0, 1] (it isn't — only that its overshoot
     * behaves better than plain Methfessel-Paxton's oscillation in practice).
     */
    MarzariVanderbiltSmearing mv(0.01);
    Real max_occupation = 0.0;
    for (Real x = -3.0; x <= 0.0; x += 0.01) {
        max_occupation = std::max(max_occupation, mv.occupation(x));
    }
    EXPECT_GT(max_occupation, 1.0);
    EXPECT_LT(max_occupation, 1.2);
}

TEST(MarzariVanderbiltSmearing, EntropyWeightVanishesAtLargeX) {
    MarzariVanderbiltSmearing mv(0.01);
    EXPECT_NEAR(mv.entropy_weight(10.0), 0.0, 1e-12);
    EXPECT_NEAR(mv.entropy_weight(-10.0), 0.0, 1e-12);
}

TEST(MarzariVanderbiltSmearing, EntropyWeightAtZero) {
    // s(0) = -y*exp(-y^2)/sqrt(2*pi), y = 1/sqrt(2)
    MarzariVanderbiltSmearing mv(0.01);
    const Real y = 1.0 / std::sqrt(2.0);
    const Real expected = -y * std::exp(-y * y) / std::sqrt(2.0 * constants::pi);
    EXPECT_NEAR(mv.entropy_weight(0.0), expected, 1e-14);
}

TEST(MarzariVanderbiltSmearing, NameIsMarzariVanderbilt) {
    MarzariVanderbiltSmearing mv(0.01);
    EXPECT_EQ(mv.name(), "marzari-vanderbilt");
}

TEST(MarzariVanderbiltSmearing, SmearingWidthReturnsConstructorValue) {
    const Real width = 0.05;
    MarzariVanderbiltSmearing mv(width);
    EXPECT_DOUBLE_EQ(mv.smearing_width(), width);
}

TEST(MarzariVanderbiltSmearing, UsesSmearingIsTrueWhenWidthPositive) {
    MarzariVanderbiltSmearing mv(0.01);
    EXPECT_TRUE(mv.uses_smearing());
}

TEST(MarzariVanderbiltSmearing, UsesSmearingIsFalseWhenWidthZero) {
    MarzariVanderbiltSmearing mv(0.0);
    EXPECT_FALSE(mv.uses_smearing());
}
