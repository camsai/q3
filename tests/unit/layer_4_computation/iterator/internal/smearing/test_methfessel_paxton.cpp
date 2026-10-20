#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_4_computation/iterator/internal/smearing/methfessel_paxton.hpp"

using namespace q3;

TEST(MethfesselPaxtonSmearing, OccupationAtZeroIsOneHalf) {
    // f(0) = erfc(0)/2 - 0 = 1/2 (the correction term is odd in x, vanishes at 0)
    MethfesselPaxtonSmearing mp(0.01);
    EXPECT_DOUBLE_EQ(mp.occupation(0.0), 0.5);
}

TEST(MethfesselPaxtonSmearing, OccupationLargeNegativeApproachesOne) {
    MethfesselPaxtonSmearing mp(0.01);
    EXPECT_NEAR(mp.occupation(-10.0), 1.0, 1e-12);
}

TEST(MethfesselPaxtonSmearing, OccupationLargePositiveApproachesZero) {
    MethfesselPaxtonSmearing mp(0.01);
    EXPECT_NEAR(mp.occupation(10.0), 0.0, 1e-12);
}

TEST(MethfesselPaxtonSmearing, OccupationAntisymmetricAboutOneHalf) {
    /**
     * f(-x) - 0.5 = -(f(x) - 0.5): the correction term is odd in x, exactly
     * like the erfc(x)/2 base term is antisymmetric about (0, 0.5).
     */
    MethfesselPaxtonSmearing mp(0.01);
    for (Real x : {0.3, 0.7, 1.5, 2.5}) {
        EXPECT_NEAR(mp.occupation(-x) - 0.5, -(mp.occupation(x) - 0.5), 1e-12);
    }
}

TEST(MethfesselPaxtonSmearing, OccupationCanExceedUnitInterval) {
    /**
     * The known, accepted trade-off of first-order Methfessel-Paxton: the
     * Hermite correction overshoots near the Fermi level, giving f slightly
     * outside [0, 1] (unlike Gaussian/Fermi-Dirac). This is expected, not a
     * bug — verifying it guards against silently "fixing" the formula later.
     */
    MethfesselPaxtonSmearing mp(0.01);
    bool found_out_of_range = false;
    for (Real x = -3.0; x <= 3.0; x += 0.01) {
        Real f = mp.occupation(x);
        if (f < 0.0 || f > 1.0) {
            found_out_of_range = true;
            break;
        }
    }
    EXPECT_TRUE(found_out_of_range);
}

TEST(MethfesselPaxtonSmearing, EntropyWeightAtZero) {
    // entropy_weight(0) = (2*0 - 1) / (4*sqrt(pi)) = -1/(4*sqrt(pi))
    MethfesselPaxtonSmearing mp(0.01);
    const Real expected = -0.25 / std::sqrt(constants::pi);
    EXPECT_NEAR(mp.entropy_weight(0.0), expected, 1e-15);
}

TEST(MethfesselPaxtonSmearing, EntropyWeightIsEvenInX) {
    // entropy_weight(x) depends only on x^2, so it must be an even function.
    MethfesselPaxtonSmearing mp(0.01);
    for (Real x : {0.3, 0.7, 1.5, 2.5}) {
        EXPECT_NEAR(mp.entropy_weight(x), mp.entropy_weight(-x), 1e-14);
    }
}

TEST(MethfesselPaxtonSmearing, EntropyWeightVanishesAtLargeX) {
    MethfesselPaxtonSmearing mp(0.01);
    EXPECT_NEAR(mp.entropy_weight(10.0), 0.0, 1e-12);
}

TEST(MethfesselPaxtonSmearing, NameIsMethfesselPaxton) {
    MethfesselPaxtonSmearing mp(0.01);
    EXPECT_EQ(mp.name(), "methfessel-paxton");
}

TEST(MethfesselPaxtonSmearing, SmearingWidthReturnsConstructorValue) {
    const Real width = 0.05;
    MethfesselPaxtonSmearing mp(width);
    EXPECT_DOUBLE_EQ(mp.smearing_width(), width);
}

TEST(MethfesselPaxtonSmearing, UsesSmearingIsTrueWhenWidthPositive) {
    MethfesselPaxtonSmearing mp(0.01);
    EXPECT_TRUE(mp.uses_smearing());
}

TEST(MethfesselPaxtonSmearing, UsesSmearingIsFalseWhenWidthZero) {
    MethfesselPaxtonSmearing mp(0.0);
    EXPECT_FALSE(mp.uses_smearing());
}

// ---- order N > 1 (Methfessel & Paxton, PRB 40, 3616, eqs. for f_N and S_N) ----

TEST(MethfesselPaxtonSmearing, DefaultOrderIsOne) {
    EXPECT_EQ(MethfesselPaxtonSmearing(0.01).order(), 1);
}

TEST(MethfesselPaxtonSmearing, OrderBelowOneThrows) {
    EXPECT_THROW(MethfesselPaxtonSmearing(0.01, 0), std::invalid_argument);
}

TEST(MethfesselPaxtonSmearing, OrderTwoMatchesExplicitHermiteForm) {
    /**
     * N = 2 written out: H_1 = 2x, H_3 = 8x^3 - 12x, H_4 = 16x^4 - 48x^2 + 12,
     * A_1 = -1/(4 sqrt(pi)), A_2 = 1/(32 sqrt(pi)).
     */
    const MethfesselPaxtonSmearing mp(0.01, 2);
    const Real sqrtpi = std::sqrt(constants::pi);
    for (Real x : {-2.1, -0.7, 0.0, 0.3, 1.4}) {
        const Real gaussian = std::exp(-x * x);
        const Real expected_occupation = 0.5 * std::erfc(x) -
                                         (2.0 * x) / (4.0 * sqrtpi) * gaussian +
                                         (8.0 * x * x * x - 12.0 * x) / (32.0 * sqrtpi) * gaussian;
        const Real hermite_4 = 16.0 * x * x * x * x - 48.0 * x * x + 12.0;
        const Real expected_entropy_weight = -0.5 * hermite_4 / (32.0 * sqrtpi) * gaussian;
        EXPECT_NEAR(mp.occupation(x), expected_occupation, 1e-14) << "x = " << x;
        EXPECT_NEAR(mp.entropy_weight(x), expected_entropy_weight, 1e-14) << "x = " << x;
    }
}

TEST(MethfesselPaxtonSmearing, EntropyDerivativeEqualsMinusXTimesOccupationDerivative) {
    /**
     * MP's generalized entropy satisfies dS_N/dx = -x δ_N(x) with δ_N = -df_N/dx, so
     * for entropy_weight = -S_N: d(entropy_weight)/dx = -x df_N/dx at every order. This
     * ties occupation() and entropy_weight() together without restating either formula.
     */
    const Real step = 1e-5;
    for (int order : {1, 2, 3, 5}) {
        const MethfesselPaxtonSmearing mp(0.01, order);
        for (Real x : {-1.7, -0.4, 0.25, 0.9, 2.2}) {
            const Real d_entropy =
                (mp.entropy_weight(x + step) - mp.entropy_weight(x - step)) / (2.0 * step);
            const Real d_occupation =
                (mp.occupation(x + step) - mp.occupation(x - step)) / (2.0 * step);
            EXPECT_NEAR(d_entropy, -x * d_occupation, 1e-8) << "order " << order << ", x = " << x;
        }
    }
}

TEST(MethfesselPaxtonSmearing, HigherOrderDeltaHasVanishingLowMoments) {
    /**
     * δ_N reproduces polynomials of degree 2N exactly: ∫ x^{2k} δ_N dx = 0 for 1 <= k <= N
     * (and = 1 for k = 0). Integrate -df_N/dx by differences of f_N on [-8, 8].
     */
    for (int order : {2, 3}) {
        const MethfesselPaxtonSmearing mp(0.01, order);
        const int intervals = 16000;
        const Real lower = -8.0;
        const Real width = 16.0 / intervals;
        for (int k = 0; k <= order; ++k) {
            Real moment = 0.0;
            for (int i = 0; i < intervals; ++i) {
                const Real left = lower + i * width;
                const Real midpoint = left + 0.5 * width;
                const Real delta_times_width = mp.occupation(left) - mp.occupation(left + width);
                moment += std::pow(midpoint, 2 * k) * delta_times_width;
            }
            EXPECT_NEAR(moment, k == 0 ? 1.0 : 0.0, 1e-6) << "order " << order << ", k = " << k;
        }
    }
}
