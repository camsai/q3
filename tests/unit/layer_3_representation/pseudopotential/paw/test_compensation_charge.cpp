/**
 * Compensation-charge synthesis: the admissible shape and the bessel pieces
 * it is built from.
 *
 * A PAW dataset that states its augmentation *charges* but ships no radial
 * Q_ij^L(r) has them rebuilt on read (synthesize_augmentation_functions). The
 * shape has to be admissible — confined to the augmentation sphere, unit L-th
 * moment, and no kink where it meets zero — and each channel pair's
 * synthesized function has to carry the moment the partial waves demand. The
 * tests that check both on a dataset that needs synthesis read a licensed
 * one and live with the add-on that reads it; these pin the building blocks.
 *
 * A UPF PAW dataset is used for the negative control: it ships its own
 * generator-produced Q_ij^L, and synthesis must leave those untouched rather
 * than overwrite information the file actually contains.
 */

#include "io/pseudopotential/factory.hpp"
#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_3_representation/pseudopotential/paw/compensation_charge.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace q3;
using namespace q3::paw_compensation;

TEST(CompensationChargeTest, SphericalBesselZerosAreZeros) {
    /**
     * j_0's zeros are the multiples of π in closed form, which pins the
     * bracketing search itself; for L > 0 there is no closed form, so the test
     * is that the function vanishes there and that the zeros come out ordered.
     */
    auto zeros_l0 = spherical_bessel_zeros(0, 4);
    ASSERT_EQ(zeros_l0.size(), 4u);
    for (int index = 0; index < 4; index++)
        EXPECT_NEAR(zeros_l0[index], M_PI * (index + 1), 1e-10);

    for (int degree = 1; degree <= 4; degree++) {
        auto zeros = spherical_bessel_zeros(degree, 3);
        ASSERT_EQ(zeros.size(), 3u) << "degree " << degree;
        for (size_t index = 0; index < zeros.size(); index++) {
            EXPECT_NEAR(spherical_bessel(degree, zeros[index]), 0.0, 1e-12)
                << "degree " << degree << " zero " << index;
            if (index > 0)
                EXPECT_GT(zeros[index], zeros[index - 1]);
        }
    }
}

TEST(CompensationChargeTest, BesselDerivativeMatchesFiniteDifferences) {
    constexpr Real step = 1e-6;
    for (int degree = 0; degree <= 4; degree++) {
        for (Real argument : {0.3, 1.0, 2.5, 6.0}) {
            Real numerical = (spherical_bessel(degree, argument + step) -
                              spherical_bessel(degree, argument - step)) /
                             (2.0 * step);
            /**
             * Relative: j_l spans orders of magnitude over this range (j_4 at
             * x = 0.3 is 1e-5 of j_0 there), so an absolute bound would be
             * slack for the small entries and would let a wrong recurrence
             * through exactly where the recurrence is delicate.
             */
            EXPECT_NEAR(spherical_bessel_derivative(degree, argument),
                        numerical,
                        1e-14 + 1e-8 * std::abs(numerical))
                << "degree " << degree << " at " << argument;
        }
    }
}

TEST(CompensationChargeTest, SynthesisLeavesAUpfDatasetAlone) {
    /**
     * A UPF PAW file's Q_ij^L are its generator's. Re-deriving them from a
     * model shape would silently replace real data with an approximation, so
     * synthesis has to be a no-op here — and the check is byte-for-byte
     * equality, not agreement to a tolerance.
     */
    auto pseudopotential =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/paw/Si.upf");
    auto& paw = dynamic_cast<PawPseudopotential&>(*pseudopotential);

    auto before = paw.augmentation_function_by_angular_momentum;
    ASSERT_FALSE(before.empty());
    synthesize_augmentation_functions(paw);
    EXPECT_EQ(paw.augmentation_function_by_angular_momentum, before);
}
