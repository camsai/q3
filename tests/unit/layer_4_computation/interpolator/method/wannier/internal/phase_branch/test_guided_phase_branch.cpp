#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/guided_phase_branch.hpp"

using namespace q3;

/**
 * The phase lands on the branch nearest -b . r_guide. A centre at 10.8 bohr
 * along a b of 0.2618 per bohr implies a phase near -0.9 pi; an overlap whose
 * principal value is +0.95 pi belongs to the same function one turn away,
 * and is read as -1.05 pi.
 */
TEST(GuidedPhaseBranch, ThePhaseIsTakenOnTheBranchTheCentreImplies) {
    const GuidedPhaseBranch branch({Vector3D{10.8, 0.0, 0.0}});
    const Vector3D b_vector{constants::two_pi / 24.0, 0.0, 0.0};

    const Real phase = branch.phase_of(std::polar(0.9, 0.95 * constants::pi), 0, b_vector);
    EXPECT_NEAR(phase, -1.05 * constants::pi, 1.0e-12);
}

/// As a diagonal element rotates through the cut, the guided phase passes through it smoothly.
TEST(GuidedPhaseBranch, ItIsContinuousAcrossTheNegativeRealAxis) {
    const GuidedPhaseBranch branch({Vector3D{12.0, 0.0, 0.0}});
    const Vector3D b_vector{constants::two_pi / 24.0, 0.0, 0.0};
    const Real above = branch.phase_of(std::polar(1.0, constants::pi - 1.0e-6), 0, b_vector);
    const Real below = branch.phase_of(std::polar(1.0, constants::pi + 1.0e-6), 0, b_vector);
    EXPECT_NEAR(above - below, -2.0e-6, 1.0e-9);
}

/// Where no phase is near the cut the guide changes nothing.
TEST(GuidedPhaseBranch, AwayFromTheCutItIsThePrincipalValue) {
    const GuidedPhaseBranch branch({Vector3D{0.4, 0.0, 0.0}, Vector3D{-0.3, 0.0, 0.0}});
    const Vector3D b_vector{constants::two_pi / 48.0, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(branch.phase_of(std::polar(0.9, -0.05), 0, b_vector), -0.05);
    EXPECT_DOUBLE_EQ(branch.phase_of(std::polar(0.9, 0.04), 1, b_vector), 0.04);
}

TEST(GuidedPhaseBranch, ItServesExactlyAsManyFunctionsAsItHasCentres) {
    const GuidedPhaseBranch branch({Vector3D{0.0, 0.0, 0.0}, Vector3D{1.0, 0.0, 0.0}});
    EXPECT_NO_THROW(branch.require_serves(2));
    EXPECT_THROW(branch.require_serves(3), std::invalid_argument);
    EXPECT_THROW(GuidedPhaseBranch(std::vector<Vector3D>{}), std::invalid_argument);
}
