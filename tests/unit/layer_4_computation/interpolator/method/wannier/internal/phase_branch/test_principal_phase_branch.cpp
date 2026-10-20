#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/principal_phase_branch.hpp"

using namespace q3;

/// The principal value, whatever the function and the direction: it is what std::arg returns.
TEST(PrincipalPhaseBranch, ThePhaseIsThePrincipalValueOfTheOverlap) {
    const PrincipalPhaseBranch branch;
    const Vector3D b_vector{0.3, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(branch.phase_of(Complex(0.0, 1.0), 0, b_vector), constants::pi / 2.0);
    EXPECT_DOUBLE_EQ(branch.phase_of(Complex(-1.0, 0.0), 3, b_vector), constants::pi);
    EXPECT_DOUBLE_EQ(branch.phase_of(std::polar(0.9, -3.0), 1, b_vector), -3.0);
}

/// It jumps by 2 pi across the negative real axis, which is what a guided branch removes.
TEST(PrincipalPhaseBranch, ItJumpsAcrossTheNegativeRealAxis) {
    const PrincipalPhaseBranch branch;
    const Vector3D b_vector{0.3, 0.0, 0.0};
    const Real above = branch.phase_of(std::polar(1.0, constants::pi - 1.0e-6), 0, b_vector);
    const Real below = branch.phase_of(std::polar(1.0, constants::pi + 1.0e-6), 0, b_vector);
    EXPECT_NEAR(above - below, constants::two_pi, 1.0e-5);
}

TEST(PrincipalPhaseBranch, ItServesAnyNumberOfFunctions) {
    const PrincipalPhaseBranch branch;
    EXPECT_NO_THROW(branch.require_serves(1));
    EXPECT_NO_THROW(branch.require_serves(18));
}
