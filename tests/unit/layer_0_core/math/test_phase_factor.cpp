#include "layer_0_core/math/phase_factor.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace q3;

// ─── compute_phase_factor: e^{-iφ} ─────────────────────────────────────────

TEST(PhaseFactor, PhaseZeroGivesOne) {
    Complex result = compute_phase_factor(0.0);
    EXPECT_NEAR(result.real(), 1.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST(PhaseFactor, PhasePiGivesMinusOne) {
    Complex result = compute_phase_factor(constants::pi);
    EXPECT_NEAR(result.real(), -1.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST(PhaseFactor, PhaseHalfPiGivesMinusI) {
    // e^{-iπ/2} = cos(π/2) - i·sin(π/2) = 0 - i
    Complex result = compute_phase_factor(constants::pi / 2.0);
    EXPECT_NEAR(result.real(), 0.0, 1e-14);
    EXPECT_NEAR(result.imag(), -1.0, 1e-14);
}

TEST(PhaseFactor, PhaseTwoPiGivesOnePeriodicity) {
    Complex result = compute_phase_factor(constants::two_pi);
    EXPECT_NEAR(result.real(), 1.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

// ─── compute_phase_factor_conjugate: e^{+iφ} ───────────────────────────────

TEST(PhaseFactorConjugate, PhaseZeroGivesOne) {
    Complex result = compute_phase_factor_conjugate(0.0);
    EXPECT_NEAR(result.real(), 1.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST(PhaseFactorConjugate, PhaseHalfPiGivesPlusI) {
    // e^{+iπ/2} = cos(π/2) + i·sin(π/2) = 0 + i
    Complex result = compute_phase_factor_conjugate(constants::pi / 2.0);
    EXPECT_NEAR(result.real(), 0.0, 1e-14);
    EXPECT_NEAR(result.imag(), 1.0, 1e-14);
}

TEST(PhaseFactorConjugate, ConjugateRelationship) {
    // compute_phase_factor_conjugate(φ) should equal conj(compute_phase_factor(φ))
    Real phase = 1.2345;
    Complex forward = compute_phase_factor(phase);
    Complex conjugate = compute_phase_factor_conjugate(phase);

    EXPECT_NEAR(conjugate.real(), forward.real(), 1e-14);
    EXPECT_NEAR(conjugate.imag(), -forward.imag(), 1e-14);
}

// ─── compute_angular_phase_factor: (-i)^l ───────────────────────────────────

TEST(AngularPhaseFactor, AngularMomentumZeroGivesOne) {
    // (-i)^0 = 1
    Complex result = compute_angular_phase_factor(0);
    EXPECT_NEAR(result.real(), 1.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST(AngularPhaseFactor, AngularMomentumOneGivesMinusI) {
    // (-i)^1 = -i = (0, -1)
    Complex result = compute_angular_phase_factor(1);
    EXPECT_NEAR(result.real(), 0.0, 1e-14);
    EXPECT_NEAR(result.imag(), -1.0, 1e-14);
}

TEST(AngularPhaseFactor, AngularMomentumTwoGivesMinusOne) {
    // (-i)^2 = -1 = (-1, 0)
    Complex result = compute_angular_phase_factor(2);
    EXPECT_NEAR(result.real(), -1.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST(AngularPhaseFactor, AngularMomentumThreeGivesPlusI) {
    // (-i)^3 = i = (0, 1)
    Complex result = compute_angular_phase_factor(3);
    EXPECT_NEAR(result.real(), 0.0, 1e-14);
    EXPECT_NEAR(result.imag(), 1.0, 1e-14);
}

TEST(AngularPhaseFactor, AngularMomentumFourPeriodicity) {
    // (-i)^4 = 1 (same as l=0)
    Complex result = compute_angular_phase_factor(4);
    EXPECT_NEAR(result.real(), 1.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST(AngularPhaseFactor, AngularMomentumFiveSameAsOne) {
    // (-i)^5 = (-i)^1 = -i
    Complex result = compute_angular_phase_factor(5);
    EXPECT_NEAR(result.real(), 0.0, 1e-14);
    EXPECT_NEAR(result.imag(), -1.0, 1e-14);
}

TEST(AngularPhaseFactor, LargeAngularMomentumPeriodicity) {
    // (-i)^100: 100 % 4 == 0 → 1
    Complex result = compute_angular_phase_factor(100);
    EXPECT_NEAR(result.real(), 1.0, 1e-14);
    EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}
