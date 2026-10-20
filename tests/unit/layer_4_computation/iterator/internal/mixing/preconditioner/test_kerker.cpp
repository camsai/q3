#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/preconditioner/kerker.hpp"

using namespace q3;

namespace {

/// |G|² in bohr⁻² for a small synthetic G set; index 0 is G = 0.
const std::vector<Real> kGSquared = {0.0, 0.25, 1.0, 4.0, 16.0, 100.0};
constexpr int kFirstNonzero = 1;

} // namespace

TEST(KerkerPreconditioner, NameIsKerker) {
    KerkerPreconditioner preconditioner(kGSquared, 1.0, kFirstNonzero);
    EXPECT_EQ(preconditioner.name(), "kerker");
}

/// The filter is G²/(G²+q0²) component by component.
TEST(KerkerPreconditioner, WeightsMatchTheClosedForm) {
    Real q0_squared = 2.0;
    KerkerPreconditioner preconditioner(kGSquared, q0_squared, kFirstNonzero);

    for (std::size_t g_index = kFirstNonzero; g_index < kGSquared.size(); ++g_index) {
        Real expected = kGSquared[g_index] / (kGSquared[g_index] + q0_squared);
        EXPECT_NEAR(preconditioner.weights()[g_index], expected, 1.0e-14)
            << "at G index " << g_index;
    }
}

/// Applying the filter scales each residual component by its weight.
TEST(KerkerPreconditioner, ApplyScalesEachComponentByItsWeight) {
    Real q0_squared = 1.5;
    KerkerPreconditioner preconditioner(kGSquared, q0_squared, kFirstNonzero);

    std::vector<Complex> residual;
    for (std::size_t g_index = 0; g_index < kGSquared.size(); ++g_index) {
        residual.emplace_back(1.0 + 0.5 * static_cast<Real>(g_index),
                              -0.25 * static_cast<Real>(g_index));
    }
    std::vector<Complex> original = residual;
    preconditioner.apply(residual);

    for (std::size_t g_index = kFirstNonzero; g_index < kGSquared.size(); ++g_index) {
        Real weight = kGSquared[g_index] / (kGSquared[g_index] + q0_squared);
        EXPECT_NEAR(residual[g_index].real(), original[g_index].real() * weight, 1.0e-14);
        EXPECT_NEAR(residual[g_index].imag(), original[g_index].imag() * weight, 1.0e-14);
    }
}

/**
 * G = 0 carries the total charge, which mixing must not touch: the weight is
 * left at 1 and apply() skips the component entirely.
 */
TEST(KerkerPreconditioner, LeavesGZeroUntouched) {
    KerkerPreconditioner preconditioner(kGSquared, 3.0, kFirstNonzero);
    EXPECT_DOUBLE_EQ(preconditioner.weights()[0], 1.0);

    std::vector<Complex> residual(kGSquared.size(), Complex(2.0, -1.0));
    preconditioner.apply(residual);
    EXPECT_DOUBLE_EQ(residual[0].real(), 2.0);
    EXPECT_DOUBLE_EQ(residual[0].imag(), -1.0);
}

/// The filter tends to the identity at large G: short wavelengths are not screened.
TEST(KerkerPreconditioner, TendsToIdentityAtLargeG) {
    std::vector<Real> g_squared = {0.0, 1.0e6, 1.0e10};
    KerkerPreconditioner preconditioner(g_squared, 1.0, 1);
    EXPECT_NEAR(preconditioner.weights()[1], 1.0, 1.0e-5);
    EXPECT_NEAR(preconditioner.weights()[2], 1.0, 1.0e-9);
    EXPECT_LT(preconditioner.weights()[1], 1.0);
}

/// A larger q0 screens harder, so every weight decreases monotonically in q0.
TEST(KerkerPreconditioner, LargerScreeningDampsMore) {
    KerkerPreconditioner weak(kGSquared, 0.5, kFirstNonzero);
    KerkerPreconditioner strong(kGSquared, 5.0, kFirstNonzero);
    for (std::size_t g_index = kFirstNonzero; g_index < kGSquared.size(); ++g_index) {
        EXPECT_LT(strong.weights()[g_index], weak.weights()[g_index]);
    }
}
