#include <gtest/gtest.h>

#include <vector>

#include "layer_4_computation/effective_potential/internal/hubbard/occupation_mixer.hpp"

using namespace q3;

namespace {

/// One manifold, one spin, one element — enough to drive every branch.
HubbardOccupationMatrices single(Real value) {
    return {{{Complex(value, 0.0)}}};
}

constexpr Real kBeta = 0.3;
constexpr Real kThreshold = 1e-9;

} // namespace

TEST(HubbardOccupationMixer, MonotoneSequenceNeverDampsAndPassesTheFreshMatrixThrough) {
    /**
     * The property that keeps well-behaved systems bit-for-bit identical to
     * the undamped code: no non-monotone step, so no damping, so the blend
     * factor stays exactly 1.0 and the mixer returns what it was handed.
     */
    HubbardOccupationMixer mixer;
    mixer.blend(single(0.0), kBeta, kThreshold); // bootstrap

    Real approach[] = {0.5, 0.25, 0.125, 0.0625, 0.03125};
    for (Real value : approach) {
        const auto& damped = mixer.blend(single(value), kBeta, kThreshold);
        EXPECT_FALSE(mixer.damping_active());
        EXPECT_DOUBLE_EQ(damped[0][0][0].real(), value) << "monotone input must pass through";
    }
}

TEST(HubbardOccupationMixer, AlternatingSequenceTripsTheTriggerOnTheThirdEvent) {
    /**
     * An oscillation alternates rather than growing every step, which is why
     * the trigger counts cumulatively. Three non-monotone events is the
     * documented threshold (`hubbard_damping_trigger_iterations`).
     */
    HubbardOccupationMixer mixer;
    mixer.blend(single(0.0), kBeta, kThreshold);

    // Each pair moves closer then further: one non-monotone event per pair.
    Real oscillation[] = {1.0, 0.1, 1.0, 0.1, 1.0, 0.1};
    int events_seen = 0;
    for (Real value : oscillation) {
        bool was_active = mixer.damping_active();
        mixer.blend(single(value), kBeta, kThreshold);
        if (!was_active && mixer.damping_active())
            events_seen++;
    }
    EXPECT_TRUE(mixer.damping_active()) << "a sustained oscillation must earn damping";
    EXPECT_EQ(events_seen, 1) << "damping should latch on once, not flap";
}

TEST(HubbardOccupationMixer, ResidualIsTheDistanceToTheFixedPointNotTheStepTaken) {
    /**
     * These differ by the damping factor, and the residual is what gates SCF
     * exit — a member doc that said "step" here would mislead anyone tuning
     * the convergence threshold.
     */
    HubbardOccupationMixer mixer;
    mixer.blend(single(0.0), kBeta, kThreshold);
    mixer.blend(single(0.5), kBeta, kThreshold);
    // n_damped is still 0.0 at entry (monotone => passthrough), so ||0.5 - 0||^2.
    EXPECT_NEAR(mixer.residual(), 0.25, 1e-12);
}

TEST(HubbardOccupationMixer, AFixedPointIsReproducedExactly) {
    /**
     * Feeding the matrix the mixer already holds must change nothing and
     * report a zero residual, damped or not — this is what makes the whole
     * scheme a no-op at self-consistency.
     */
    HubbardOccupationMixer mixer;
    mixer.blend(single(0.42), kBeta, kThreshold);
    const auto& damped = mixer.blend(single(0.42), kBeta, kThreshold);
    EXPECT_DOUBLE_EQ(damped[0][0][0].real(), 0.42);
    EXPECT_NEAR(mixer.residual(), 0.0, 1e-30);
}

TEST(HubbardOccupationMixer, ALayoutChangeRebootstrapsInsteadOfIndexingAMismatchedCache) {
    /**
     * The manifold layout should not change mid-run, but if it does, both the
     * damped and the residual caches are stale in the same way. Indexing
     * either one element-by-element against the new shape is an out-of-bounds
     * read, so the mixer must start over rather than blend.
     */
    HubbardOccupationMixer mixer;
    mixer.blend(single(0.5), kBeta, kThreshold);

    HubbardOccupationMatrices wider = {{{Complex(0.1, 0.0), Complex(0.2, 0.0)}}};
    const auto& damped = mixer.blend(wider, kBeta, kThreshold);

    ASSERT_EQ(damped[0][0].size(), 2u);
    EXPECT_DOUBLE_EQ(damped[0][0][0].real(), 0.1) << "must re-bootstrap from the new shape";
    EXPECT_DOUBLE_EQ(damped[0][0][1].real(), 0.2);
    EXPECT_GT(mixer.residual(), 1e30) << "no prior state to measure against";
}

TEST(HubbardOccupationMixer, DampedStepsStayBoundedUnderAHardOscillation) {
    /**
     * With the Anderson clamp in place, a violent alternation must still
     * produce steps inside the interval it is oscillating over. An unclamped
     * secant is exactly what would overshoot here.
     */
    HubbardOccupationMixer mixer;
    mixer.blend(single(0.0), kBeta, kThreshold);
    for (int iteration = 0; iteration < 12; iteration++) {
        const auto& damped = mixer.blend(single(iteration % 2 == 0 ? 1.0 : 0.0), kBeta, kThreshold);
        Real value = damped[0][0][0].real();
        EXPECT_GE(value, -0.5) << "iteration " << iteration;
        EXPECT_LE(value, 1.5) << "iteration " << iteration;
    }
    EXPECT_TRUE(mixer.damping_active());
}

TEST(HubbardOccupationMixer, AndersonStandsDownInsideTheConvergenceThreshold) {
    /**
     * The final approach must run on the plain damped path: with the
     * extrapolation still active there, cross-basis energies moved by more
     * than the 1e-7 Ry gate they are compared at. Below the threshold the step
     * must therefore be exactly `beta * residual`.
     */
    HubbardOccupationMixer mixer;
    mixer.blend(single(0.0), kBeta, kThreshold);
    for (int iteration = 0; iteration < 6; iteration++)
        mixer.blend(single(iteration % 2 == 0 ? 1.0 : 0.0), kBeta, kThreshold);
    ASSERT_TRUE(mixer.damping_active());

    Real before = mixer.damped()[0][0][0].real();
    Real target = before + 1e-8; // mismatch^2 = 1e-16, well inside the threshold
    const auto& damped = mixer.blend(single(target), kBeta, kThreshold);
    EXPECT_NEAR(damped[0][0][0].real(), before + kBeta * (target - before), 1e-18);
}
