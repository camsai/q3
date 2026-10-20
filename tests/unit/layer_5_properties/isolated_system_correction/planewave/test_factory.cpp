#include <gtest/gtest.h>

#include <stdexcept>

#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/factory.hpp"

using namespace q3;

// ── Scheme resolution ───────────────────────────────────────────────

TEST(IsolatedSystemCorrectionFactory, NoneResolvesToNullCalculator) {
    auto calculator = create_isolated_system_correction_calculator("none");
    EXPECT_EQ(calculator->name(), "none");
}

TEST(IsolatedSystemCorrectionFactory, EmptyStringResolvesToNullCalculator) {
    auto calculator = create_isolated_system_correction_calculator("");
    EXPECT_EQ(calculator->name(), "none");
}

TEST(IsolatedSystemCorrectionFactory, MakovPayneResolvesToMakovPayneCalculator) {
    auto calculator = create_isolated_system_correction_calculator("makov-payne");
    EXPECT_EQ(calculator->name(), "makov-payne");
}

TEST(IsolatedSystemCorrectionFactory, MartynaTuckermanResolvesToMartynaTuckermanCalculator) {
    auto calculator = create_isolated_system_correction_calculator("martyna-tuckerman");
    EXPECT_EQ(calculator->name(), "martyna-tuckerman");
}

TEST(IsolatedSystemCorrectionFactory, UnknownSchemeThrows) {
    /**
     * An unrecognized scheme must fail loudly, not silently run without the
     * correction the user asked for. (The YAML parser validates the value
     * too; this covers programmatic callers that bypass parsing.)
     * "point-countercharge" is a real QE-family scheme name not yet
     * implemented in Q3 (see IsolatedSystemCorrectionCalculator's
     * docstring), picked so this test can't accidentally start passing for
     * the wrong reason (an actually-unrecognized string) once it is added.
     */
    EXPECT_THROW(create_isolated_system_correction_calculator("point-countercharge"),
                 std::runtime_error);
    EXPECT_THROW(create_isolated_system_correction_calculator("mp"), std::runtime_error);
}

/**
 * ── Null object contract ────────────────────────────────────────────
 *
 * The SCF flavor calls the correction interface unconditionally, so the
 * "none" scheme must report an unapplied, zero-term result -- reporting code
 * keys off `applied` to decide whether to emit anything.
 */

TEST(NullIsolatedSystemCorrectionCalculator, ReportsUnappliedEmptyResult) {
    auto calculator = create_isolated_system_correction_calculator("none");

    /**
     * The Null scheme must not touch the state at all -- a default-constructed
     * (null-pointer-filled) SCF state passing through proves it.
     */
    NormConservingPlanewaveKohnShamSCFState scf_state;
    auto result = calculator->compute(scf_state);

    EXPECT_FALSE(result.applied);
    EXPECT_EQ(result.method, "none");
    EXPECT_TRUE(result.terms_in_ry.empty());
    EXPECT_EQ(result.total_in_ry(), 0.0);
}
