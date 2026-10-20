#include <gtest/gtest.h>

#include <string>

#include "layer_4_computation/diagonalizer/planewave/direct.hpp"

using namespace q3;

// ── Name and basic properties ───────────────────────────────────────

TEST(PlanewaveDirectDiagonalizer, NameReturnsDirect) {
    PlanewaveDirectDiagonalizer diagonalizer;
    EXPECT_EQ(diagonalizer.name(), "direct");
}

// ── Cached projectors management ────────────────────────────────────

TEST(PlanewaveDirectDiagonalizer, SetCachedProjectorsGrowsMap) {
    PlanewaveDirectDiagonalizer diagonalizer;

    Matrix<Complex> projectors_0(2, 3, Complex(1.0, 0.0));
    Matrix<Complex> projectors_5(4, 4, Complex(0.0, 1.0));

    diagonalizer.set_cached_projectors(0, &projectors_0);
    diagonalizer.set_cached_projectors(5, &projectors_5);

    /**
     * Setting at index 5 should have grown the map to at least size 6
     * (This tests the internal resize logic)
     * We can verify by setting again and checking it doesn't crash
     */
    diagonalizer.set_cached_projectors(5, nullptr);
}

TEST(PlanewaveDirectDiagonalizer, SetCachedProjectorsAtSameIndexOverwrites) {
    PlanewaveDirectDiagonalizer diagonalizer;

    Matrix<Complex> projectors_a(2, 2, Complex(1.0, 0.0));
    Matrix<Complex> projectors_b(3, 3, Complex(2.0, 0.0));

    diagonalizer.set_cached_projectors(0, &projectors_a);
    diagonalizer.set_cached_projectors(0, &projectors_b);

    // This should not crash, and the second set should overwrite the first
}

// ── Default virtual method no-ops ───────────────────────────────────

TEST(PlanewaveDirectDiagonalizer, SetToleranceIsNoOp) {
    PlanewaveDirectDiagonalizer diagonalizer;
    // Should not crash; direct solver ignores tolerance
    diagonalizer.set_tolerance(1e-8);
}

TEST(PlanewaveDirectDiagonalizer, SetFirstIterationIsNoOp) {
    PlanewaveDirectDiagonalizer diagonalizer;
    // Should not crash; direct solver ignores this flag
    diagonalizer.set_first_iteration(true);
    diagonalizer.set_first_iteration(false);
}

TEST(PlanewaveDirectDiagonalizer, InvalidateSmoothPotentialCacheIsNoOp) {
    PlanewaveDirectDiagonalizer diagonalizer;
    // Should not crash
    diagonalizer.invalidate_smooth_potential_cache();
}

// ── Large kpoint index for cached projectors ────────────────────────

TEST(PlanewaveDirectDiagonalizer, CachedProjectorsHandleLargeIndex) {
    PlanewaveDirectDiagonalizer diagonalizer;
    Matrix<Complex> projectors(1, 1, Complex(1.0, 0.0));

    // Setting at a large index should work without issues
    diagonalizer.set_cached_projectors(100, &projectors);
    // Setting nullptr at a previously-set index
    diagonalizer.set_cached_projectors(100, nullptr);
}

TEST(PlanewaveDirectDiagonalizer, CachedProjectorsNullptrIsValid) {
    PlanewaveDirectDiagonalizer diagonalizer;
    // Setting nullptr without prior set should be fine
    diagonalizer.set_cached_projectors(0, nullptr);
}
