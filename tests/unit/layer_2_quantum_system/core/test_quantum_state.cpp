#include "layer_2_quantum_system/core/quantum_state.hpp"
#include <gtest/gtest.h>

using namespace q3;

namespace {

/**
 * Minimal concrete wavefunction type for testing QuantumState.
 *
 * Stores a single real value so that we can verify wavefunction
 * storage in QuantumState without depending on any layer-3 type.
 */
struct MockWavefunction {
    Real amplitude = 0.0;
};

// ─── QuantumState ───────────────────────────────────────────────────────────

} // namespace

TEST(QuantumState, DefaultEigenvalueIsZero) {
    QuantumState<MockWavefunction> state;
    EXPECT_DOUBLE_EQ(state.eigenvalue, 0.0);
}

TEST(QuantumState, DefaultOccupationIsZero) {
    QuantumState<MockWavefunction> state;
    EXPECT_DOUBLE_EQ(state.occupation, 0.0);
}

TEST(QuantumState, IsOccupiedReturnsFalseForZeroOccupation) {
    QuantumState<MockWavefunction> state;
    state.occupation = 0.0;
    EXPECT_FALSE(state.is_occupied());
}

TEST(QuantumState, IsOccupiedReturnsTrueForPositiveOccupation) {
    QuantumState<MockWavefunction> state;
    state.occupation = 1.0;
    EXPECT_TRUE(state.is_occupied());
}

TEST(QuantumState, IsOccupiedReturnsTrueForSmallPositiveOccupation) {
    QuantumState<MockWavefunction> state;
    state.occupation = 1.0e-15;
    EXPECT_TRUE(state.is_occupied());
}

TEST(QuantumState, IsOccupiedReturnsFalseForNegativeOccupation) {
    QuantumState<MockWavefunction> state;
    state.occupation = -0.1;
    EXPECT_FALSE(state.is_occupied());
}
