#include "layer_2_quantum_system/quantum_system.hpp"
#include <gtest/gtest.h>

using namespace q3;

namespace {

/**
 * Minimal mock wavefunction for testing QuantumSystem.
 */
struct MockWavefunction {
    Real amplitude = 0.0;
};

// Use QuantumState with our mock wavefunction
using MockQuantumState = QuantumState<MockWavefunction>;

/**
 * Concrete subclass of QuantumSystem that exposes protected
 * state storage for testing.
 */
class TestableQuantumSystem : public QuantumSystem<MockQuantumState> {
  public:
    /**
     * Add a quantum state to the system.
     */
    void add_quantum_state(const MockQuantumState& state) {
        quantum_states_.push_back(state);
    }

    /**
     * Resize the state storage to a given count.
     */
    void resize_states(int count) {
        quantum_states_.resize(count);
    }
};

// ─── QuantumSystem ──────────────────────────────────────────────────────────

} // namespace

TEST(QuantumSystem, DefaultCrystalStructureIsNull) {
    TestableQuantumSystem system;
    EXPECT_EQ(system.crystal_structure, nullptr);
}

TEST(QuantumSystem, DefaultNumberOfQuantumStatesIsZero) {
    TestableQuantumSystem system;
    EXPECT_EQ(system.number_of_quantum_states(), 0);
}

TEST(QuantumSystem, NumberOfQuantumStatesAfterAdding) {
    TestableQuantumSystem system;

    MockQuantumState state;
    state.eigenvalue = -1.0;
    state.occupation = 1.0;
    system.add_quantum_state(state);

    EXPECT_EQ(system.number_of_quantum_states(), 1);
}

TEST(QuantumSystem, NumberOfQuantumStatesAfterAddingMultiple) {
    TestableQuantumSystem system;

    for (int index = 0; index < 5; ++index) {
        MockQuantumState state;
        state.eigenvalue = static_cast<Real>(index) * -0.5;
        system.add_quantum_state(state);
    }

    EXPECT_EQ(system.number_of_quantum_states(), 5);
}

TEST(QuantumSystem, GetQuantumStateByIndexConst) {
    TestableQuantumSystem system;

    MockQuantumState state;
    state.eigenvalue = -2.5;
    state.occupation = 1.0;
    state.wavefunction.amplitude = 3.14;
    system.add_quantum_state(state);

    const auto& retrieved_state = system.get_quantum_state_by_index(0);
    EXPECT_DOUBLE_EQ(retrieved_state.eigenvalue, -2.5);
    EXPECT_DOUBLE_EQ(retrieved_state.occupation, 1.0);
    EXPECT_DOUBLE_EQ(retrieved_state.wavefunction.amplitude, 3.14);
}

TEST(QuantumSystem, GetQuantumStateByIndexMutable) {
    TestableQuantumSystem system;

    MockQuantumState state;
    state.eigenvalue = 0.0;
    system.add_quantum_state(state);

    system.get_quantum_state_by_index(0).eigenvalue = -5.0;
    EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(0).eigenvalue, -5.0);
}

TEST(QuantumSystem, GetQuantumStateByIndexReturnsCorrectState) {
    TestableQuantumSystem system;

    for (int index = 0; index < 3; ++index) {
        MockQuantumState state;
        state.eigenvalue = static_cast<Real>(index) * 1.1;
        system.add_quantum_state(state);
    }

    EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(0).eigenvalue, 0.0);
    EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(1).eigenvalue, 1.1);
    EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(2).eigenvalue, 2.2);
}

TEST(QuantumSystem, CrystalStructureCanBeSetToNonNull) {
    TestableQuantumSystem system;

    // Use a dummy pointer value to verify storage (not dereferenced)
    int dummy = 0;
    system.crystal_structure = reinterpret_cast<Crystal*>(&dummy);
    EXPECT_NE(system.crystal_structure, nullptr);

    // Reset to null
    system.crystal_structure = nullptr;
    EXPECT_EQ(system.crystal_structure, nullptr);
}

TEST(QuantumSystem, ResizeStatesCreatesDefaultStates) {
    TestableQuantumSystem system;
    system.resize_states(4);

    EXPECT_EQ(system.number_of_quantum_states(), 4);
    for (int index = 0; index < 4; ++index) {
        EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(index).eigenvalue, 0.0);
        EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(index).occupation, 0.0);
        EXPECT_FALSE(system.get_quantum_state_by_index(index).is_occupied());
    }
}

TEST(QuantumSystem, MutableAccessDoesNotAffectOtherStates) {
    TestableQuantumSystem system;
    system.resize_states(3);

    system.get_quantum_state_by_index(1).eigenvalue = -10.0;

    EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(0).eigenvalue, 0.0);
    EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(1).eigenvalue, -10.0);
    EXPECT_DOUBLE_EQ(system.get_quantum_state_by_index(2).eigenvalue, 0.0);
}
