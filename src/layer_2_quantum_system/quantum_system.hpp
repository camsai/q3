#pragma once

/**
 * QuantumSystem — generic quantum system.
 *
 * A collection of quantum states defined on a crystal structure,
 * templated on the state type. Concrete Kohn-Sham and
 * representation-specific systems extend this at layer 3.
 *
 * Hierarchy:
 *   QuantumSystem<QuantumStateType> (L2: crystal, state storage)
 *     └── KohnShamQuantumSystem<QuantumStateType> (L3: + spin)
 *           └── PlanewaveKohnShamQuantumSystem (L3: + kpoints, basis)
 *                 └── NormConservingPlanewaveKohnShamQuantumSystem (L3: + PP basis)
 *                       └── UltrasoftPlanewaveKohnShamQuantumSystem (L3: +
 * augmentation)
 */

#include <vector>

#include "layer_2_quantum_system/core/quantum_state.hpp"

namespace q3 {

class Crystal;

/**
 * A quantum system containing a collection of quantum states
 * defined on a crystal structure.
 *
 * Templated on the state type, which itself is templated on the
 * wavefunction representation. This base class owns the crystal
 * pointer, the state storage, and provides basic accessors.
 */
template <typename QuantumStateType> class QuantumSystem {
  public:
    virtual ~QuantumSystem() = default;

    Crystal* crystal_structure{nullptr};

    const QuantumSystem<QuantumStateType>* preloaded_quantum_system_{nullptr};

    /**
     * Set the preloaded quantum system pointer.
     */
    void load_quantum_system(const QuantumSystem<QuantumStateType>& system) {
        preloaded_quantum_system_ = &system;
    }

    /**
     * Copy all electronic state data from another QuantumSystem.
     */
    virtual void copy_quantum_states_from(const QuantumSystem<QuantumStateType>& source) {
        quantum_states_ = source.quantum_states_;
        crystal_structure = source.crystal_structure;
    }

    /// Number of quantum states in the system.
    [[nodiscard]] int number_of_quantum_states() const {
        return static_cast<int>(quantum_states_.size());
    }

    /// Access a quantum state by flat index.
    [[nodiscard]] const QuantumStateType& get_quantum_state_by_index(int index) const {
        return quantum_states_[index];
    }
    QuantumStateType& get_quantum_state_by_index(int index) {
        return quantum_states_[index];
    }

  protected:
    std::vector<QuantumStateType> quantum_states_;
};

} // namespace q3
