#pragma once

/**
 * KohnShamQuantumSystem — Kohn-Sham specialization of the quantum system.
 *
 * Adds spin configuration to the generic QuantumSystem base.
 * Crystal structure is inherited from QuantumSystem.
 * Still templated on the state type — concrete representations
 * (planewave, pseudopotential) specialize further.
 */

#include "layer_2_quantum_system/quantum_system.hpp"

namespace q3 {

/**
 * Kohn-Sham quantum system: crystal (inherited) + spin + state storage.
 *
 * Inherits crystal and state vector from QuantumSystem<QuantumStateType>.
 */
template <typename QuantumStateType>
class KohnShamQuantumSystem : public QuantumSystem<QuantumStateType> {
  public:
    ~KohnShamQuantumSystem() override = default;

    int number_of_spin_components{1};
};

} // namespace q3
