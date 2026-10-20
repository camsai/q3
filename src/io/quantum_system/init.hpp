#pragma once

/**
 * QuantumSystemCheckpoint — abstract base for saving/loading
 * full quantum system state (charge density + eigenvalues + wavefunctions).
 *
 * Extends the generic Checkpoint marker. Domain-specific snapshot types
 * and save/load signatures are defined at each layer below.
 */

#include "io/checkpoint/checkpoint.hpp"

namespace q3 {

class QuantumSystemCheckpoint : public Checkpoint {
  public:
    ~QuantumSystemCheckpoint() override = default;
};

} // namespace q3
