#pragma once

/**
 * Checkpoint — root abstract interface for all checkpoint I/O.
 *
 * Marker base for the checkpoint hierarchy. Each domain layer
 * (quantum_system, charge_density, wavefunction) extends this
 * with its own snapshot type and save/load signatures.
 */

namespace q3 {

class Checkpoint {
  public:
    virtual ~Checkpoint() = default;
};

} // namespace q3
