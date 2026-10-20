#pragma once

/**
 * KohnShamQuantumSystemCheckpoint — Kohn-Sham level quantum system checkpoint.
 *
 * Provides the typed save/load interface for persisting SCF state
 * using the KohnShamQuantumSystemSnapshot DTO.
 */

#include <string>

#include "io/quantum_system/init.hpp"
#include "io/quantum_system/kohn_sham/snapshot.hpp"

namespace q3 {

class KohnShamQuantumSystemCheckpoint : public QuantumSystemCheckpoint {
  public:
    ~KohnShamQuantumSystemCheckpoint() override = default;

    /**
     * Save a quantum system snapshot to file.
     */
    virtual bool save(const std::string& filename,
                      const KohnShamQuantumSystemSnapshot& snapshot,
                      bool save_wavefunctions = false) const = 0;

    /**
     * Load a quantum system snapshot from file.
     */
    virtual bool load(const std::string& filename,
                      KohnShamQuantumSystemSnapshot& snapshot) const = 0;
};

} // namespace q3
