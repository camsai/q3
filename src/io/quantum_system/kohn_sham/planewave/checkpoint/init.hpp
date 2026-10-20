#pragma once

/**
 * PlanewaveKohnShamQuantumSystemCheckpoint — planewave-level base + snapshot.
 *
 * PlanewaveKohnShamQuantumSystemSnapshot extends KohnShamQuantumSystemSnapshot
 * with planewave-specific wavefunction coefficients.
 *
 * PlanewaveKohnShamQuantumSystemCheckpoint inherits save/load from the
 * KS level — format-specific implementations (Q3, HDF5, etc.) override there.
 */

#include <vector>

#include "io/quantum_system/kohn_sham/init.hpp"

namespace q3 {

/**
 * Planewave extension of the KS snapshot.
 *
 * Adds the full wavefunction expansion coefficients in the planewave basis,
 * indexed as [kpoint_state][band][plane_wave].
 */
struct PlanewaveKohnShamQuantumSystemSnapshot : KohnShamQuantumSystemSnapshot {
    std::vector<std::vector<std::vector<Complex>>> wavefunction_coefficients;
};

/**
 * Planewave-level checkpoint base.
 *
 * Inherits save/load interface from KohnShamQuantumSystemCheckpoint.
 * Leaf implementations (Q3, HDF5, etc.) provide the actual I/O.
 */
class PlanewaveKohnShamQuantumSystemCheckpoint : public KohnShamQuantumSystemCheckpoint {
  public:
    ~PlanewaveKohnShamQuantumSystemCheckpoint() override = default;
};

} // namespace q3
