#pragma once

/**
 * Planewave Kohn-Sham quantum state type aliases.
 *
 * Specializes QuantumState for the planewave representation hierarchy:
 *   PlanewaveKohnShamQuantumState
 *     └── NormConservingPlanewaveKohnShamQuantumState
 *           └── UltrasoftPlanewaveKohnShamQuantumState
 */

#include "layer_2_quantum_system/core/quantum_state.hpp"
#include "layer_3_representation/wavefunction/planewave/init.hpp"

namespace q3 {

/// A single planewave Kohn-Sham quantum state.
using PlanewaveKohnShamQuantumState = QuantumState<PlanewaveWavefunction>;

/// A single pseudopotential planewave Kohn-Sham quantum state.
using NormConservingPlanewaveKohnShamQuantumState = PlanewaveKohnShamQuantumState;

/// A single ultrasoft pseudopotential planewave Kohn-Sham quantum state.
using UltrasoftPlanewaveKohnShamQuantumState = NormConservingPlanewaveKohnShamQuantumState;

} // namespace q3
