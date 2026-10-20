#pragma once

/**
 * AtomicForcesContributions — per-contribution force breakdown.
 *
 * Plain data holder for the diagnostic breakdown of Hellmann-Feynman forces
 * by physical mechanism.  Returned alongside the forces vector from
 * ForceCalculator::compute_forces().
 *
 * This type has no JSON serialization dependency.  If a diagnostic dump is
 * needed, the Python layer reads the flat values from results.json and formats
 * them as needed.
 */

#include <vector>

#include "layer_0_core/types/vector_3d.hpp"

namespace q3 {

/// Per-contribution force breakdown (Ry/bohr).
struct AtomicForcesContributions {
    std::vector<Vector3D> local;           // local pseudopotential
    std::vector<Vector3D> ewald;           // Ewald ionic
    std::vector<Vector3D> nonlocal;        // nonlocal KB projector
    std::vector<Vector3D> core_correction; // nonlinear core correction (NLCC)
    std::vector<Vector3D> augmentation;    // ultrasoft augmentation (USPP)
    std::vector<Vector3D> dispersion;      // DFT-D3(BJ) semi-empirical dispersion
    std::vector<Vector3D> hubbard;         // DFT+U (Dudarev) Hellmann-Feynman
};

} // namespace q3
