#pragma once

/**
 * PseudopotentialForceContributions — per-contribution force vectors
 * returned by the pseudopotential force calculator.
 *
 * Separates local, nonlocal, core-correction, and augmentation
 * (ultrasoft) contributions for analysis and debugging.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

struct PseudopotentialForceContributions {
    std::vector<Vector3D> local;
    std::vector<Vector3D> nonlocal;
    std::vector<Vector3D> core_correction;
    std::vector<Vector3D> augmentation;
    std::vector<Vector3D> hubbard;
};

} // namespace q3
