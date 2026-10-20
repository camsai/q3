#pragma once

/**
 * Atomic wavefunction from a pseudopotential.
 *
 * The pseudo-atomic wavefunctions χ_i(r) used for initial
 * wavefunction construction and charge density generation.
 */

#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Atomic wavefunction from a pseudopotential.
struct WavefunctionAtomic {
    std::string label;
    int angular_momentum;
    Real occupations;
    std::vector<Real> radial_values; // χ(r)
};

} // namespace q3
