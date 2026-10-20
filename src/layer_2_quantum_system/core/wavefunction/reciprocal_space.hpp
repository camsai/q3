#pragma once

/**
 * Reciprocal-space wavefunction.
 *
 * A wavefunction defined as a function of a reciprocal-space vector.
 * ψ(G): reciprocal space → C
 */

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/reciprocal_space_vector.hpp"
#include "layer_2_quantum_system/core/wavefunction/init.hpp"

namespace q3 {

/// A wavefunction in reciprocal space: ψ(G).
class ReciprocalSpaceWavefunction : public Wavefunction {
  public:
    ~ReciprocalSpaceWavefunction() override = default;

    /// Evaluate the wavefunction at a reciprocal-space vector.
    [[nodiscard]] virtual Complex evaluate(const ReciprocalSpaceVector& g_vector) const = 0;
};

} // namespace q3
