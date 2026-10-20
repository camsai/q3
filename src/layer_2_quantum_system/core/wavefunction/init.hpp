#pragma once

/**
 * Abstract wavefunction base.
 *
 * A wavefunction ψ is a complex-valued function that represents
 * a quantum mechanical state. This abstract base defines the
 * interface without specifying the domain (real or reciprocal space)
 * or the representation (grid, basis set, etc.).
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Abstract base for all wavefunction types.
class Wavefunction {
  public:
    virtual ~Wavefunction() = default;

    /// Number of components per wavefunction (always 1 for scalar states).
    [[nodiscard]] virtual int number_of_components() const = 0;
};

} // namespace q3
