#pragma once

/**
 * FermiEnergyCalculator — abstract base for Fermi energy calculation.
 *
 * Calculators are purely algorithmic: they read eigenvalues and occupations
 * from a quantum system snapshot and update the Fermi energy and occupations.
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Abstract base for Fermi energy calculators.
class FermiEnergyCalculator {
  public:
    virtual ~FermiEnergyCalculator() = default;

    /// Compute occupations and return the Fermi energy.
    virtual Real calculate_occupations() = 0;
};

} // namespace q3
