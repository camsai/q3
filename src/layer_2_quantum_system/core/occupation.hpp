#pragma once

/**
 * Electronic occupation configuration.
 *
 * Basic quantum state information about how electrons fill states.
 * This describes the occupation scheme, not the computed occupation
 * numbers themselves (those live in calculated data holders).
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Electronic occupation configuration for a quantum system.
struct Occupation {
    /// Total number of electrons.
    Real number_of_electrons = 0.0;

    /// Spin degeneracy factor (2.0 for unpolarized, 1.0 for spin-polarized).
    Real spin_degeneracy = 2.0;

    /// Number of occupied bands per spin channel.
    int occupied_bands_per_spin = 0;

    /// Whether smearing is active (fractional occupations).
    bool is_smeared = false;
};

} // namespace q3
