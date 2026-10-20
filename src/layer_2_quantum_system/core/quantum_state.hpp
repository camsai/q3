#pragma once

/**
 * QuantumState — generic quantum state.
 *
 * A single quantum state consisting of a wavefunction,
 * its energy eigenvalue, and its occupation number.
 * Templated on the wavefunction type, which is
 * representation-specific (defined in layer 3).
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * A single quantum state: wavefunction + eigenvalue + occupation.
 */
template <typename WavefunctionType> struct QuantumState {
    /// The wavefunction (representation-specific).
    WavefunctionType wavefunction;

    /// Energy eigenvalue in Rydberg.
    Real eigenvalue = 0.0;

    /// Occupation number (0.0 to 1.0 per spin channel).
    Real occupation = 0.0;

    /// Whether this state is occupied.
    [[nodiscard]] bool is_occupied() const {
        return occupation > 0.0;
    }
};

} // namespace q3
