#pragma once

/**
 * AtomSpecies: describes an elemental species used in a calculation.
 *
 * Holds the element symbol, atomic mass, valence charge from the
 * pseudopotential, and the pseudopotential filename.
 */

#include <string>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Describes one atomic species (type).
 */
struct AtomSpecies {
    std::string symbol;               // e.g. "Si"
    Real mass;                        // atomic mass (amu)
    Real valence_charge;              // valence charge from pseudopotential
    std::string pseudopotential_file; // pseudopotential filename
};

} // namespace q3
