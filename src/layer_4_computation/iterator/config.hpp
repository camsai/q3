#pragma once

/**
 * SCFIteratorConfig — configuration for the SCF iteration loop.
 *
 * Holds convergence criteria, iteration limits, and a reference to the
 * crystal structure.
 */

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"

namespace q3 {

struct SCFIteratorConfig {
    Crystal* crystal_structure = nullptr;
    int number_of_spin_components = 1;
    Real convergence_threshold = 1e-6; // Ry
    int maximum_iterations = 100;
};

} // namespace q3
