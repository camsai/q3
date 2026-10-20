#pragma once

/**
 * DisentanglementSettings — the damping, the tolerance and the budget of the
 * subspace iteration (SubspaceDisentangler).
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

struct [[nodiscard]] DisentanglementSettings {
    /// Mixing of Z between iterations; 1 is undamped and oscillates on metals.
    Real damping = 0.5;
    /// Relative, because an absolute spread threshold does not transfer between systems.
    Real convergence_tolerance = 1.0e-10;
    int maximum_iterations = 4000;
    /// Iterations the relative change must stay below the tolerance for.
    int convergence_window = 5;
};

} // namespace q3
