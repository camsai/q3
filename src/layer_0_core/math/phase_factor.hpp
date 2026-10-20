#pragma once

/**
 * Phase factor utility functions.
 *
 * Computes complex phase factors e^{±iφ} and angular phase factors (-i)^l.
 * Pure math utilities with no domain-specific dependencies.
 */

#include <cmath>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Compute e^{-iφ} = cos(φ) - i·sin(φ)  (standard structure factor phase).
[[nodiscard]] inline Complex compute_phase_factor(Real phase) {
    return Complex(std::cos(phase), -std::sin(phase));
}

/// Compute e^{+iφ} = cos(φ) + i·sin(φ)  (conjugate structure factor phase).
[[nodiscard]] inline Complex compute_phase_factor_conjugate(Real phase) {
    return Complex(std::cos(phase), std::sin(phase));
}

/**
 * Compute the angular phase factor (-i)^l for a given angular momentum l.
 *
 * Returns the complex power of -i:
 *   l % 4 == 0 →  1
 *   l % 4 == 1 → -i
 *   l % 4 == 2 → -1
 *   l % 4 == 3 →  i
 */
[[nodiscard]] inline Complex compute_angular_phase_factor(int angular_momentum) {
    switch (angular_momentum % 4) {
        case 1:
            return Complex(0.0, -1.0);
        case 2:
            return Complex(-1.0, 0.0);
        case 3:
            return Complex(0.0, 1.0);
        default:
            return Complex(1.0, 0.0);
    }
}

} // namespace q3
