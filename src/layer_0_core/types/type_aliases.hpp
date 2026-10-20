#pragma once

/**
 * Fundamental scalar type aliases for the Q3 codebase.
 *
 * Defines `Real` (double-precision floating point) and `Complex`
 * (std::complex<double>) as the canonical numeric types used throughout
 * all computational layers.
 */

#include <complex>

namespace q3 {

// ---------------------------------------------------------------------------

// Fundamental scalar types

// ---------------------------------------------------------------------------
using Real = double;
using Complex = std::complex<double>;

} // namespace q3
