#pragma once

/**
 * Single-precision (FP32) scalar type aliases for mixed-precision GPU computation.
 *
 * Complements the existing FP64 types in type_aliases.hpp:
 *   Real    = double             →  RealSingle    = float
 *   Complex = std::complex<double>  →  ComplexSingle = std::complex<float>
 *
 * These types are used by GPU BLAS wrappers and mixed-precision Davidson
 * iterations where bulk H|ψ⟩ computation runs in FP32 for performance.
 */

#include <complex>

namespace q3 {

// Single-precision scalar types for GPU mixed-precision computation
using RealSingle = float;
using ComplexSingle = std::complex<float>;

} // namespace q3
