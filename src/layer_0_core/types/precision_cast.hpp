#pragma once

/**
 * Precision cast utilities for FP64 ↔ FP32 complex number conversion.
 *
 * Provides CPU-side element-wise conversion between Complex (std::complex<double>)
 * and ComplexSingle (std::complex<float>) arrays. GPU implementations may override
 * these with CUDA kernels for device-resident data.
 *
 * These utilities are used by the mixed-precision Davidson loop to convert
 * wavefunctions between FP64 (for precision-sensitive operations) and FP32
 * (for bulk H|ψ⟩ computation on GPU).
 */

#include <complex>

#include "layer_0_core/types/single_precision.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

namespace precision {

/**
 * Cast an array of double-precision complex values to single-precision.
 *
 * Element-wise narrowing conversion: each Complex element is converted to
 * ComplexSingle by truncating the mantissa from 52 to 23 bits.
 *
 * @param count        Number of elements to convert.
 * @param source       Input array of Complex (FP64) values.
 * @param destination  Output array of ComplexSingle (FP32) values.
 *                     Must be pre-allocated with at least count elements.
 */
inline void cast_double_to_single(int count, const Complex* source, ComplexSingle* destination) {
    for (int index = 0; index < count; ++index) {
        destination[index] = ComplexSingle(static_cast<float>(source[index].real()),
                                           static_cast<float>(source[index].imag()));
    }
}

/**
 * Cast an array of single-precision complex values to double-precision.
 *
 * Element-wise widening conversion: each ComplexSingle element is converted
 * to Complex by zero-extending the mantissa.
 *
 * @param count        Number of elements to convert.
 * @param source       Input array of ComplexSingle (FP32) values.
 * @param destination  Output array of Complex (FP64) values.
 *                     Must be pre-allocated with at least count elements.
 */
inline void cast_single_to_double(int count, const ComplexSingle* source, Complex* destination) {
    for (int index = 0; index < count; ++index) {
        destination[index] = Complex(static_cast<double>(source[index].real()),
                                     static_cast<double>(source[index].imag()));
    }
}

/**
 * Cast an array of double-precision real values to single-precision.
 *
 * @param count        Number of elements to convert.
 * @param source       Input array of Real (FP64) values.
 * @param destination  Output array of RealSingle (FP32) values.
 */
inline void cast_real_double_to_single(int count, const Real* source, RealSingle* destination) {
    for (int index = 0; index < count; ++index) {
        destination[index] = static_cast<float>(source[index]);
    }
}

/**
 * Cast an array of single-precision real values to double-precision.
 *
 * @param count        Number of elements to convert.
 * @param source       Input array of RealSingle (FP32) values.
 * @param destination  Output array of Real (FP64) values.
 */
inline void cast_real_single_to_double(int count, const RealSingle* source, Real* destination) {
    for (int index = 0; index < count; ++index) {
        destination[index] = static_cast<double>(source[index]);
    }
}

} // namespace precision
} // namespace q3
