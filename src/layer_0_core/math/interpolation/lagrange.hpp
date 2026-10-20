#pragma once

/**
 * 4-point Lagrange interpolation on a uniform grid.
 *
 * Provides a reusable utility for interpolating tabulated functions
 * (beta functions, atomic charge density, core charge, Q-functions, etc.)
 * that are stored on a uniform q-grid with spacing dq.
 *
 * This replaces the repeated inline interpolation pattern:
 *   px = q/dq - int(q/dq)
 *   ux = 1-px; vx = 2-px; wx = 3-px
 *   f(q) ≈ f[i0]*ux*vx*wx/6 + f[i0+1]*px*vx*wx/2 - f[i0+2]*px*ux*wx/2 + f[i0+3]*px*ux*vx/6
 */

#include <vector>

#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

/**
 * Interpolate a function tabulated on a uniform grid using 4-point Lagrange interpolation.
 *
 * @param table       Tabulated function values at uniform q-grid points.
 * @param q_value     The q-point at which to interpolate.
 * @param q_spacing   The uniform grid spacing.
 * @param table_size  Number of points in the table.
 * @return            Interpolated value, or 0 if out of range.
 */
inline Real lagrange_interpolation_4point(const std::vector<Real>& table,
                                          Real q_value,
                                          Real q_spacing,
                                          int table_size) {
    Real fractional_index = q_value / q_spacing;
    int base_index = static_cast<int>(fractional_index);
    Real fractional_part = fractional_index - base_index;
    if (base_index + 3 >= table_size)
        return 0.0;
    Real weight_1 = 1.0 - fractional_part;
    Real weight_2 = 2.0 - fractional_part;
    Real weight_3 = 3.0 - fractional_part;
    return table[base_index] * weight_1 * weight_2 * weight_3 / 6.0 +
           table[base_index + 1] * fractional_part * weight_2 * weight_3 / 2.0 -
           table[base_index + 2] * fractional_part * weight_1 * weight_3 / 2.0 +
           table[base_index + 3] * fractional_part * weight_1 * weight_2 / 6.0;
}

/**
 * Compute the derivative of a 4-point Lagrange interpolant on a uniform grid.
 *
 * This is the analytical derivative of lagrange_interpolation_4point with respect to q_value.
 *
 * @param table       Tabulated function values at uniform q-grid points.
 * @param q_value     The q-point at which to evaluate the derivative.
 * @param q_spacing   The uniform grid spacing.
 * @param table_size  Number of points in the table.
 * @return            Derivative value df/dq, or 0 if out of range.
 */
inline Real lagrange_interpolation_4point_derivative(const std::vector<Real>& table,
                                                     Real q_value,
                                                     Real q_spacing,
                                                     int table_size) {
    Real fractional_index = q_value / q_spacing;
    int base_index = static_cast<int>(fractional_index);
    Real fractional_part = fractional_index - base_index;
    if (base_index + 3 >= table_size)
        return 0.0;
    Real weight_1 = 1.0 - fractional_part;
    Real weight_2 = 2.0 - fractional_part;
    Real weight_3 = 3.0 - fractional_part;
    // d/dq of the 4-point Lagrange interpolant, divided by q_spacing for chain rule
    return (table[base_index] * (-weight_2 * weight_3 - weight_1 * weight_3 - weight_1 * weight_2) /
                6.0 +
            table[base_index + 1] *
                (+weight_2 * weight_3 - fractional_part * weight_3 - fractional_part * weight_2) /
                2.0 -
            table[base_index + 2] *
                (+weight_1 * weight_3 - fractional_part * weight_3 - fractional_part * weight_1) /
                2.0 +
            table[base_index + 3] *
                (+weight_1 * weight_2 - fractional_part * weight_2 - fractional_part * weight_1) /
                6.0) /
           q_spacing;
}

} // namespace q3
