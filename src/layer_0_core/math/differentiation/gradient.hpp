#pragma once

/**
 * Numerical gradient on a non-uniform grid.
 *
 * At each interior node the derivative is that of the unique quadratic
 * through the node and its two neighbours, which on a non-uniform grid is
 * second-order accurate where a plain centered difference is only first.
 *
 * Endpoints are conventions, not consequences of that formula, and the caller
 * has to accept both. The last node's derivative is set to zero: this is
 * written for functions that have already decayed by the outer mesh point --
 * a pseudo-core charge is the case in hand -- where a one-sided stencil on
 * the sparsest part of a logarithmic grid would differentiate noise. The
 * first node's derivative is extrapolated from the next two rather than
 * computed, because the 3-point stencil has no centre there; a one-sided
 * formula on the first three nodes is the standard alternative and is the
 * re-derivation to prefer if this is ever revisited.
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Compute gradient on a non-uniform grid using a 3-point asymmetric formula.
 *
 * @param function_values  Function values at grid points
 * @param gradient         Output gradient array
 * @param grid             Grid point positions
 * @param mesh_size        Number of grid points
 */
inline void
gradient_nonuniform(const Real* function_values, Real* gradient, const Real* grid, int mesh_size) {
    for (int grid_index = 1; grid_index < mesh_size - 1; grid_index++) {
        Real delta_r_forward = grid[grid_index + 1] - grid[grid_index];
        Real delta_r_backward = grid[grid_index - 1] - grid[grid_index];
        Real delta_f_forward = function_values[grid_index + 1] - function_values[grid_index];
        Real delta_f_backward = function_values[grid_index - 1] - function_values[grid_index];
        gradient[grid_index] =
            (delta_r_forward * delta_r_forward * delta_f_backward -
             delta_r_backward * delta_r_backward * delta_f_forward) /
            (delta_r_forward * delta_r_backward * (grid[grid_index + 1] - grid[grid_index - 1]));
    }
    gradient[mesh_size - 1] = 0.0;
    // Extrapolate gradient[0] from gradient[1] and gradient[2]
    if (mesh_size >= 3) {
        gradient[0] =
            gradient[1] + (gradient[2] - gradient[1]) * (grid[0] - grid[1]) / (grid[2] - grid[1]);
    } else {
        gradient[0] = gradient[1];
    }
}

} // namespace q3
