#pragma once

/**
 * Simpson's rule numerical integration on a non-uniform grid.
 */

#include <algorithm>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Simpson's rule integration on a grid: ∫ f(r) dr
 * where dr is given by integration_factors[i] = dr/di.
 *
 * @param function_values             Function values at grid points
 * @param integration_factors         Integration weight factors (dr/di)
 * @param number_of_grid_points       Number of grid points to integrate over
 * @return Integrated value
 */
inline Real simpson_rule(const std::vector<Real>& function_values,
                         const std::vector<Real>& integration_factors,
                         int number_of_grid_points) {
    Real integrated_sum = 0.0;
    int mesh_size = std::min(number_of_grid_points, static_cast<int>(function_values.size()));
    if (mesh_size < 3) {
        return 0.0;
    }

    for (int grid_index = 1; grid_index < mesh_size - 1; grid_index++) {
        Real integration_weight = (grid_index % 2 == 1) ? 4.0 : 2.0;
        integrated_sum +=
            integration_weight * function_values[grid_index] * integration_factors[grid_index];
    }

    if (mesh_size % 2 == 1) {
        integrated_sum += function_values[0] * integration_factors[0] +
                          function_values[mesh_size - 1] * integration_factors[mesh_size - 1];
    } else {
        integrated_sum +=
            function_values[0] * integration_factors[0] -
            function_values[mesh_size - 3] * integration_factors[mesh_size - 3] * 0.25 +
            function_values[mesh_size - 2] * integration_factors[mesh_size - 2] +
            function_values[mesh_size - 1] * integration_factors[mesh_size - 1] * 1.25;
    }

    return integrated_sum / 3.0;
}

} // namespace q3
