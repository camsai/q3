#pragma once

/**
 * UniformQGrid — metadata for a uniform q = |G| interpolation grid.
 *
 * Used by Fourier-transformed radial functions tabulated on a uniform q-grid
 * (beta projectors, core charge, augmentation Q functions, etc.).
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

struct UniformQGrid {
    Real q_spacing = 0.0;
    int number_of_q_points = 0;

    /**
     * Build a grid covering [0, maximum_q_value] with the given spacing.
     */
    static UniformQGrid from_maximum_q(Real maximum_q_value, Real q_spacing) {
        UniformQGrid grid;
        grid.q_spacing = q_spacing;
        grid.number_of_q_points = static_cast<int>(maximum_q_value / q_spacing) + 4;
        return grid;
    }

    /**
     * q value at a grid index.
     */
    [[nodiscard]] Real q_at_index(int index) const {
        return index * q_spacing;
    }

    /**
     * All q values on the grid.
     */
    [[nodiscard]] std::vector<Real> q_point_values() const {
        std::vector<Real> values(number_of_q_points);
        for (int index = 0; index < number_of_q_points; index++) {
            values[index] = q_at_index(index);
        }
        return values;
    }
};

} // namespace q3
