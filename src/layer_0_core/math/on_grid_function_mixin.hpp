#pragma once

/**
 * On-grid function mixin.
 *
 * Mixin for functions whose values are stored at discrete grid points
 * rather than evaluated analytically. The grid is a PointsGrid
 * (or subclass like GVectorGrid).
 *
 * Templated on the grid type and value type to allow both
 * real-space grids with Real values and reciprocal-space grids
 * with Complex coefficients.
 */

#include <vector>

#include "layer_0_core/points_grid.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Mixin for functions defined on a discrete grid of points.
template <typename GridType, typename ValueType> class OnGridFunctionMixin {
  public:
    /// Access the grid this function is defined on.
    [[nodiscard]] const GridType& grid() const {
        return grid_;
    }

    /// Access all values on the grid.
    [[nodiscard]] const std::vector<ValueType>& values() const {
        return values_;
    }

    /// Access a single value by grid point index.
    [[nodiscard]] const ValueType& value_at_index(int index) const {
        return values_[index];
    }

    ValueType& value_at_index(int index) {
        return values_[index];
    }

    /// Number of grid points.
    [[nodiscard]] int number_of_grid_points() const {
        return static_cast<int>(values_.size());
    }

  protected:
    GridType grid_;
    std::vector<ValueType> values_;
};

} // namespace q3
