#pragma once

/**
 * Generic grid of 3D points with weights.
 *
 * Base abstraction for any set of weighted 3D points (k-points, q-points,
 * angular quadrature points, etc.). Provides storage and iteration over
 * point/weight pairs without encoding any physics.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * A point in 3D space.
 */
struct [[nodiscard]] Point3D {
    Vector3D coordinates;
};

/**
 * A point in 3D space with an associated integration weight.
 */
struct [[nodiscard]] WeightedPoint3D : Point3D {
    Real weight{1.0};
};

/**
 * Generic grid of weighted 3D points.
 *
 * Subclasses (e.g., KPointsGrid) extend this with domain-specific generation
 * logic while inheriting the common storage and accessors.
 */
class PointsGrid {
  public:
    PointsGrid() = default;
    virtual ~PointsGrid() = default;

    [[nodiscard]] int number_of_points() const {
        return static_cast<int>(points_.size());
    }

    [[nodiscard]] const std::vector<WeightedPoint3D>& points() const {
        return points_;
    }

    [[nodiscard]] const WeightedPoint3D& point(int index) const {
        return points_[index];
    }

    [[nodiscard]] bool empty() const {
        return points_.empty();
    }

  protected:
    std::vector<WeightedPoint3D> points_;
};

} // namespace q3
