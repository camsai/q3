#pragma once

/**
 * Reciprocal space vectors grid.
 *
 * A grid of vectors in reciprocal space, inheriting from PointsGrid.
 * This is the generic reciprocal-space analog — specialized by
 * GVectorsGrid in layer 3 for plane-wave basis sets.
 */

#include <vector>

#include "layer_0_core/points_grid.hpp"
#include "layer_1_structure/reciprocal_space/reciprocal_space_vector.hpp"

namespace q3 {

/// A grid of reciprocal-space vectors.
class ReciprocalSpaceVectorsGrid : public PointsGrid {
  public:
    ReciprocalSpaceVectorsGrid() = default;

    [[nodiscard]] int number_of_vectors() const {
        return static_cast<int>(vectors_.size());
    }

    [[nodiscard]] const std::vector<ReciprocalSpaceVector>& vectors() const {
        return vectors_;
    }

    [[nodiscard]] const ReciprocalSpaceVector& vector(int index) const {
        return vectors_[index];
    }

  protected:
    std::vector<ReciprocalSpaceVector> vectors_;
};

} // namespace q3
