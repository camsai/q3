#pragma once

/**
 * UniformQGridTable — function values on a uniform q-grid with Lagrange interpolation.
 */

#include <vector>

#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

class UniformQGridTable {
  public:
    UniformQGridTable() = default;

    void resize(const UniformQGrid& grid) {
        grid_ = grid;
        values_.assign(grid.number_of_q_points, 0.0);
    }

    void assign(const UniformQGrid& grid, Real fill_value = 0.0) {
        grid_ = grid;
        values_.assign(grid.number_of_q_points, fill_value);
    }

    [[nodiscard]] const UniformQGrid& grid() const {
        return grid_;
    }

    [[nodiscard]] int size() const {
        return static_cast<int>(values_.size());
    }

    [[nodiscard]] const std::vector<Real>& values() const {
        return values_;
    }

    [[nodiscard]] std::vector<Real>& values() {
        return values_;
    }

    [[nodiscard]] Real operator[](int index) const {
        return values_[index];
    }

    [[nodiscard]] Real& operator[](int index) {
        return values_[index];
    }

    [[nodiscard]] Real interpolate(Real q_value) const {
        if (values_.empty()) {
            return 0.0;
        }
        if (q_value < math::tolerances::zero_vector_magnitude_threshold) {
            return values_[0];
        }
        return lagrange_interpolation_4point(
            values_, q_value, grid_.q_spacing, grid_.number_of_q_points);
    }

    [[nodiscard]] Real interpolate_derivative(Real q_value) const {
        if (values_.empty()) {
            return 0.0;
        }
        return lagrange_interpolation_4point_derivative(
            values_, q_value, grid_.q_spacing, grid_.number_of_q_points);
    }

  private:
    UniformQGrid grid_;
    std::vector<Real> values_;
};

} // namespace q3
