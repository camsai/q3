#pragma once

/**
 * KPointsGrid: Monkhorst-Pack k-point mesh generator.
 *
 * Generates a uniform k-point grid with configurable dimensions, offsets,
 * and weights. Supports adding individual k-points for irreducible Brillouin
 * zone reduction.
 *
 * Implements the uniform meshes of H. J. Monkhorst & J. D. Pack, Phys. Rev. B
 * 13, 5188 (1976).
 */

#include <cmath>
#include <string>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint.hpp"

namespace q3 {

/**
 * Generates and stores Monkhorst-Pack k-point meshes.
 */
class KPointsGrid {
  public:
    KPointsGrid() = default;

    /**
     * Generate Monkhorst-Pack mesh with dimensions and offsets.
     * Offsets are 0 or 1 (as in QE input).
     */
    void generate(int grid_dimension_x,
                  int grid_dimension_y,
                  int grid_dimension_z,
                  int shift_x,
                  int shift_y,
                  int shift_z) {
        grid_dimensions_ = {grid_dimension_x, grid_dimension_y, grid_dimension_z};
        grid_shifts_ = {shift_x, shift_y, shift_z};
        kpoints_.clear();
        int total_grid_points = grid_dimension_x * grid_dimension_y * grid_dimension_z;
        Real weight = 2.0 / total_grid_points; // weight per k-point (factor 2 for spin degeneracy)

        for (int x_index = 0; x_index < grid_dimension_x; x_index++) {
            for (int y_index = 0; y_index < grid_dimension_y; y_index++) {
                for (int z_index = 0; z_index < grid_dimension_z; z_index++) {
                    Vector3D coordinates;
                    coordinates.x = static_cast<Real>(x_index) / grid_dimension_x +
                                    (shift_x ? 0.5 : 0.0) / grid_dimension_x;
                    coordinates.y = static_cast<Real>(y_index) / grid_dimension_y +
                                    (shift_y ? 0.5 : 0.0) / grid_dimension_y;
                    coordinates.z = static_cast<Real>(z_index) / grid_dimension_z +
                                    (shift_z ? 0.5 : 0.0) / grid_dimension_z;

                    // Fold back to [-0.5, 0.5)
                    coordinates.x -= std::floor(coordinates.x + 0.5);
                    coordinates.y -= std::floor(coordinates.y + 0.5);
                    coordinates.z -= std::floor(coordinates.z + 0.5);

                    KPoint kpoint;
                    kpoint.coordinates = coordinates;
                    kpoint.weight = weight;
                    kpoints_.push_back(kpoint);
                }
            }
        }
    }

    [[nodiscard]] int number_of_kpoints() const {
        return static_cast<int>(kpoints_.size());
    }
    [[nodiscard]] const std::vector<KPoint>& kpoints() const {
        return kpoints_;
    }
    [[nodiscard]] const KPoint& operator[](int index) const {
        return kpoints_[index];
    }

    /**
     * Add a single k-point (used by IBZ reduction).
     */
    void add_kpoint(const KPoint& kpoint) {
        kpoints_.push_back(kpoint);
    }

    /**
     * Scale every k-point weight by a constant factor.
     *
     * Generated weights embed the spin degeneracy of 2 (they sum to 2 over the
     * grid). Spin-polarized (LSDA) calculations enumerate the two spin channels
     * as explicit k-point states, so each state carries half of that weight:
     * scale by 0.5 once, right after the grid (and any IBZ reduction) is built.
     */
    void scale_weights(Real factor) {
        for (auto& kpoint : kpoints_) {
            kpoint.weight *= factor;
        }
    }

    /**
     * Access the grid dimensions (x, y, z).
     */
    [[nodiscard]] const std::array<int, 3>& grid_dimensions() const {
        return grid_dimensions_;
    }

    /**
     * Access the grid shifts (x, y, z).
     */
    [[nodiscard]] const std::array<int, 3>& grid_shifts() const {
        return grid_shifts_;
    }

    /**
     * Log the k-point grid contents with coordinates and weights.
     */
    void pretty_print() const {
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints(); kpoint_index++) {
            const auto& kpoint = kpoints_[kpoint_index];
            log::info("       k({}) = ({:.7f}, {:.7f}, {:.7f}), weight = {:.7f}",
                      kpoint_index + 1,
                      kpoint.coordinates.x,
                      kpoint.coordinates.y,
                      kpoint.coordinates.z,
                      kpoint.weight);
        }
    }

    /**
     * Set the Monkhorst-Pack grid metadata (dimensions and shifts).
     *
     * Used by symmetry reduction and k-point distribution to preserve the
     * requested generated-grid parameters. Explicit k-point lists retain the
     * default zero dimensions and remain distinguishable from generated grids.
     */
    void set_grid_metadata(const std::array<int, 3>& dimensions, const std::array<int, 3>& shifts) {
        grid_dimensions_ = dimensions;
        grid_shifts_ = shifts;
    }

  private:
    std::vector<KPoint> kpoints_;
    std::array<int, 3> grid_dimensions_ = {0, 0, 0};
    std::array<int, 3> grid_shifts_ = {0, 0, 0};
};

} // namespace q3
