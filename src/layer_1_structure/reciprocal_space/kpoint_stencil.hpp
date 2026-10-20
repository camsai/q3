#pragma once

/**
 * KPointStencil — the finite-difference stencil on a k-mesh: for every
 * k-point, which k-point each neighbour direction lands on, and for every
 * direction its b-vector and weight.
 *
 * Everything that sums over the neighbours of a k-point -- a subspace
 * iteration, a spread, its gradient -- reads this. Resolved here once, from
 * the grid and the shells, it is the one object those consumers share and the
 * one place a malformed table is refused.
 *
 * It is built one of two ways, each by name. from_grid() resolves k + b on a
 * full grid from its neighbour shells. from_table() takes the neighbours as
 * given, and carries no b-vectors unless given them: the gauge-invariant
 * spread needs only weights and neighbours, and a ring of k-points has none
 * to give. Anything that reads a phase -- the spread's centres, its gradient
 * -- asks has_b_vectors() first. The two are named rather than overloaded
 * because both take two braced lists, and a call written with empty braces
 * fits either.
 */

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"

namespace q3 {

class KPointStencil {
  public:
    /// From a full grid and its neighbour shells: k + b resolved for every k and every direction.
    [[nodiscard]] static KPointStencil
    from_grid(const std::array<int, 3>& grid_dimensions,
              const std::vector<KPointNeighborDirection>& directions) {
        return KPointStencil(grid_dimensions, directions);
    }

    /**
     * From an explicit table -- one row of neighbour indices per k-point, one
     * entry per direction -- and the weights; b-vectors may be omitted.
     */
    [[nodiscard]] static KPointStencil from_table(std::vector<std::vector<int>> neighbor_table,
                                                  std::vector<Real> weights,
                                                  std::vector<Vector3D> b_vectors = {}) {
        return KPointStencil(std::move(neighbor_table), std::move(weights), std::move(b_vectors));
    }

    [[nodiscard]] int number_of_kpoints() const {
        return static_cast<int>(neighbor_table_.size());
    }
    [[nodiscard]] int number_of_directions() const {
        return static_cast<int>(weights_.size());
    }

    /// The k-point that k + b lands on.
    [[nodiscard]] int neighbor(int kpoint, int direction) const {
        return neighbor_table_[static_cast<std::size_t>(kpoint)]
                              [static_cast<std::size_t>(direction)];
    }

    [[nodiscard]] Real weight(int direction) const {
        return weights_[static_cast<std::size_t>(direction)];
    }
    [[nodiscard]] const std::vector<Real>& weights() const {
        return weights_;
    }
    [[nodiscard]] Real total_weight() const {
        Real total = 0.0;
        for (const Real weight : weights_) {
            total += weight;
        }
        return total;
    }

    [[nodiscard]] bool has_b_vectors() const {
        return !b_vectors_.empty();
    }
    /// The b-vector of a direction, in whatever units the shells were solved in.
    [[nodiscard]] const Vector3D& b_vector(int direction) const {
        require_b_vectors();
        return b_vectors_[static_cast<std::size_t>(direction)];
    }
    [[nodiscard]] const std::vector<Vector3D>& b_vectors() const {
        require_b_vectors();
        return b_vectors_;
    }

    /// Whether the stencil was built from a grid, so its directions carry integer steps.
    [[nodiscard]] bool has_grid_steps() const {
        return !grid_steps_.empty();
    }
    [[nodiscard]] const IntegerVector3D& grid_step(int direction) const {
        if (!has_grid_steps()) {
            throw std::logic_error("KPointStencil: a stencil built from a table carries no grid "
                                   "steps");
        }
        return grid_steps_[static_cast<std::size_t>(direction)];
    }
    [[nodiscard]] const std::array<int, 3>& grid_dimensions() const {
        return grid_dimensions_;
    }

    /**
     * The first direction whose step reaches the same k-point as its
     * opposite, if there is one. A Wannier centre is a finite-difference Berry phase,
     * and it needs b and -b to be different k-points. On an axis with two grid
     * points they are the SAME point: k+b and k-b coincide, so the two
     * directions carry one shared overlap block and two opposite b-vectors,
     * and the terms [Im ln M_nn + b . r_n]^2 they contribute cannot both be
     * small whatever r_n is. Omega_D is then bounded away from zero by the
     * mesh rather than by the physics, and the centres along that axis are
     * noise. Only a stencil built from a grid knows its steps; one built from
     * a table has no answer to give.
     */
    [[nodiscard]] std::optional<int> two_point_axis_direction() const {
        for (std::size_t direction = 0; direction < grid_steps_.size(); direction++) {
            const IntegerVector3D& step = grid_steps_[direction];
            const IntegerVector3D opposite = {-step.x, -step.y, -step.z};
            if (KPointNeighborShells::neighbor_kpoint_index(0, step, grid_dimensions_) ==
                KPointNeighborShells::neighbor_kpoint_index(0, opposite, grid_dimensions_)) {
                return static_cast<int>(direction);
            }
        }
        return std::nullopt;
    }

  private:
    KPointStencil(const std::array<int, 3>& grid_dimensions,
                  const std::vector<KPointNeighborDirection>& directions)
        : grid_dimensions_(grid_dimensions) {
        for (int axis = 0; axis < 3; axis++) {
            if (grid_dimensions[axis] <= 0) {
                throw std::invalid_argument(
                    "KPointStencil: the grid dimensions must be positive, got " +
                    std::to_string(grid_dimensions[0]) + "x" + std::to_string(grid_dimensions[1]) +
                    "x" + std::to_string(grid_dimensions[2]));
            }
        }
        for (const KPointNeighborDirection& direction : directions) {
            weights_.push_back(direction.weight);
            b_vectors_.push_back(direction.reciprocal_lattice_translation);
            grid_steps_.push_back(direction.grid_step);
        }
        const int number_of_kpoints = grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2];
        neighbor_table_.reserve(static_cast<std::size_t>(number_of_kpoints));
        for (int kpoint = 0; kpoint < number_of_kpoints; kpoint++) {
            std::vector<int> row;
            row.reserve(directions.size());
            for (const KPointNeighborDirection& direction : directions) {
                row.push_back(KPointNeighborShells::neighbor_kpoint_index(
                    kpoint, direction.grid_step, grid_dimensions));
            }
            neighbor_table_.push_back(std::move(row));
        }
        require_consistent();
    }

    KPointStencil(std::vector<std::vector<int>> neighbor_table,
                  std::vector<Real> weights,
                  std::vector<Vector3D> b_vectors)
        : neighbor_table_(std::move(neighbor_table))
        , weights_(std::move(weights))
        , b_vectors_(std::move(b_vectors)) {
        require_consistent();
    }

    void require_consistent() const {
        if (neighbor_table_.empty()) {
            throw std::invalid_argument("KPointStencil: no k-points");
        }
        if (weights_.empty()) {
            throw std::invalid_argument(
                "KPointStencil: no neighbour directions; there is nothing to make a subspace "
                "smooth against and no phase to read");
        }
        if (!b_vectors_.empty() && b_vectors_.size() != weights_.size()) {
            throw std::invalid_argument("KPointStencil: one b-vector per direction or none; " +
                                        std::to_string(b_vectors_.size()) + " were given for " +
                                        std::to_string(weights_.size()) + " directions");
        }
        const int number_of_kpoints = static_cast<int>(neighbor_table_.size());
        for (std::size_t kpoint = 0; kpoint < neighbor_table_.size(); kpoint++) {
            if (neighbor_table_[kpoint].size() != weights_.size()) {
                throw std::invalid_argument("KPointStencil: k-point " + std::to_string(kpoint) +
                                            " does not carry one neighbour per direction");
            }
            for (const int neighbor : neighbor_table_[kpoint]) {
                if (neighbor < 0 || neighbor >= number_of_kpoints) {
                    throw std::invalid_argument(
                        "KPointStencil: k-point " + std::to_string(kpoint) + " names k-point " +
                        std::to_string(neighbor) + " as a neighbour, which is outside the " +
                        std::to_string(number_of_kpoints) + " k-points given");
                }
            }
        }
    }

    void require_b_vectors() const {
        if (b_vectors_.empty()) {
            throw std::logic_error("KPointStencil: this stencil carries no b-vectors, so nothing "
                                   "that reads a phase can run on it");
        }
    }

    std::vector<std::vector<int>> neighbor_table_;
    std::vector<Real> weights_;
    std::vector<Vector3D> b_vectors_;
    std::vector<IntegerVector3D> grid_steps_;
    std::array<int, 3> grid_dimensions_ = {0, 0, 0};
};

} // namespace q3
