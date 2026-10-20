#pragma once

/**
 * KPointNeighborShells: b-vector shells and finite-difference weights for a
 * uniform (Monkhorst-Pack) k-point grid.
 *
 * Wannier interpolation and localization need a set of vectors b connecting
 * each k-point to its neighbors on the grid, with weights w_b satisfying
 * the completeness condition (Eq. B1 of Marzari & Vanderbilt, PRB 56, 12847
 * (1997)):
 *
 *   sum_b w_b * b_alpha * b_beta = delta_alpha_beta      alpha, beta in {x,y,z}
 *
 * so that a finite difference of a k-dependent quantity along the b-vectors
 * reconstructs its true gradient. This class finds the smallest set of
 * "shells" (groups of b-vectors of equal length, always closed under
 * b -> -b) that satisfies the condition, growing the set in order of
 * increasing |b| and rejecting any shell whose vectors are linearly
 * dependent on the ones already kept -- so the final set uses the shortest
 * (most local) b-vectors that make the condition solvable.
 *
 * b-vectors are stored as an integer "grid step" (the offset, in
 * Monkhorst-Pack grid-index units, from a k-point to its neighbor) together
 * with the Cartesian translation that step corresponds to. The grid step
 * lets resolve_neighbor() find a specific k-point's neighbor by exact
 * integer arithmetic -- no floating-point ambiguity near the zone boundary.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint.hpp"

namespace q3 {

/**
 * One direction within a neighbor shell: an integer grid step and the
 * Cartesian translation it corresponds to. All directions in the same
 * shell (same |b|) share one weight.
 */
struct [[nodiscard]] KPointNeighborDirection {
    IntegerVector3D grid_step;
    Vector3D reciprocal_lattice_translation;
    Real weight = 0.0;
};

/**
 * The k-point and reciprocal-lattice-vector shift a neighbor direction
 * resolves to, starting from a specific physical k-point on the grid.
 */
struct [[nodiscard]] ResolvedKPointNeighbor {
    int neighbor_kpoint_index = 0;
    /**
     * Integer reciprocal-lattice-vector (Miller-index) shift: the
     * physical k-point at `neighbor_kpoint_index` plus this shift, in
     * reciprocal lattice vector units, equals the starting k-point plus
     * the unwrapped Cartesian translation.
     */
    IntegerVector3D periodic_image_shift;
};

class KPointNeighborShells {
  public:
    KPointNeighborShells() = default;

    /**
     * Find the b-vector shells and weights for a Monkhorst-Pack grid with
     * the given dimensions, in the reciprocal lattice described by
     * `reciprocal_lattice_vectors` (rows = b1, b2, b3, in any consistent
     * Cartesian unit system -- the completeness condition is scale
     * covariant, so the computed weights come out in the inverse square of
     * whatever units the caller passes in).
     *
     * `search_range` bounds the cube of candidate integer grid steps
     * searched for shells (each component ranges over
     * [-search_range, search_range]); `maximum_shell_count` bounds how many
     * shells may be accepted before giving up.
     *
     * Throws std::runtime_error if no set of shells within those bounds
     * satisfies the completeness condition to `completeness_tolerance`, or
     * if a solved weight comes out negative beyond numerical noise -- both
     * signal a lattice/grid combination this search range cannot handle.
     */
    void compute(const Matrix3x3& reciprocal_lattice_vectors,
                 int grid_dimension_x,
                 int grid_dimension_y,
                 int grid_dimension_z,
                 int search_range = 3,
                 int maximum_shell_count = 12,
                 Real shell_relative_tolerance = 1e-6,
                 Real singular_value_tolerance = 1e-10,
                 Real completeness_tolerance = 1e-10) {
        neighbor_directions_.clear();
        number_of_shells_ = 0;

        const std::array<int, 3> grid_dimensions = {
            grid_dimension_x, grid_dimension_y, grid_dimension_z};

        std::vector<Shell> candidate_shells = enumerate_candidate_shells(
            reciprocal_lattice_vectors, grid_dimensions, search_range, shell_relative_tolerance);

        /**
         * 6 independent components of a symmetric 3x3 tensor, in the fixed
         * order xx, yy, zz, xy, xz, yz.
         */
        std::vector<std::array<Real, 6>> accepted_shell_tensors;
        std::vector<const Shell*> accepted_shells;
        const std::array<Real, 6> target = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};

        for (const Shell& candidate : candidate_shells) {
            if (static_cast<int>(accepted_shells.size()) >= maximum_shell_count) {
                break;
            }

            std::array<Real, 6> candidate_tensor = shell_tensor(candidate);
            accepted_shell_tensors.push_back(candidate_tensor);
            accepted_shells.push_back(&candidate);

            LeastSquaresSolution solution =
                solve_least_squares(accepted_shell_tensors, target, singular_value_tolerance);

            if (!solution.smallest_singular_value_cleared_tolerance) {
                /**
                 * This shell added no new independent direction to the
                 * completeness system -- it is redundant with the shells
                 * already accepted. Reject it and keep searching.
                 */
                accepted_shell_tensors.pop_back();
                accepted_shells.pop_back();
                continue;
            }

            if (solution.residual_norm < completeness_tolerance) {
                finalize(accepted_shells, solution.weights);
                return;
            }
        }

        throw std::runtime_error("KPointNeighborShells::compute: no set of shells up to " +
                                 std::to_string(maximum_shell_count) + " shells (search range " +
                                 std::to_string(search_range) +
                                 ") satisfies the completeness condition to the requested "
                                 "tolerance -- widen search_range or maximum_shell_count.");
    }

    [[nodiscard]] const std::vector<KPointNeighborDirection>& neighbor_directions() const {
        return neighbor_directions_;
    }

    [[nodiscard]] int number_of_shells() const {
        return number_of_shells_;
    }

    /**
     * Which physical k-point k + b lands on, from the grid shape alone.
     *
     * Separate from resolve_neighbor() because a consumer that already trusts
     * the k-point list -- the stencil, walking a record's own neighbour table
     * -- needs the index and not the periodic-image shift, and must get it
     * from exactly this arithmetic rather than a second copy of it. The
     * enumeration order is the one KPointsGrid produces: first index slowest,
     * third fastest.
     */
    [[nodiscard]] static int neighbor_kpoint_index(int starting_kpoint_index,
                                                   const IntegerVector3D& grid_step,
                                                   const std::array<int, 3>& grid_dimensions) {
        const int dimension_x = grid_dimensions[0];
        const int dimension_y = grid_dimensions[1];
        const int dimension_z = grid_dimensions[2];

        const int index_z = starting_kpoint_index % dimension_z;
        const int index_y = (starting_kpoint_index / dimension_z) % dimension_y;
        const int index_x = starting_kpoint_index / (dimension_y * dimension_z);

        return wrap_index(index_x + grid_step.x, dimension_x) * dimension_y * dimension_z +
               wrap_index(index_y + grid_step.y, dimension_y) * dimension_z +
               wrap_index(index_z + grid_step.z, dimension_z);
    }

    /**
     * Resolve the physical k-point index and periodic-image (reciprocal
     * lattice vector) shift that `grid_step` leads to, starting from
     * `starting_kpoint_index` on a full, unreduced Monkhorst-Pack grid of
     * `grid_dimensions`. `physical_kpoints` must be that grid's coordinate
     * list, enumerated in the same (x-major, y, z-minor) nested order as
     * KPointsGrid::generate() -- the neighbor's flat index is recovered by
     * exact integer arithmetic on that ordering, not by searching
     * coordinates.
     */
    [[nodiscard]] static ResolvedKPointNeighbor
    resolve_neighbor(int starting_kpoint_index,
                     const IntegerVector3D& grid_step,
                     const std::array<int, 3>& grid_dimensions,
                     const std::vector<KPoint>& physical_kpoints) {
        ResolvedKPointNeighbor resolved;
        resolved.neighbor_kpoint_index =
            neighbor_kpoint_index(starting_kpoint_index, grid_step, grid_dimensions);

        const Vector3D& starting_coordinates = physical_kpoints[starting_kpoint_index].coordinates;
        const Vector3D& neighbor_coordinates =
            physical_kpoints[resolved.neighbor_kpoint_index].coordinates;

        for (int component = 0; component < 3; component++) {
            const Real target =
                starting_coordinates[component] + static_cast<Real>(grid_step[component]) /
                                                      static_cast<Real>(grid_dimensions[component]);
            const Real difference = target - neighbor_coordinates[component];
            const int shift_component = static_cast<int>(std::lround(difference));

            if (std::abs(difference - static_cast<Real>(shift_component)) > 1e-6) {
                throw std::runtime_error(
                    "KPointNeighborShells::resolve_neighbor: the k-point list does not "
                    "enumerate a full, unreduced Monkhorst-Pack grid with the given "
                    "dimensions in the expected order (first index slowest, third fastest, "
                    "as KPointsGrid and q3.workflows.interpolation.full_grid_kpoints produce) "
                    "-- the neighbor's coordinate does not differ from the target by an "
                    "integer number of reciprocal lattice vectors.");
            }
            resolved.periodic_image_shift[component] = shift_component;
        }

        return resolved;
    }

  private:
    struct Shell {
        std::vector<IntegerVector3D> grid_steps;
        std::vector<Vector3D> translations;
        Real magnitude_squared = 0.0;
    };

    struct LeastSquaresSolution {
        std::vector<Real> weights;
        Real residual_norm = 0.0;
        bool smallest_singular_value_cleared_tolerance = false;
    };

    static int wrap_index(int raw_index, int dimension) {
        int wrapped = raw_index % dimension;
        if (wrapped < 0) {
            wrapped += dimension;
        }
        return wrapped;
    }

    /**
     * Sum, over every vector in the shell, of its outer product with
     * itself, packed into the 6 independent symmetric components (xx, yy,
     * zz, xy, xz, yz).
     */
    static std::array<Real, 6> shell_tensor(const Shell& shell) {
        std::array<Real, 6> tensor = {0, 0, 0, 0, 0, 0};
        for (const Vector3D& translation : shell.translations) {
            tensor[0] += translation.x * translation.x;
            tensor[1] += translation.y * translation.y;
            tensor[2] += translation.z * translation.z;
            tensor[3] += translation.x * translation.y;
            tensor[4] += translation.x * translation.z;
            tensor[5] += translation.y * translation.z;
        }
        return tensor;
    }

    /**
     * Enumerate integer grid steps in the search cube, convert each to a
     * Cartesian translation, and group them into shells of equal length
     * (within `shell_relative_tolerance`), sorted by increasing length.
     * Every shell is closed under grid_step -> -grid_step by construction,
     * since the search cube is symmetric about the origin.
     */
    static std::vector<Shell>
    enumerate_candidate_shells(const Matrix3x3& reciprocal_lattice_vectors,
                               const std::array<int, 3>& grid_dimensions,
                               int search_range,
                               Real shell_relative_tolerance) {
        struct Candidate {
            IntegerVector3D grid_step;
            Vector3D translation;
            Real magnitude_squared;
        };

        std::vector<Candidate> candidates;
        for (int step_x = -search_range; step_x <= search_range; step_x++) {
            for (int step_y = -search_range; step_y <= search_range; step_y++) {
                for (int step_z = -search_range; step_z <= search_range; step_z++) {
                    if (step_x == 0 && step_y == 0 && step_z == 0) {
                        continue;
                    }
                    Vector3D translation = (static_cast<Real>(step_x) / grid_dimensions[0]) *
                                               reciprocal_lattice_vectors[0] +
                                           (static_cast<Real>(step_y) / grid_dimensions[1]) *
                                               reciprocal_lattice_vectors[1] +
                                           (static_cast<Real>(step_z) / grid_dimensions[2]) *
                                               reciprocal_lattice_vectors[2];
                    candidates.push_back(
                        {{step_x, step_y, step_z}, translation, translation.norm2()});
                }
            }
        }

        /**
         * Stable, so directions of equal length keep the enumeration order
         * above. With std::sort the within-shell order -- hence
         * `direction_index`, the `.nnkp`/`.mmn` block order and the record's
         * bytes -- would depend on the standard library's introsort, i.e. on
         * libc++ versus libstdc++.
         */
        std::stable_sort(candidates.begin(),
                         candidates.end(),
                         [](const Candidate& left, const Candidate& right) {
                             return left.magnitude_squared < right.magnitude_squared;
                         });

        std::vector<Shell> shells;
        for (const Candidate& candidate : candidates) {
            if (!shells.empty() &&
                std::abs(candidate.magnitude_squared - shells.back().magnitude_squared) <=
                    shell_relative_tolerance * shells.back().magnitude_squared) {
                shells.back().grid_steps.push_back(candidate.grid_step);
                shells.back().translations.push_back(candidate.translation);
                continue;
            }
            Shell shell;
            shell.grid_steps.push_back(candidate.grid_step);
            shell.translations.push_back(candidate.translation);
            shell.magnitude_squared = candidate.magnitude_squared;
            shells.push_back(std::move(shell));
        }
        return shells;
    }

    /**
     * Solve the (accumulated shell count) x 6 least-squares system
     * A^T w = target for the per-shell weights w via the SVD-based
     * pseudo-inverse, where each row of A is one shell's tensor. Also
     * reports whether the smallest singular value of A (transposed to
     * shells x 6) clears `singular_value_tolerance`, which is how a
     * newly added shell is judged to contribute a genuinely new
     * direction rather than one linearly dependent on the shells already
     * present.
     */
    static LeastSquaresSolution
    solve_least_squares(const std::vector<std::array<Real, 6>>& shell_tensors,
                        const std::array<Real, 6>& target,
                        Real singular_value_tolerance) {
        const int number_of_shells = static_cast<int>(shell_tensors.size());
        const int number_of_equations = 6;

        /**
         * Column-major A, number_of_equations rows x number_of_shells
         * columns: column s holds shell_tensors[s].
         */
        std::vector<Real> matrix_a(static_cast<std::size_t>(number_of_equations) *
                                   number_of_shells);
        for (int shell_index = 0; shell_index < number_of_shells; shell_index++) {
            for (int component = 0; component < number_of_equations; component++) {
                matrix_a[shell_index * number_of_equations + component] =
                    shell_tensors[shell_index][component];
            }
        }

        const int minimum_dimension = std::min(number_of_equations, number_of_shells);
        std::vector<Real> singular_values(minimum_dimension);
        std::vector<Real> left_singular_vectors(static_cast<std::size_t>(number_of_equations) *
                                                number_of_equations);
        std::vector<Real> right_singular_vectors_transposed(
            static_cast<std::size_t>(number_of_shells) * number_of_shells);

        std::vector<Real> matrix_a_copy = matrix_a;
        int workspace_size = -1;
        Real optimal_workspace_size = 0.0;
        int info = 0;

        lapack::dgesvd_("A",
                        "A",
                        &number_of_equations,
                        &number_of_shells,
                        matrix_a_copy.data(),
                        &number_of_equations,
                        singular_values.data(),
                        left_singular_vectors.data(),
                        &number_of_equations,
                        right_singular_vectors_transposed.data(),
                        &number_of_shells,
                        &optimal_workspace_size,
                        &workspace_size,
                        &info);
        if (info != 0) {
            throw std::runtime_error(
                "KPointNeighborShells: DGESVD workspace query failed with info = " +
                std::to_string(info));
        }

        workspace_size = static_cast<int>(optimal_workspace_size);
        std::vector<Real> workspace(workspace_size);
        matrix_a_copy = matrix_a;

        lapack::dgesvd_("A",
                        "A",
                        &number_of_equations,
                        &number_of_shells,
                        matrix_a_copy.data(),
                        &number_of_equations,
                        singular_values.data(),
                        left_singular_vectors.data(),
                        &number_of_equations,
                        right_singular_vectors_transposed.data(),
                        &number_of_shells,
                        workspace.data(),
                        &workspace_size,
                        &info);
        if (info != 0) {
            throw std::runtime_error("KPointNeighborShells: DGESVD failed with info = " +
                                     std::to_string(info));
        }

        const Real largest_singular_value = singular_values.empty() ? 0.0 : singular_values[0];
        const Real smallest_singular_value =
            singular_values.empty() ? 0.0 : singular_values[minimum_dimension - 1];

        LeastSquaresSolution solution;
        solution.smallest_singular_value_cleared_tolerance =
            (number_of_shells <= number_of_equations) &&
            (smallest_singular_value >
             singular_value_tolerance * std::max(1.0, largest_singular_value));

        /**
         * Pseudo-inverse solve: w = V * Sigma^+ * U^T * target, truncating
         * singular values at the tolerance so a still-degenerate system
         * (more shells than independent directions) does not blow up.
         * left_singular_vectors is U, column-major with leading dimension
         * number_of_equations: U(equation_row, singular_index) is at
         * flat[singular_index * number_of_equations + equation_row].
         */
        std::vector<Real> u_transpose_target(number_of_equations, 0.0);
        for (int singular_index = 0; singular_index < number_of_equations; singular_index++) {
            Real sum = 0.0;
            for (int equation_row = 0; equation_row < number_of_equations; equation_row++) {
                sum += left_singular_vectors[singular_index * number_of_equations + equation_row] *
                       target[equation_row];
            }
            u_transpose_target[singular_index] = sum;
        }

        /**
         * right_singular_vectors_transposed is V**T, column-major with
         * leading dimension number_of_shells: VT(singular_index,
         * shell_index) = V(shell_index, singular_index) -- the
         * shell_index-th component of the singular_index-th right
         * singular vector -- is at flat[shell_index * number_of_shells +
         * singular_index].
         */
        std::vector<Real> weights(number_of_shells, 0.0);
        for (int shell_index = 0; shell_index < number_of_shells; shell_index++) {
            Real sum = 0.0;
            for (int singular_index = 0; singular_index < minimum_dimension; singular_index++) {
                if (singular_values[singular_index] <=
                    singular_value_tolerance * std::max(1.0, largest_singular_value)) {
                    continue;
                }
                sum += right_singular_vectors_transposed[shell_index * number_of_shells +
                                                         singular_index] *
                       u_transpose_target[singular_index] / singular_values[singular_index];
            }
            weights[shell_index] = sum;
        }

        Real residual_squared = 0.0;
        for (int component = 0; component < number_of_equations; component++) {
            Real predicted = 0.0;
            for (int shell_index = 0; shell_index < number_of_shells; shell_index++) {
                predicted += shell_tensors[shell_index][component] * weights[shell_index];
            }
            Real component_residual = predicted - target[component];
            residual_squared += component_residual * component_residual;
        }

        solution.weights = std::move(weights);
        solution.residual_norm = std::sqrt(residual_squared);
        return solution;
    }

    void finalize(const std::vector<const Shell*>& accepted_shells,
                  const std::vector<Real>& weights) {
        for (int shell_index = 0; shell_index < static_cast<int>(accepted_shells.size());
             shell_index++) {
            const Shell& shell = *accepted_shells[shell_index];
            const Real weight = weights[shell_index];
            if (weight < -1e-8) {
                throw std::runtime_error(
                    "KPointNeighborShells::compute: the least-squares solve produced a "
                    "negative shell weight (" +
                    std::to_string(weight) +
                    ") -- the completeness condition has no physically valid solution "
                    "with the shells found; widen search_range.");
            }
            for (std::size_t direction_index = 0; direction_index < shell.grid_steps.size();
                 direction_index++) {
                neighbor_directions_.push_back({shell.grid_steps[direction_index],
                                                shell.translations[direction_index],
                                                weight});
            }
        }
        number_of_shells_ = static_cast<int>(accepted_shells.size());
    }

    std::vector<KPointNeighborDirection> neighbor_directions_;
    int number_of_shells_ = 0;
};

} // namespace q3
