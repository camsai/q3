#pragma once

/**
 * WignerSeitzSupercell — the real-space dual of a k-point mesh: the lattice
 * vectors R inside the Wigner-Seitz cell of the supercell the mesh spans,
 * with the degeneracy of each vector on that cell's boundary.
 *
 * This is geometry, not electronic structure: lattice vectors and three mesh
 * dimensions in, a set of lattice vectors with weights out. It sits beside
 * KPointNeighborShells, which is the same mesh's reciprocal-space
 * neighbourhood, and for the same reason: both are properties the mesh
 * induces on the crystal, and the computation that consumes them (the
 * Wannier k -> R transform, in layer 4) is not what defines them.
 *
 * H(R) = (1/N_k) sum_k e^(-i k.R) H(k) is only well conditioned if the R set
 * is chosen as the Wigner-Seitz cell of the SUPERCELL spanned by the k-point
 * mesh -- i.e. every R kept is at least as close to the origin as to any
 * other superlattice point. Taking a naive box of R instead puts weight on
 * vectors that are periodic images of shorter ones, and the interpolated
 * bands acquire spurious structure between the coarse k-points where nothing
 * constrains them.
 *
 * Points exactly equidistant from several superlattice points sit on the
 * Wigner-Seitz boundary and belong to all of them at once. They are kept, and
 * carry a degeneracy weight equal to how many they are tied with, so the
 * transform stays an exact inverse of the k-space one: dividing by the
 * degeneracy shares such a point's contribution among its images rather than
 * counting it several times. Getting this wrong is a classic source of
 * interpolation error that is invisible at the coarse k-points (where the
 * transform is exact by construction) and only shows up in between -- which
 * is exactly where interpolation is supposed to be doing work.
 *
 * Both search boxes are DERIVED from the lattice rather than fixed. An
 * earlier version searched candidates in a +-2-supercell box and tested them
 * against a +-1-supercell box of images, which is right for isotropic meshes
 * on the cells the tests used and wrong for anisotropic ones on oblique
 * cells: an FCC primitive cell with a 4x4x2 mesh came out with total weight
 * 32.33 instead of 32, and an 8x8x1 mesh kept the weight exactly right while
 * missing six boundary vectors -- so the weight check alone cannot catch
 * every version of this. The bounds below are rigorous:
 *
 *  - Every point x of the Wigner-Seitz cell satisfies |x| <= D/2, where D is
 *    the longest body diagonal of the superlattice parallelepiped. (Write x
 *    in superlattice coordinates and round each to the nearest integer: the
 *    remainder is (1/2) sum_i e_i S_i with |e_i| <= 1, whose norm is maximised
 *    at a vertex of the cube, i.e. on a body diagonal. That nearest vertex is
 *    a superlattice point, and x is at least as close to the origin as to it.)
 *  - A lattice vector with |R| <= D/2 has superlattice coefficient
 *    |c_i| = |R . S*_i| <= (D/2) |S*_i|, with S*_i the dual vectors
 *    (S_i . S*_j = delta_ij). That bounds the candidate box per axis.
 *  - The same rounding argument puts every point within D/2 of SOME
 *    superlattice point, so the nearest image T* of any candidate satisfies
 *    |T*| <= |R| + D/2; images that tie for nearest are no farther. That
 *    bounds the image box from the largest candidate norm in the box.
 *
 * The identity sum_R 1/degeneracy = N_k is then asserted at the end of
 * compute(): it holds for every correct Wigner-Seitz construction, so a
 * violation is a bug in this class, not a property of the input, and it is
 * refused rather than logged.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// One retained lattice vector with the number of images it ties with.
struct [[nodiscard]] WignerSeitzLatticeVector {
    IntegerVector3D lattice_translation;
    /// Cartesian R in the same units as the lattice vectors handed in.
    Vector3D cartesian;
    int degeneracy = 1;
};

class WignerSeitzSupercell {
  public:
    /**
     * Build the R set for a `grid_dimension_x` x `..._y` x `..._z` k-mesh.
     *
     * `lattice_vectors` rows are the primitive real-space vectors a1, a2, a3;
     * the supercell is grid_dimension_i * a_i.
     *
     * `distance_tolerance` decides when two distances count as equal, i.e.
     * when a point is on the boundary. It is relative to the shortest
     * superlattice vector so it transfers across cell sizes.
     */
    void compute(const Matrix3x3& lattice_vectors,
                 int grid_dimension_x,
                 int grid_dimension_y,
                 int grid_dimension_z,
                 Real distance_tolerance = 1e-6) {
        lattice_vectors_.clear();
        if (grid_dimension_x <= 0 || grid_dimension_y <= 0 || grid_dimension_z <= 0) {
            throw std::invalid_argument(
                "WignerSeitzSupercell::compute: every grid dimension must be positive");
        }
        const std::array<int, 3> grid_dimensions = {
            grid_dimension_x, grid_dimension_y, grid_dimension_z};
        const int number_of_kpoints = grid_dimension_x * grid_dimension_y * grid_dimension_z;

        Matrix3x3 superlattice;
        for (int axis = 0; axis < 3; axis++) {
            superlattice[axis] = lattice_vectors[axis] * static_cast<Real>(grid_dimensions[axis]);
        }
        if (std::abs(superlattice.determinant()) <= 0.0) {
            throw std::invalid_argument(
                "WignerSeitzSupercell::compute: the lattice vectors are linearly dependent");
        }

        /**
         * The longest body diagonal of the superlattice parallelepiped. The
         * sign of the first vector is fixed by inversion symmetry, so four
         * diagonals cover all eight.
         */
        Real longest_diagonal = 0.0;
        for (Real sign_y : {-1.0, 1.0}) {
            for (Real sign_z : {-1.0, 1.0}) {
                Vector3D diagonal =
                    superlattice[0] + superlattice[1] * sign_y + superlattice[2] * sign_z;
                longest_diagonal = std::max(longest_diagonal, diagonal.norm());
            }
        }
        const Real ball_radius = 0.5 * longest_diagonal;

        // Dual vectors S*_j are the columns of the inverse of the row matrix.
        const Matrix3x3 inverse_superlattice = superlattice.inverse();
        std::array<Real, 3> dual_norm = {0.0, 0.0, 0.0};
        for (int axis = 0; axis < 3; axis++) {
            Vector3D dual(inverse_superlattice[0][axis],
                          inverse_superlattice[1][axis],
                          inverse_superlattice[2][axis]);
            dual_norm[axis] = dual.norm();
        }

        Real shortest_superlattice_length = superlattice[0].norm();
        for (int axis = 1; axis < 3; axis++) {
            shortest_superlattice_length =
                std::min(shortest_superlattice_length, superlattice[axis].norm());
        }
        const Real absolute_tolerance = distance_tolerance * shortest_superlattice_length;

        /**
         * Candidate box: lattice vectors whose norm can be at most D/2. The
         * bound is on the superlattice coefficient, so it scales by the grid
         * dimension to become a bound on the lattice index.
         */
        std::array<int, 3> candidate_bound = {0, 0, 0};
        for (int axis = 0; axis < 3; axis++) {
            candidate_bound[axis] =
                std::max(1,
                         static_cast<int>(std::floor(
                             grid_dimensions[axis] * ball_radius * dual_norm[axis] + 1e-9)));
        }

        // The norm is convex, so the farthest candidate in the box is a corner.
        Real farthest_candidate = 0.0;
        for (Real sign_y : {-1.0, 1.0}) {
            for (Real sign_z : {-1.0, 1.0}) {
                Vector3D corner = lattice_vectors[0] * static_cast<Real>(candidate_bound[0]) +
                                  lattice_vectors[1] * (sign_y * candidate_bound[1]) +
                                  lattice_vectors[2] * (sign_z * candidate_bound[2]);
                farthest_candidate = std::max(farthest_candidate, corner.norm());
            }
        }

        /**
         * Image box: every superlattice point that can be as close to some
         * candidate as the origin is. The larger of the two bounds covers
         * both the nearest-image argument and the tie argument.
         */
        const Real reach =
            std::max(farthest_candidate + ball_radius, longest_diagonal) + absolute_tolerance;
        std::array<int, 3> image_bound = {0, 0, 0};
        for (int axis = 0; axis < 3; axis++) {
            image_bound[axis] =
                std::max(1, static_cast<int>(std::ceil(reach * dual_norm[axis] - 1e-9)));
        }

        std::vector<Vector3D> superlattice_translations;
        for (int image_x = -image_bound[0]; image_x <= image_bound[0]; image_x++) {
            for (int image_y = -image_bound[1]; image_y <= image_bound[1]; image_y++) {
                for (int image_z = -image_bound[2]; image_z <= image_bound[2]; image_z++) {
                    superlattice_translations.push_back(
                        superlattice[0] * static_cast<Real>(image_x) +
                        superlattice[1] * static_cast<Real>(image_y) +
                        superlattice[2] * static_cast<Real>(image_z));
                }
            }
        }

        for (int index_x = -candidate_bound[0]; index_x <= candidate_bound[0]; index_x++) {
            for (int index_y = -candidate_bound[1]; index_y <= candidate_bound[1]; index_y++) {
                for (int index_z = -candidate_bound[2]; index_z <= candidate_bound[2]; index_z++) {
                    Vector3D candidate = lattice_vectors[0] * static_cast<Real>(index_x) +
                                         lattice_vectors[1] * static_cast<Real>(index_y) +
                                         lattice_vectors[2] * static_cast<Real>(index_z);

                    Real own_distance_squared = candidate.norm2();
                    Real minimum_distance_squared = own_distance_squared;
                    for (const Vector3D& translation : superlattice_translations) {
                        Vector3D shifted = candidate - translation;
                        minimum_distance_squared =
                            std::min(minimum_distance_squared, shifted.norm2());
                    }

                    Real own_distance = std::sqrt(own_distance_squared);
                    Real minimum_distance = std::sqrt(minimum_distance_squared);
                    if (own_distance > minimum_distance + absolute_tolerance) {
                        continue; // a periodic image sits strictly closer
                    }

                    int degeneracy = 0;
                    for (const Vector3D& translation : superlattice_translations) {
                        Vector3D shifted = candidate - translation;
                        if (std::abs(shifted.norm() - minimum_distance) <= absolute_tolerance) {
                            degeneracy++;
                        }
                    }
                    lattice_vectors_.push_back(
                        {IntegerVector3D(index_x, index_y, index_z), candidate, degeneracy});
                }
            }
        }

        if (lattice_vectors_.empty()) {
            throw std::runtime_error(
                "WignerSeitzSupercell::compute: no lattice vectors retained -- the search range "
                "cannot cover the Wigner-Seitz cell of this lattice");
        }

        const Real weight = total_weight();
        if (std::abs(weight - static_cast<Real>(number_of_kpoints)) >
            1e-8 * static_cast<Real>(number_of_kpoints)) {
            throw std::runtime_error(
                "WignerSeitzSupercell::compute: the retained set sums to a weight of " +
                std::to_string(weight) + " but the " + std::to_string(grid_dimension_x) + "x" +
                std::to_string(grid_dimension_y) + "x" + std::to_string(grid_dimension_z) +
                " mesh has " + std::to_string(number_of_kpoints) +
                " k-points. The two must agree for the k -> R -> k transform to be an "
                "identity; this is a defect in the Wigner-Seitz construction, not in the input.");
        }
    }

    [[nodiscard]] const std::vector<WignerSeitzLatticeVector>& lattice_vectors() const {
        return lattice_vectors_;
    }

    [[nodiscard]] int number_of_lattice_vectors() const {
        return static_cast<int>(lattice_vectors_.size());
    }

    /**
     * Sum of 1/degeneracy over the retained set. For a correct Wigner-Seitz
     * construction this equals the number of k-points exactly: each of the
     * N_k cells of the supercell is covered once, with boundary points
     * sharing their coverage. compute() refuses to return a set for which it
     * does not hold; callers may still use it to check that a supercell was
     * built for the mesh they are about to transform over.
     */
    [[nodiscard]] Real total_weight() const {
        Real total = 0.0;
        for (const WignerSeitzLatticeVector& entry : lattice_vectors_) {
            total += 1.0 / static_cast<Real>(entry.degeneracy);
        }
        return total;
    }

  private:
    std::vector<WignerSeitzLatticeVector> lattice_vectors_;
};

} // namespace q3
