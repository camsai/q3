#pragma once

/**
 * How real spherical harmonics, and the crystal's symmetry operations, turn
 * into the matrices that rotate an on-site quantity's angular indices.
 *
 * Two consumers share this: the DFT+U occupation matrix
 * (`effective_potential/internal/hubbard/symmetrization.hpp`) and the PAW
 * on-site occupations (`occupation_symmetrization.hpp` beside this file).
 * Both average an atom-resolved, angular-indexed array over the point group,
 * and both need the same two ingredients — D^l(R) and the Cartesian form of a
 * fractional-coordinate operation — so they are defined once here rather than
 * in whichever module happened to need them first.
 */

#include <vector>

#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/symmetry/operation.hpp"
#include "layer_4_computation/paw/one_center/angular_grid.hpp"

namespace q3 {

/**
 * D^l(R): how the real spherical harmonics of degree `l` mix under a Cartesian
 * rotation.
 *
 *     D_mm'(R) = ∫ dΩ  Y_lm(R r̂)  Y_lm'(r̂)
 *
 * so that Y_lm(R r̂) = Σ_m' D_mm'(R) Y_lm'(r̂). The integrand is a product of
 * two degree-l harmonics, so a grid exact to degree 2l integrates it exactly;
 * `build_angular_grid` is constructed on that contract and the PAW one-center
 * engine relies on the same property.
 *
 * Rows and columns are indexed by position in `harmonic_order_sequence(l)`,
 * which is the order both the Hubbard manifolds and the expanded projector
 * channels are laid out in.
 *
 * Improper operations (det R = −1) need no special handling: a degree-l real
 * harmonic has parity (−1)^l, and evaluating at the rotated direction carries
 * that automatically.
 */
[[nodiscard]] inline std::vector<Real>
build_real_harmonic_rotation_matrix(int angular_momentum, const Matrix3x3& cartesian_rotation) {
    int manifold_size = 2 * angular_momentum + 1;
    std::vector<Real> rotation_matrix(static_cast<size_t>(manifold_size) * manifold_size, 0.0);

    auto angular_grid = paw_one_center::build_angular_grid(2 * angular_momentum);
    const auto& orders = harmonic_order_sequence(angular_momentum);

    for (const auto& point : angular_grid) {
        Vector3D direction{point.x, point.y, point.z};
        Vector3D rotated{0.0, 0.0, 0.0};
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                rotated[row] += cartesian_rotation[row][column] * direction[column];

        for (int m = 0; m < manifold_size; m++) {
            Real rotated_harmonic =
                compute_real_spherical_harmonic(angular_momentum, orders[m], rotated);
            for (int m_prime = 0; m_prime < manifold_size; m_prime++) {
                Real harmonic =
                    compute_real_spherical_harmonic(angular_momentum, orders[m_prime], direction);
                rotation_matrix[static_cast<size_t>(m) * manifold_size + m_prime] +=
                    point.weight * rotated_harmonic * harmonic;
            }
        }
    }
    return rotation_matrix;
}

/**
 * The Cartesian rotation of a symmetry operation stored in crystal coordinates.
 *
 * `SymmetryOperation::rotation` acts on fractional coordinates, so it has to be
 * conjugated by the lattice before it can rotate a direction in space:
 * R_cart = A ᵀ · S · (A ᵀ)^{-1} with A the matrix of lattice vectors as rows.
 * The reciprocal lattice supplies the inverse directly, since
 * (Aᵀ)^{-1} = B / 2π with B the reciprocal vectors as rows.
 */
[[nodiscard]] inline Matrix3x3 cartesian_rotation_of(const Crystal& crystal_structure,
                                                     const SymmetryOperation& operation) {
    const Matrix3x3& direct = crystal_structure.lattice_vectors();
    const Matrix3x3& reciprocal = crystal_structure.reciprocal_lattice_vectors();

    Matrix3x3 cartesian{};
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            Real accumulated = 0.0;
            for (int first = 0; first < 3; first++)
                for (int second = 0; second < 3; second++)
                    accumulated += direct[first][row] *
                                   static_cast<Real>(operation.rotation[first][second]) *
                                   reciprocal[second][column];
            cartesian[row][column] = accumulated;
        }
    }
    return cartesian;
}

} // namespace q3
