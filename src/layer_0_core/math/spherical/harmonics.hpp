#pragma once

/**
 * Spherical harmonics Y_lm and degree/order indexing utilities.
 *
 * Provides:
 * - Real spherical harmonics Y_lm(theta, phi), Condon-Shortley phase retained
 * - Degree/order indexing: m = 0 first, then (+|m|, -|m|) pairs in ascending
 *   |m|, which keeps the cos(m*phi) and sin(m*phi) partners of one |m|
 *   adjacent
 *
 * Both are basis choices. Everything Q3 computes is invariant under any other
 * consistent choice, so what matters is that every consumer in the tree uses
 * these two functions rather than reproducing the mapping.
 *
 * Terminology: l = degree, m = order (standard mathematical convention).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Return harmonic order sequence: 0, 1, -1, 2, -2, ...
 * This alternating-sign convention places m=0 first, then pairs
 * (|m|, -|m|) in ascending |m| order. Widely used for
 * real spherical harmonic indexing and projector enumeration.
 *
 * @param degree  Spherical harmonic degree l
 * @return Vector of order values in alternating-sign sequence
 */
inline std::vector<int> harmonic_order_sequence(int degree) {
    std::vector<int> sequence;
    sequence.push_back(0);
    for (int order_index = 1; order_index <= degree; order_index++) {
        sequence.push_back(order_index);
        sequence.push_back(-order_index);
    }
    return sequence;
}

/**
 * Compute combined (l,m) index from degree and order.
 * Maps the alternating-sign ordering to a flat 0-based index:
 *   m=0  → l²
 *   m>0  → l² + 2m - 1
 *   m<0  → l² + 2|m|
 *
 * @param degree  Spherical harmonic degree l
 * @param order   Spherical harmonic order m
 * @return Combined lm index (0-based)
 */
inline int harmonic_combined_degree_order_index(int degree, int order) {
    if (order == 0) {
        return degree * degree;
    }
    if (order > 0) {
        return degree * degree + 2 * order - 1;
    }
    return degree * degree - 2 * order;
}

/**
 * Compute real spherical harmonic Y_lm at a Cartesian direction vector.
 * Hard-coded closed forms for l=0..4, carrying the Condon-Shortley (-1)^m
 * phase through to the real combinations -- so Y_11 is proportional to -x/r,
 * not +x/r as in the many presentations that drop the phase once the
 * harmonics are real. Agrees with the recursion in
 * compute_real_spherical_harmonics_batch below to ~1e-14 over the whole
 * l <= 4 set, which is the property the one-center path depends on.
 *
 * @param degree     Spherical harmonic degree l (0..4)
 * @param order      Spherical harmonic order m (-l..+l)
 * @param direction  Cartesian direction vector (need not be normalized)
 * @return Y_lm evaluated at the direction of the vector
 */
inline Real compute_real_spherical_harmonic(int degree, int order, const Vector3D& direction) {
    Real magnitude = direction.norm();
    if (magnitude < math::tolerances::zero_vector_magnitude_threshold) {
        if (degree == 0)
            return 1.0 / std::sqrt(constants::four_pi);
        return 0.0;
    }
    Real normalized_x = direction.x / magnitude;
    Real normalized_y = direction.y / magnitude;
    Real normalized_z = direction.z / magnitude;

    if (degree == 0) {
        return 1.0 / std::sqrt(constants::four_pi);
    } else if (degree == 1) {
        Real normalization_factor = std::sqrt(3.0 / constants::four_pi);
        if (order == -1)
            return -normalization_factor * normalized_y;
        if (order == 0)
            return normalization_factor * normalized_z;
        if (order == 1)
            return -normalization_factor * normalized_x;
    } else if (degree == 2) {
        Real normalization_factor_l2 = std::sqrt(15.0 / constants::four_pi);
        if (order == -2)
            return normalization_factor_l2 * normalized_x * normalized_y;
        if (order == -1)
            return -normalization_factor_l2 * normalized_y * normalized_z;
        if (order == 0)
            return std::sqrt(5.0 / constants::four_pi) / 2.0 *
                   (3.0 * normalized_z * normalized_z - 1.0);
        if (order == 1)
            return -normalization_factor_l2 * normalized_x * normalized_z;
        if (order == 2)
            return normalization_factor_l2 / 2.0 *
                   (normalized_x * normalized_x - normalized_y * normalized_y);
    } else if (degree == 3) {
        if (order == -3)
            return -std::sqrt(35.0 / (32.0 * constants::pi)) * normalized_y *
                   (3.0 * normalized_x * normalized_x - normalized_y * normalized_y);
        if (order == -2)
            return std::sqrt(105.0 / (4.0 * constants::pi)) * normalized_x * normalized_y *
                   normalized_z;
        if (order == -1)
            return -std::sqrt(21.0 / (32.0 * constants::pi)) * normalized_y *
                   (5.0 * normalized_z * normalized_z - 1.0);
        if (order == 0)
            return std::sqrt(7.0 / (16.0 * constants::pi)) * normalized_z *
                   (5.0 * normalized_z * normalized_z - 3.0);
        if (order == 1)
            return -std::sqrt(21.0 / (32.0 * constants::pi)) * normalized_x *
                   (5.0 * normalized_z * normalized_z - 1.0);
        if (order == 2)
            return std::sqrt(105.0 / (16.0 * constants::pi)) * normalized_z *
                   (normalized_x * normalized_x - normalized_y * normalized_y);
        if (order == 3)
            return -std::sqrt(35.0 / (32.0 * constants::pi)) * normalized_x *
                   (normalized_x * normalized_x - 3.0 * normalized_y * normalized_y);
    } else if (degree == 4) {
        Real normalized_x2 = normalized_x * normalized_x,
             normalized_y2 = normalized_y * normalized_y,
             normalized_z2 = normalized_z * normalized_z;
        if (order == -4)
            return (3.0 / 4.0) * std::sqrt(35.0 / (constants::pi)) * normalized_x * normalized_y *
                   (normalized_x2 - normalized_y2);
        if (order == -3)
            return -(3.0 / 4.0) * std::sqrt(35.0 / (2.0 * constants::pi)) * normalized_y *
                   normalized_z * (3.0 * normalized_x2 - normalized_y2);
        if (order == -2)
            return (3.0 / 4.0) * std::sqrt(5.0 / (constants::pi)) * normalized_x * normalized_y *
                   (7.0 * normalized_z2 - 1.0);
        if (order == -1)
            return -(3.0 / 4.0) * std::sqrt(5.0 / (2.0 * constants::pi)) * normalized_y *
                   normalized_z * (7.0 * normalized_z2 - 3.0);
        if (order == 0)
            return (3.0 / 16.0) * std::sqrt(1.0 / (constants::pi)) *
                   (35.0 * normalized_z2 * normalized_z2 - 30.0 * normalized_z2 + 3.0);
        if (order == 1)
            return -(3.0 / 4.0) * std::sqrt(5.0 / (2.0 * constants::pi)) * normalized_x *
                   normalized_z * (7.0 * normalized_z2 - 3.0);
        if (order == 2)
            return (3.0 / 8.0) * std::sqrt(5.0 / (constants::pi)) *
                   (normalized_x2 - normalized_y2) * (7.0 * normalized_z2 - 1.0);
        if (order == 3)
            return -(3.0 / 4.0) * std::sqrt(35.0 / (2.0 * constants::pi)) * normalized_x *
                   normalized_z * (normalized_x2 - 3.0 * normalized_y2);
        if (order == 4)
            return (3.0 / 16.0) * std::sqrt(35.0 / (constants::pi)) *
                   (normalized_x2 * normalized_x2 - 6.0 * normalized_x2 * normalized_y2 +
                    normalized_y2 * normalized_y2);
    }
    /**
     * Above degree 4 there is no closed form here, and there never silently
     * was one: this returned 0.0, which is a legal value of every harmonic
     * and therefore indistinguishable from an answer. A caller that asked for
     * a g-channel harmonic got a number back, used it, and produced a result
     * with no error anywhere in the chain -- the Gaunt table built by
     * collocation is the case that made this concrete, where the zeros left
     * the sampling matrix singular in exactly the rows they should have
     * filled, and the inversion skipped those pivots rather than failing.
     *
     * Refusing is not a limitation this introduces; it is the existing
     * limitation, said out loud. Extending the closed forms to l = 5 and 6,
     * or routing this through the recursion that
     * compute_real_spherical_harmonics_batch already uses for arbitrary
     * degree, is the way to lift it -- see TODOs.md.
     */
    throw std::invalid_argument(
        "compute_real_spherical_harmonic: degree " + std::to_string(degree) +
        " is above the highest closed form implemented here (4), so no value can be "
        "returned. An f-channel dataset reaches this through a Q-function expansion of "
        "degree " +
        std::to_string(degree) +
        "; it needs harmonics up to degree 6, which Q3 does not "
        "yet have.");
}

/**
 * Compute real spherical harmonics Y_lm for a batch of G-vectors.
 *
 * Two stages per G-vector: the normalized associated Legendre values Q(l,m)
 * are built by upward recursion into their own triangular workspace, then the
 * azimuthal expansion writes each Y_lm into the slot
 * harmonic_combined_degree_order_index names for its (l, m). The output array
 * is only ever written, never read back as scratch.
 *
 * @param number_of_lm_components        Total number of (l,m) components = (lmax+1)²
 * @param number_of_g_vectors            Number of G-vectors in the batch
 * @param g_vectors                      Cartesian G-vectors (3-component arrays)
 * @param g_squared                      |G|² for each G-vector
 * @param spherical_harmonics_values     Output: values[g_index][lm] for all G-vectors and (l,m)
 * components
 */
inline void
compute_real_spherical_harmonics_batch(int number_of_lm_components,
                                       int number_of_g_vectors,
                                       const std::vector<std::array<Real, 3>>& g_vectors,
                                       const std::vector<Real>& g_squared,
                                       std::vector<std::vector<Real>>& spherical_harmonics_values) {
    int maximum_angular_momentum = 0;
    while ((maximum_angular_momentum + 1) * (maximum_angular_momentum + 1) <
           number_of_lm_components)
        maximum_angular_momentum++;

    spherical_harmonics_values.assign(number_of_g_vectors,
                                      std::vector<Real>(number_of_lm_components, 0.0));

    if (number_of_lm_components == 0 || number_of_g_vectors == 0)
        return;

    /**
     * |G| below which the direction is not defined at all. In a plane-wave
     * basis this is G = 0 exactly and nothing else: the next-smallest |G| is a
     * reciprocal lattice vector. Y_lm still has to return something there, and
     * the convention below is the one the callers have always been given --
     * stated here rather than left to fall out of the arithmetic.
     */
    constexpr Real degenerate_direction_threshold = 1e-9;

    /**
     * Q(l,m) = sqrt((l-m)!/(l+m)!) * P_l^m(cos_theta) for 0 <= m <= l, the
     * combination the azimuthal expansion needs. Held in its own triangular
     * workspace of exactly the (l,m) pairs that exist, rather than parked in
     * the unused slots of the output array: the recursion reads three Q values
     * per step, and aliasing them onto the results makes the write pattern of
     * the second phase depend on the layout of the first.
     *
     * Normalizing before recursing rather than after is what keeps the
     * intermediates O(1) -- the unnormalized P_l^m grow like (2l-1)!! and
     * overflow well before the l this routine is asked for.
     */
    const int legendre_size = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 2) / 2;
    std::vector<Real> legendre(static_cast<size_t>(legendre_size), 0.0);
    const auto legendre_index = [](int degree, int order) {
        return degree * (degree + 1) / 2 + order;
    };

    for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
        const Real g_modulus = std::sqrt(g_squared[g_index]);
        const bool direction_is_degenerate = g_modulus < degenerate_direction_threshold;

        const Real cos_theta = direction_is_degenerate ? 0.0 : g_vectors[g_index][2] / g_modulus;
        const Real sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

        /**
         * atan2 is total on every real direction, including the x = 0 meridian
         * where a quotient is not. It is not total on the degenerate direction:
         * atan2(0, 0) is 0, which would silently move G = 0 onto the x axis and
         * change every m != 0 component there. So that one case keeps its own
         * branch and its own documented convention (the +y meridian).
         */
        const Real azimuthal_angle = direction_is_degenerate
                                         ? std::copysign(constants::pi / 2.0, g_vectors[g_index][1])
                                         : std::atan2(g_vectors[g_index][1], g_vectors[g_index][0]);

        std::fill(legendre.begin(), legendre.end(), 0.0);
        legendre[legendre_index(0, 0)] = 1.0;
        if (maximum_angular_momentum >= 1) {
            legendre[legendre_index(1, 0)] = cos_theta;
            legendre[legendre_index(1, 1)] = -sin_theta / std::sqrt(2.0);
        }

        for (int degree = 2; degree <= maximum_angular_momentum; degree++) {
            for (int order = 0; order <= degree - 2; order++) {
                const Real denominator =
                    std::sqrt(static_cast<Real>(degree * degree - order * order));
                legendre[legendre_index(degree, order)] =
                    cos_theta * (2 * degree - 1) / denominator *
                        legendre[legendre_index(degree - 1, order)] -
                    std::sqrt(static_cast<Real>((degree - 1) * (degree - 1) - order * order)) /
                        denominator * legendre[legendre_index(degree - 2, order)];
            }
            const Real diagonal_factor = std::sqrt(static_cast<Real>(2 * degree - 1));
            const Real previous_diagonal = legendre[legendre_index(degree - 1, degree - 1)];
            legendre[legendre_index(degree, degree - 1)] =
                cos_theta * diagonal_factor * previous_diagonal;
            legendre[legendre_index(degree, degree)] = -diagonal_factor /
                                                       std::sqrt(static_cast<Real>(2 * degree)) *
                                                       sin_theta * previous_diagonal;
        }

        /**
         * Each Y_lm is written to the slot harmonic_combined_degree_order_index
         * names for its (l, m), so this loop states where every value lands
         * instead of walking a counter that has to stay in step with it.
         */
        std::vector<Real>& values = spherical_harmonics_values[g_index];
        values[harmonic_combined_degree_order_index(0, 0)] =
            legendre[legendre_index(0, 0)] / std::sqrt(constants::four_pi);

        for (int degree = 1; degree <= maximum_angular_momentum; degree++) {
            const Real normalization =
                std::sqrt(static_cast<Real>(2 * degree + 1) / constants::four_pi);
            values[harmonic_combined_degree_order_index(degree, 0)] =
                normalization * legendre[legendre_index(degree, 0)];

            const Real off_axis_normalization = normalization * std::sqrt(2.0);
            for (int order = 1; order <= degree; order++) {
                const Real legendre_value = legendre[legendre_index(degree, order)];
                values[harmonic_combined_degree_order_index(degree, order)] =
                    off_axis_normalization * legendre_value * std::cos(order * azimuthal_angle);
                values[harmonic_combined_degree_order_index(degree, -order)] =
                    off_axis_normalization * legendre_value * std::sin(order * azimuthal_angle);
            }
        }
    }
}

/**
 * Compute derivatives of real spherical harmonics dY_lm/dG_i for a batch of G-vectors.
 * Central finite differences on the batch Ylm function rather than closed-form
 * angular derivatives -- the same trade Q3 makes for the one-center tangential
 * gradients: one differentiation path to keep correct instead of a per-(l,m)
 * table, at a cost negligible next to the two batch evaluations it already
 * needs. The step is relative (delta * |G|) because Y_lm depends only on the
 * direction, so |G| is the only scale in the problem and a fixed absolute step
 * would be wrong across a shell structure spanning orders of magnitude.
 *
 * This is much more efficient than computing derivatives per-projector-per-G because:
 * - The batch Ylm function computes all (l,m) components at once per G-vector
 * - Only 2 batch calls per polarization direction (6 total) regardless of number of projectors
 *
 * @param number_of_lm_components        Total number of (l,m) components = (lmax+1)²
 * @param number_of_g_vectors            Number of G-vectors in the batch
 * @param g_vectors                      Cartesian G-vectors (3-component arrays)
 * @param g_squared                      |G|² for each G-vector
 * @param polarization_index             Cartesian direction index (0=x, 1=y, 2=z)
 * @param spherical_harmonic_derivatives Output: derivatives[g_index][lm]
 */
inline void compute_spherical_harmonic_derivatives_batch(
    int number_of_lm_components,
    int number_of_g_vectors,
    const std::vector<std::array<Real, 3>>& g_vectors,
    const std::vector<Real>& g_squared,
    int polarization_index,
    std::vector<std::vector<Real>>& spherical_harmonic_derivatives) {

    constexpr Real delta = 1.0e-6;
    constexpr Real epsilon_threshold = 1.0e-9;

    // Create perturbed G-vectors: G + delta*|G|*e_i and G - delta*|G|*e_i
    std::vector<std::array<Real, 3>> g_vectors_plus(number_of_g_vectors);
    std::vector<std::array<Real, 3>> g_vectors_minus(number_of_g_vectors);
    std::vector<Real> g_squared_plus(number_of_g_vectors);
    std::vector<Real> g_squared_minus(number_of_g_vectors);
    std::vector<Real> delta_g(number_of_g_vectors);

    for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
        Real g_modulus = std::sqrt(g_squared[g_index]);
        delta_g[g_index] = delta * g_modulus;

        g_vectors_plus[g_index] = g_vectors[g_index];
        g_vectors_minus[g_index] = g_vectors[g_index];
        g_vectors_plus[g_index][polarization_index] += delta_g[g_index];
        g_vectors_minus[g_index][polarization_index] -= delta_g[g_index];

        g_squared_plus[g_index] = g_vectors_plus[g_index][0] * g_vectors_plus[g_index][0] +
                                  g_vectors_plus[g_index][1] * g_vectors_plus[g_index][1] +
                                  g_vectors_plus[g_index][2] * g_vectors_plus[g_index][2];
        g_squared_minus[g_index] = g_vectors_minus[g_index][0] * g_vectors_minus[g_index][0] +
                                   g_vectors_minus[g_index][1] * g_vectors_minus[g_index][1] +
                                   g_vectors_minus[g_index][2] * g_vectors_minus[g_index][2];
    }

    // Compute Y_lm at G + dG and G - dG
    std::vector<std::vector<Real>> ylm_plus, ylm_minus;
    compute_real_spherical_harmonics_batch(
        number_of_lm_components, number_of_g_vectors, g_vectors_plus, g_squared_plus, ylm_plus);
    compute_real_spherical_harmonics_batch(
        number_of_lm_components, number_of_g_vectors, g_vectors_minus, g_squared_minus, ylm_minus);

    // Compute central finite difference: dY/dG_i = (Y(G+dG) - Y(G-dG)) / (2*dG)
    spherical_harmonic_derivatives.resize(number_of_g_vectors,
                                          std::vector<Real>(number_of_lm_components, 0.0));
    for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
        if (g_squared[g_index] > epsilon_threshold) {
            Real inverse_two_delta_g = 0.5 / delta_g[g_index];
            for (int lm_index = 0; lm_index < number_of_lm_components; lm_index++) {
                spherical_harmonic_derivatives[g_index][lm_index] =
                    (ylm_plus[g_index][lm_index] - ylm_minus[g_index][lm_index]) *
                    inverse_two_delta_g;
            }
        }
    }
}

} // namespace q3
