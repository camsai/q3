#pragma once

/**
 * Angular quadrature for one-center exchange-correlation.
 *
 * XC is not linear in the density, so it cannot be evaluated channel by
 * channel the way Hartree can: the lm-decomposed density has to be summed onto
 * real directions, the functional evaluated pointwise there, and the result
 * projected back onto Y_LM. That needs a quadrature exact for the products
 * that appear — which is a Gauss-Legendre grid in cos θ crossed with a uniform
 * grid in φ.
 *
 * Quantum ESPRESSO's PAW one-center machinery uses exactly this family, and the
 * one-center parity gates — the µRy QE control and the absolute-reference
 * gate — confirm the same choice reproduces both references, so quadrature is not a
 * candidate explanation for a mismatch against either. A Lebedev grid would
 * also work and is not used: it would be a third convention for no accuracy
 * gain at one-center scale.
 *
 * Sizing: Gauss-Legendre with n points integrates polynomials in cos θ up to
 * degree 2n−1 exactly, and products of two Y_LM reach degree 2·l_max, so
 * n = l_max + 1 suffices for the density-squared products; the φ grid needs
 * more than 2·l_max points to resolve the azimuthal modes. Both are taken
 * with a factor of three of headroom, since the cost is negligible at
 * one-center scale and GGA raises the effective degree.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {
namespace paw_one_center {

/// One direction on the unit sphere with its quadrature weight.
struct AngularPoint {
    Real x{};
    Real y{};
    Real z{};
    Real weight{}; // ∫dΩ f ≈ Σ weight · f, so Σ weight = 4π
};

/**
 * Gauss-Legendre nodes and weights on [-1, 1], by Newton iteration on the
 * Legendre polynomial. Converges quadratically; the iteration is run to
 * machine tolerance because the exactness gate leaves no room for a loose
 * root.
 */
inline void
gauss_legendre_nodes(int number_of_points, std::vector<Real>& nodes, std::vector<Real>& weights) {
    nodes.assign(number_of_points, 0.0);
    weights.assign(number_of_points, 0.0);
    for (int index = 0; index < number_of_points; index++) {
        // Chebyshev starting guess, then Newton on P_n.
        Real node = std::cos(constants::pi * (static_cast<Real>(index) + 0.75) /
                             (static_cast<Real>(number_of_points) + 0.5));
        Real derivative = 0.0;
        for (int iteration = 0; iteration < 100; iteration++) {
            Real previous = 1.0;
            Real current = 0.0;
            for (int degree = 0; degree < number_of_points; degree++) {
                Real older = current;
                current = previous;
                previous =
                    ((2.0 * degree + 1.0) * node * current - degree * older) / (degree + 1.0);
            }
            derivative = number_of_points * (node * previous - current) / (node * node - 1.0);
            Real step = previous / derivative;
            node -= step;
            if (std::abs(step) < 1e-15)
                break;
        }
        nodes[index] = node;
        weights[index] = 2.0 / ((1.0 - node * node) * derivative * derivative);
    }
}

/**
 * Build the θ × φ product grid.
 *
 * @param maximum_angular_momentum  l_max the grid must integrate products of
 */
inline std::vector<AngularPoint> build_angular_grid(int maximum_angular_momentum) {
    int theta_points = 3 * (maximum_angular_momentum + 1);
    int phi_points = 3 * (maximum_angular_momentum + 1);
    if (theta_points < 2)
        theta_points = 2;
    if (phi_points < 2)
        phi_points = 2;

    std::vector<Real> cosines;
    std::vector<Real> theta_weights;
    gauss_legendre_nodes(theta_points, cosines, theta_weights);

    std::vector<AngularPoint> grid;
    grid.reserve(static_cast<size_t>(theta_points) * phi_points);
    const Real phi_step = 2.0 * constants::pi / static_cast<Real>(phi_points);
    for (int theta_index = 0; theta_index < theta_points; theta_index++) {
        Real cosine = cosines[theta_index];
        Real sine = std::sqrt(std::max(Real(0.0), 1.0 - cosine * cosine));
        for (int phi_index = 0; phi_index < phi_points; phi_index++) {
            Real phi = phi_step * static_cast<Real>(phi_index);
            AngularPoint point;
            point.x = sine * std::cos(phi);
            point.y = sine * std::sin(phi);
            point.z = cosine;
            point.weight = theta_weights[theta_index] * phi_step;
            grid.push_back(point);
        }
    }
    return grid;
}

} // namespace paw_one_center
} // namespace q3
