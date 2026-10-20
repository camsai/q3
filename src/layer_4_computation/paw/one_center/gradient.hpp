#pragma once

/**
 * Density gradients on the one-center radial × angular grid.
 *
 * A GGA functional needs σ = |∇n|², and on this grid the gradient separates
 * into a radial and a tangential part:
 *
 *     ∇n = Σ_lm [ f'_lm(r) Y_lm(Ω) r̂  +  (f_lm(r)/r) ∇_Ω Y_lm(Ω) ]
 *
 * with f_lm(r) = u_lm(r)/r² the actual density channel (the stored form is
 * r-weighted). The two pieces are orthogonal, so
 * σ = (∂n/∂r)² + |∇_Ω n|²/r².
 *
 * The angular gradients ∇_Ω Y_lm depend only on direction, so they are
 * tabulated once for the whole grid and reused at every radius.
 *
 * Endpoint care: scalar-relativistic partial waves behave as r^γ with
 * non-integer γ toward the origin, so a radial derivative that assumes
 * polynomial behaviour there is wrong for these datasets specifically. The
 * scheme below differentiates in index space on the logarithmic grid — where
 * r^γ becomes a smooth exponential — and the gate differentiates the datasets'
 * own partial waves rather than only analytic test functions.
 */

#include <array>
#include <cmath>
#include <vector>

#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/paw/one_center/angular_grid.hpp"

namespace q3 {
namespace paw_one_center {

/**
 * df/dr on a non-uniform grid, by five-point central differences in index
 * space divided by dr/di.
 *
 * Differentiating in index space is what makes this safe on a logarithmic
 * mesh: a power law r^γ is an exponential in the index, so the stencil sees a
 * smooth function even where the density itself has a cusp-like profile.
 */
inline std::vector<Real> radial_derivative(const std::vector<Real>& values,
                                           const std::vector<Real>& measures,
                                           int mesh_size) {
    std::vector<Real> derivative(mesh_size, 0.0);
    if (mesh_size < 5)
        return derivative;

    for (int index = 2; index < mesh_size - 2; index++) {
        Real index_derivative = (values[index - 2] - 8.0 * values[index - 1] +
                                 8.0 * values[index + 1] - values[index + 2]) /
                                12.0;
        derivative[index] = index_derivative / measures[index];
    }
    // One-sided fourth-order stencils at the four boundary points.
    auto forward = [&](int index) {
        return (-25.0 * values[index] + 48.0 * values[index + 1] - 36.0 * values[index + 2] +
                16.0 * values[index + 3] - 3.0 * values[index + 4]) /
               12.0;
    };
    auto backward = [&](int index) {
        return (25.0 * values[index] - 48.0 * values[index - 1] + 36.0 * values[index - 2] -
                16.0 * values[index - 3] + 3.0 * values[index - 4]) /
               12.0;
    };
    derivative[0] = forward(0) / measures[0];
    derivative[1] = forward(1) / measures[1];
    derivative[mesh_size - 1] = backward(mesh_size - 1) / measures[mesh_size - 1];
    derivative[mesh_size - 2] = backward(mesh_size - 2) / measures[mesh_size - 2];
    return derivative;
}

/**
 * Tangential gradients ∇_Ω Y_lm at every angular grid point, [point][lm][xyz].
 *
 * Obtained by differentiating Y_lm along the unit sphere: perturbing the
 * direction and renormalizing projects out the radial component
 * automatically, which is exactly the tangential derivative wanted here.
 */
inline std::vector<std::vector<std::array<Real, 3>>>
tabulate_harmonic_gradients(const std::vector<AngularPoint>& angular_grid, int number_of_lm) {
    int maximum_angular_momentum = 0;
    while ((maximum_angular_momentum + 1) * (maximum_angular_momentum + 1) < number_of_lm)
        maximum_angular_momentum++;

    constexpr Real step = 1e-5;
    std::vector<std::vector<std::array<Real, 3>>> gradients(
        angular_grid.size(), std::vector<std::array<Real, 3>>(number_of_lm, {0.0, 0.0, 0.0}));

    for (size_t point = 0; point < angular_grid.size(); point++) {
        std::array<Real, 3> direction{
            angular_grid[point].x, angular_grid[point].y, angular_grid[point].z};
        for (int axis = 0; axis < 3; axis++) {
            std::array<Real, 3> plus = direction;
            std::array<Real, 3> minus = direction;
            plus[axis] += step;
            minus[axis] -= step;
            Real plus_norm = std::sqrt(plus[0] * plus[0] + plus[1] * plus[1] + plus[2] * plus[2]);
            Real minus_norm =
                std::sqrt(minus[0] * minus[0] + minus[1] * minus[1] + minus[2] * minus[2]);
            Vector3D plus_direction{plus[0] / plus_norm, plus[1] / plus_norm, plus[2] / plus_norm};
            Vector3D minus_direction{
                minus[0] / minus_norm, minus[1] / minus_norm, minus[2] / minus_norm};

            for (int degree = 0; degree <= maximum_angular_momentum; degree++) {
                for (int order : harmonic_order_sequence(degree)) {
                    int lm = harmonic_combined_degree_order_index(degree, order);
                    if (lm >= number_of_lm)
                        continue;
                    Real value_plus =
                        compute_real_spherical_harmonic(degree, order, plus_direction);
                    Real value_minus =
                        compute_real_spherical_harmonic(degree, order, minus_direction);
                    gradients[point][lm][axis] = (value_plus - value_minus) / (2.0 * step);
                }
            }
        }
    }
    return gradients;
}

/// ∇n at one radius, for every angular point: radial part, tangential part, σ.
struct GradientAtRadius {
    std::vector<Real> radial_component;          ///< ∂n/∂r per angle
    std::vector<std::array<Real, 3>> tangential; ///< ∇_Ω n / r per angle
    std::vector<Real> gradient_squared;          ///< σ = |∇n|² per angle
};

/**
 * Assemble ∇n and σ at one radius from the density channels and their radial
 * derivatives.
 *
 * @param channel_values      f_lm(r) — the true density channels, not r-weighted
 * @param channel_derivatives f'_lm(r)
 */
inline GradientAtRadius
build_gradient_at_radius(const std::vector<Real>& channel_values,
                         const std::vector<Real>& channel_derivatives,
                         Real radius,
                         const std::vector<std::vector<Real>>& harmonics,
                         const std::vector<std::vector<std::array<Real, 3>>>& harmonic_gradients) {
    int number_of_angles = static_cast<int>(harmonics.size());
    int number_of_lm = static_cast<int>(channel_values.size());

    GradientAtRadius result;
    result.radial_component.assign(number_of_angles, 0.0);
    result.tangential.assign(number_of_angles, {0.0, 0.0, 0.0});
    result.gradient_squared.assign(number_of_angles, 0.0);

    Real inverse_radius = 1.0 / radius;
    for (int angle = 0; angle < number_of_angles; angle++) {
        Real radial = 0.0;
        std::array<Real, 3> tangential{0.0, 0.0, 0.0};
        for (int lm = 0; lm < number_of_lm; lm++) {
            radial += channel_derivatives[lm] * harmonics[angle][lm];
            for (int axis = 0; axis < 3; axis++)
                tangential[axis] += channel_values[lm] * harmonic_gradients[angle][lm][axis];
        }
        for (int axis = 0; axis < 3; axis++)
            tangential[axis] *= inverse_radius;

        result.radial_component[angle] = radial;
        result.tangential[angle] = tangential;
        result.gradient_squared[angle] = radial * radial + tangential[0] * tangential[0] +
                                         tangential[1] * tangential[1] +
                                         tangential[2] * tangential[2];
    }
    return result;
}

} // namespace paw_one_center
} // namespace q3
