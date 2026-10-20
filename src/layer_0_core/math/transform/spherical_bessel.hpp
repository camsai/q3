#pragma once

/**
 * Spherical Bessel transform (Fourier-Bessel / Hankel transform).
 *
 * Computes f(q) = 4π ∫₀^∞ f(r) · j_l(qr) · r · dr  for each q in q_points,
 * evaluated numerically via Simpson's rule on the provided grid.
 *
 * This is the radial part of a 3D Fourier transform in spherical coordinates.
 * Used to tabulate beta projectors, atomic charge density, and core charge
 * density from their real-space representations onto reciprocal space.
 */

#include <vector>

#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/spherical/bessel.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Spherical Bessel transform of a function onto a uniform q-grid.
 *
 * @param grid                  r values on the (typically logarithmic) mesh
 * @param integration_factors   dr/di integration factors for Simpson's rule
 * @param function              f(r) to transform
 * @param degree                l for the spherical Bessel function j_l
 * @param mesh_size             number of grid points to integrate over
 * @param q_points              q values at which to evaluate the transform
 * @param fourier_transform     [out] f(q) result, resized to q_points.size()
 */
inline void transform_spherical_bessel(const std::vector<Real>& grid,
                                       const std::vector<Real>& integration_factors,
                                       const std::vector<Real>& function,
                                       int degree,
                                       int mesh_size,
                                       const std::vector<Real>& q_points,
                                       std::vector<Real>& fourier_transform) {
    fourier_transform.resize(q_points.size());
    std::vector<Real> integrand(mesh_size);
    std::vector<Real> bessel_values(mesh_size);
    for (size_t q_index = 0; q_index < q_points.size(); q_index++) {
        Real q_value = q_points[q_index];
        spherical_bessel_batch(degree, q_value, grid.data(), mesh_size, bessel_values.data());
        for (int grid_index = 0; grid_index < mesh_size; grid_index++) {
            integrand[grid_index] =
                function[grid_index] * grid[grid_index] * bessel_values[grid_index];
        }
        fourier_transform[q_index] =
            constants::four_pi * simpson_rule(integrand, integration_factors, mesh_size);
    }
}

} // namespace q3
