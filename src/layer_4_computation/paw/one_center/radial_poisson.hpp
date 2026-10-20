#pragma once

/**
 * Radial Poisson solve for one angular-momentum channel of a one-center
 * density.
 *
 * A one-center density is carried as lm-decomposed radial channels, and its
 * Hartree potential separates per channel into a pair of radial integrals:
 *
 *     V_L(r) = 8π/(2L+1) · [ r^(-(L+1)) ∫₀^r u_L(r') r'^L dr'
 *                          + r^L       ∫_r^∞ u_L(r') r'^(-(L+1)) dr' ]
 *
 * in Rydberg units (e² = 2), with `u_L(r) = r² ρ_L(r)` — the r-weighted form,
 * chosen to match how the partial waves are already stored (r·φ) and how
 * Quantum ESPRESSO carries its own lm-decomposed one-center densities, so no
 * factor of r² has to be inserted or removed at the boundary between modules.
 *
 * Quantum ESPRESSO solves the equivalent second-order ODE with a Numerov
 * tridiagonal scheme rather than these quadratures. The two converge to the
 * same answer; the direct form is used here because it can be checked against
 * closed-form solutions to the accuracy of the quadrature alone, which is what
 * makes this step gateable before any PAW physics exists. If the µRy gate
 * later shows a discretization difference on the coarser dataset grids,
 * switching to a Numerov solve is a contained change behind this same
 * signature.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {
namespace paw_one_center {

/**
 * Cumulative integral ∫₀^{r_i} g(r) dr on a non-uniform grid.
 *
 * Simpson over each pair of intervals, with a three-point half-step
 * (5,8,−1)/12 rule for the odd points in between, so every grid point carries
 * fourth-order accuracy rather than only the even ones. `values` are the
 * integrand samples; `measures` are dr/di.
 */
inline std::vector<Real> cumulative_integral(const std::vector<Real>& values,
                                             const std::vector<Real>& measures,
                                             int mesh_size) {
    std::vector<Real> weighted(mesh_size);
    for (int index = 0; index < mesh_size; index++)
        weighted[index] = values[index] * measures[index];

    std::vector<Real> cumulative(mesh_size, 0.0);
    if (mesh_size < 3)
        return cumulative;

    for (int index = 2; index < mesh_size; index += 2) {
        cumulative[index] =
            cumulative[index - 2] +
            (weighted[index - 2] + 4.0 * weighted[index - 1] + weighted[index]) / 3.0;
    }
    for (int index = 1; index < mesh_size; index += 2) {
        if (index + 1 < mesh_size) {
            cumulative[index] =
                cumulative[index - 1] +
                (5.0 * weighted[index - 1] + 8.0 * weighted[index] - weighted[index + 1]) / 12.0;
        } else {
            cumulative[index] =
                cumulative[index - 1] + (weighted[index - 1] + weighted[index]) / 2.0;
        }
    }
    return cumulative;
}

/**
 * Solve for the Hartree potential of one (L, M) density channel.
 *
 * @param angular_momentum        L of this channel
 * @param radial_grid             r values
 * @param radial_grid_derivative  dr/di
 * @param density_r_squared       u_L(r) = r² ρ_L(r)
 * @param mesh_size               points to use
 * @param potential               [out] V_L(r), resized to mesh_size
 */
inline void solve_radial_hartree(int angular_momentum,
                                 const std::vector<Real>& radial_grid,
                                 const std::vector<Real>& radial_grid_derivative,
                                 const std::vector<Real>& density_r_squared,
                                 int mesh_size,
                                 std::vector<Real>& potential) {
    potential.assign(mesh_size, 0.0);
    if (mesh_size < 3)
        return;

    std::vector<Real> inner_integrand(mesh_size);
    std::vector<Real> outer_integrand(mesh_size);
    for (int index = 0; index < mesh_size; index++) {
        Real radius = radial_grid[index];
        Real radius_to_l = std::pow(radius, angular_momentum);
        inner_integrand[index] = density_r_squared[index] * radius_to_l;
        outer_integrand[index] = density_r_squared[index] / (radius_to_l * radius);
    }

    std::vector<Real> inner =
        cumulative_integral(inner_integrand, radial_grid_derivative, mesh_size);
    std::vector<Real> outer_cumulative =
        cumulative_integral(outer_integrand, radial_grid_derivative, mesh_size);
    Real outer_total = outer_cumulative[mesh_size - 1];

    const Real prefactor = 2.0 * constants::four_pi / static_cast<Real>(2 * angular_momentum + 1);
    for (int index = 0; index < mesh_size; index++) {
        Real radius = radial_grid[index];
        Real radius_to_l = std::pow(radius, angular_momentum);
        Real outer = outer_total - outer_cumulative[index];
        potential[index] =
            prefactor * (inner[index] / (radius_to_l * radius) + radius_to_l * outer);
    }
}

/**
 * Hartree energy of a set of one-center density channels against their own
 * potentials: E_H = ½ Σ_lm ∫ V_lm(r) u_lm(r) dr.
 *
 * The one-half is the usual double-counting factor, applied once after the
 * channels are summed rather than inside each integral — a scalar pulled out of
 * a sum, so the two placements differ only in operation count.
 */
inline Real radial_hartree_energy(const std::vector<std::vector<Real>>& potential_channels,
                                  const std::vector<std::vector<Real>>& density_channels,
                                  const std::vector<Real>& radial_grid_derivative,
                                  int mesh_size) {
    Real energy = 0.0;
    for (size_t channel = 0; channel < potential_channels.size(); channel++) {
        std::vector<Real> integrand(mesh_size);
        for (int index = 0; index < mesh_size; index++) {
            integrand[index] =
                potential_channels[channel][index] * density_channels[channel][index];
        }
        auto cumulative = cumulative_integral(integrand, radial_grid_derivative, mesh_size);
        energy += cumulative[mesh_size - 1];
    }
    return 0.5 * energy;
}

} // namespace paw_one_center
} // namespace q3
