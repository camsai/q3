#pragma once

/**
 * One-center energy and potential: E¹ − Ẽ¹ and the potentials D_ij is built
 * from.
 *
 * The two-branch decomposition is the PAW formalism's own (Blöchl, PRB 50,
 * 17953 (1994)): an all-electron and a pseudo branch, each contributing
 * Hartree plus exchange-correlation, combined with opposite signs. What
 * differs between the branches is only which density and which core density
 * each uses —
 *
 *     all-electron:  n¹,            core = the frozen all-electron core
 *     pseudo:        ñ¹ + n̂,        core = the *pseudized* core
 *
 * — and getting the second of those wrong (using one core for both) shifts
 * every element with a core correction by a smooth, plausible amount.
 *
 * The result is reported as four separate numbers rather than one total, on
 * purpose: they are the same four-way (Hartree|XC x AE|PS) per-atom split
 * Quantum ESPRESSO reports, so a µRy-gate failure can be bisected against the
 * reference term by term instead of staring at two totals.
 *
 * Hartree separates per lm channel and is solved exactly; XC does not, because
 * it is nonlinear in the density. So XC is evaluated by summing the channels
 * onto real directions on the angular grid, calling the functional pointwise
 * there, and projecting the potential back onto Y_LM — which is what the
 * quadrature's exactness gate exists to make safe.
 */

#include <array>
#include <cmath>
#include <vector>

#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/paw/one_center/angular_grid.hpp"
#include "layer_4_computation/paw/one_center/densities.hpp"
#include "layer_4_computation/paw/one_center/gradient.hpp"
#include "layer_4_computation/paw/one_center/radial_poisson.hpp"

namespace q3 {
namespace paw_one_center {

/// Real spherical harmonics at every angular grid point, [point][lm].
inline std::vector<std::vector<Real>>
tabulate_harmonics(const std::vector<AngularPoint>& angular_grid, int number_of_lm) {
    int maximum_angular_momentum = 0;
    while ((maximum_angular_momentum + 1) * (maximum_angular_momentum + 1) < number_of_lm)
        maximum_angular_momentum++;

    std::vector<std::vector<Real>> harmonics(angular_grid.size(),
                                             std::vector<Real>(number_of_lm, 0.0));
    for (size_t point = 0; point < angular_grid.size(); point++) {
        Vector3D direction{angular_grid[point].x, angular_grid[point].y, angular_grid[point].z};
        for (int degree = 0; degree <= maximum_angular_momentum; degree++) {
            for (int order : harmonic_order_sequence(degree)) {
                int index = harmonic_combined_degree_order_index(degree, order);
                if (index < number_of_lm)
                    harmonics[point][index] =
                        compute_real_spherical_harmonic(degree, order, direction);
            }
        }
    }
    return harmonics;
}

/**
 * The extra terms a dataset's convention asks the one-center engine to
 * include.
 *
 * Default-constructed means "none", which is the absolute formulation UPF-PAW
 * datasets need. Anything switched on here belongs to an atomic-referenced
 * convention, whose datasets have had the corresponding piece removed from
 * their tabulated D_ion at generation time. Switching one on for a UPF dataset
 * double-counts it: doing so moved diamond from 1.98 to 5221 meV/atom against
 * QE, which is how the separation was established rather than assumed.
 *
 * Filled by `PawOneCenterConvention`, never by callers directly.
 */
struct OneCenterOptions {
    /**
     * Whether E_xc[n_core] is removed from the all-electron branch.
     *
     * That branch evaluates E_xc at n_valence + n_core, which necessarily
     * contains the core's interaction with itself — a property of the frozen
     * core, not of this calculation. An atomic-referenced convention removes
     * it as a setup-time constant. A dataset may state a related header value,
     * but one covering only the remainder from outside the augmentation sphere
     * cannot stand in for the term, which therefore has to be computed rather
     * than read.
     */
    bool subtract_core_exchange_correlation{false};
};

/// One branch (all-electron or pseudo) of the one-center terms.
struct OneCenterBranch {
    Real hartree_energy{};
    Real xc_energy{};
    /// v_H + v_rho, lm-decomposed — what D_ij's local term integrates against.
    std::vector<std::vector<Real>> potential_lm;

    /**
     * Gradient-correction weights, empty for LDA.
     *
     * A GGA energy depends on ∇n, so its derivative carries a term
     * 2∫v_σ ∇n·∇δn. Rather than take the divergence ∇·(2v_σ∇n) — which would
     * introduce a numerical error the D ≡ ∂E/∂ρ identity cannot absorb — the
     * term is integrated by parts against the perturbation's own channels:
     *
     *     2∫ v_σ ∇n·∇δn dV = Σ_lm ∫ dr [ r² g'_lm P_lm + g_lm Q_lm ]
     *
     * with δn = Σ_lm g_lm(r) Y_lm(Ω), and
     *
     *     P_lm(r) = ∫dΩ 2 v_σ (∂n/∂r) Y_lm
     *     Q_lm(r) = ∫dΩ 2 v_σ (∇_Ω n · ∇_Ω Y_lm)
     *
     * so the identity holds by construction rather than by cancellation.
     */
    std::vector<std::vector<Real>> gradient_radial_weight_lm;  ///< P_lm
    std::vector<std::vector<Real>> gradient_angular_weight_lm; ///< Q_lm

    [[nodiscard]] bool has_gradient_correction() const {
        return !gradient_radial_weight_lm.empty();
    }
};

/// Both branches, kept separate so a gate failure can be localized.
struct OneCenterEnergy {
    OneCenterBranch all_electron;
    OneCenterBranch pseudo;

    /// E¹ − Ẽ¹, the quantity that enters the total energy.
    [[nodiscard]] Real total() const {
        return (all_electron.hartree_energy + all_electron.xc_energy) -
               (pseudo.hartree_energy + pseudo.xc_energy);
    }
};

/**
 * Exchange-correlation for one branch on the radial × angular grid.
 *
 * @param density_channels  u_lm(r) = r² ρ_lm(r)
 * @param core_density      spherical ρ_core(r), added to the total before the
 *                          functional is evaluated (never to the Hartree term)
 */
inline void compute_one_center_xc(const std::vector<std::vector<Real>>& density_channels,
                                  const std::vector<Real>& core_density,
                                  const std::vector<Real>& radial_grid,
                                  const std::vector<Real>& radial_grid_derivative,
                                  int mesh_size,
                                  const std::vector<AngularPoint>& angular_grid,
                                  const std::vector<std::vector<Real>>& harmonics,
                                  const XCFunctional& functional,
                                  Real& energy,
                                  std::vector<std::vector<Real>>& potential_lm) {
    int number_of_lm = static_cast<int>(density_channels.size());
    int number_of_angles = static_cast<int>(angular_grid.size());

    potential_lm.assign(number_of_lm, std::vector<Real>(mesh_size, 0.0));
    energy = 0.0;

    std::vector<Real> density_at_angles(number_of_angles);
    std::vector<Real> xc_potential(number_of_angles);
    std::vector<Real> xc_energy_density(number_of_angles);
    std::vector<Real> radial_energy(mesh_size, 0.0);
    std::vector<std::vector<Real>> radial_potential(number_of_lm,
                                                    std::vector<Real>(mesh_size, 0.0));

    for (int point = 0; point < mesh_size; point++) {
        Real radius = radial_grid[point];
        Real inverse_r_squared = 1.0 / (radius * radius);
        Real core = core_density.empty() ? 0.0 : core_density[point];

        for (int angle = 0; angle < number_of_angles; angle++) {
            Real value = 0.0;
            for (int lm = 0; lm < number_of_lm; lm++)
                value += density_channels[lm][point] * harmonics[angle][lm];
            density_at_angles[angle] = value * inverse_r_squared + core;
        }

        functional.compute(density_at_angles, xc_potential, xc_energy_density);

        /**
         * ε_xc is per particle, so the energy density is ε·n; the r² restores
         * the volume element the r-weighted storage already carries elsewhere.
         */
        Real energy_at_radius = 0.0;
        for (int angle = 0; angle < number_of_angles; angle++) {
            energy_at_radius +=
                angular_grid[angle].weight * xc_energy_density[angle] * density_at_angles[angle];
        }
        radial_energy[point] = energy_at_radius * radius * radius;

        for (int lm = 0; lm < number_of_lm; lm++) {
            Real projected = 0.0;
            for (int angle = 0; angle < number_of_angles; angle++) {
                projected +=
                    angular_grid[angle].weight * xc_potential[angle] * harmonics[angle][lm];
            }
            radial_potential[lm][point] = projected;
        }
    }

    auto cumulative = cumulative_integral(radial_energy, radial_grid_derivative, mesh_size);
    energy = cumulative[mesh_size - 1];
    potential_lm = radial_potential;
}

/**
 * Gradient-corrected exchange-correlation on the radial × angular grid.
 *
 * Same shape as the LDA routine, plus σ = |∇n|² and the by-parts weights the
 * D_ij gradient term needs. The core density is spherical, so it contributes
 * to the radial derivative through its own slope and nothing to the tangential
 * part.
 */
inline void compute_one_center_xc_gradient_corrected(
    const std::vector<std::vector<Real>>& density_channels,
    const std::vector<Real>& core_density,
    const std::vector<Real>& radial_grid,
    const std::vector<Real>& radial_grid_derivative,
    int mesh_size,
    const std::vector<AngularPoint>& angular_grid,
    const std::vector<std::vector<Real>>& harmonics,
    const std::vector<std::vector<std::array<Real, 3>>>& harmonic_gradients,
    const XCFunctional& functional,
    Real& energy,
    std::vector<std::vector<Real>>& potential_lm,
    std::vector<std::vector<Real>>& radial_weight_lm,
    std::vector<std::vector<Real>>& angular_weight_lm) {
    int number_of_lm = static_cast<int>(density_channels.size());
    int number_of_angles = static_cast<int>(angular_grid.size());

    /**
     * The true density channels f_lm = u_lm/r², and their radial derivatives.
     * The core is spherical, so it enters only the L = 0 channel — through
     * Y_00, hence the √(4π).
     */
    std::vector<std::vector<Real>> channel_values(number_of_lm, std::vector<Real>(mesh_size, 0.0));
    for (int lm = 0; lm < number_of_lm; lm++) {
        for (int point = 0; point < mesh_size; point++) {
            Real radius = radial_grid[point];
            channel_values[lm][point] = density_channels[lm][point] / (radius * radius);
        }
    }
    if (!core_density.empty()) {
        const Real inverse_y00 = std::sqrt(constants::four_pi);
        for (int point = 0; point < mesh_size; point++)
            channel_values[0][point] += core_density[point] * inverse_y00;
    }

    std::vector<std::vector<Real>> channel_derivatives(number_of_lm);
    for (int lm = 0; lm < number_of_lm; lm++)
        channel_derivatives[lm] =
            radial_derivative(channel_values[lm], radial_grid_derivative, mesh_size);

    potential_lm.assign(number_of_lm, std::vector<Real>(mesh_size, 0.0));
    radial_weight_lm.assign(number_of_lm, std::vector<Real>(mesh_size, 0.0));
    angular_weight_lm.assign(number_of_lm, std::vector<Real>(mesh_size, 0.0));

    std::vector<Real> density_at_angles(number_of_angles);
    std::vector<Real> xc_potential(number_of_angles);
    std::vector<Real> gradient_correction(number_of_angles);
    std::vector<Real> xc_energy_density(number_of_angles);
    std::vector<Real> radial_energy(mesh_size, 0.0);

    std::vector<Real> values_at_radius(number_of_lm);
    std::vector<Real> derivatives_at_radius(number_of_lm);

    for (int point = 0; point < mesh_size; point++) {
        Real radius = radial_grid[point];
        for (int lm = 0; lm < number_of_lm; lm++) {
            values_at_radius[lm] = channel_values[lm][point];
            derivatives_at_radius[lm] = channel_derivatives[lm][point];
        }

        auto gradient = build_gradient_at_radius(
            values_at_radius, derivatives_at_radius, radius, harmonics, harmonic_gradients);

        for (int angle = 0; angle < number_of_angles; angle++) {
            Real value = 0.0;
            for (int lm = 0; lm < number_of_lm; lm++)
                value += values_at_radius[lm] * harmonics[angle][lm];
            density_at_angles[angle] = value;
        }

        functional.compute_gradient_corrected(density_at_angles,
                                              gradient.gradient_squared,
                                              xc_potential,
                                              gradient_correction,
                                              xc_energy_density);

        Real energy_at_radius = 0.0;
        for (int angle = 0; angle < number_of_angles; angle++) {
            energy_at_radius +=
                angular_grid[angle].weight * xc_energy_density[angle] * density_at_angles[angle];
        }
        radial_energy[point] = energy_at_radius * radius * radius;

        for (int lm = 0; lm < number_of_lm; lm++) {
            Real local = 0.0;
            Real radial_weight = 0.0;
            Real angular_weight = 0.0;
            for (int angle = 0; angle < number_of_angles; angle++) {
                Real weight = angular_grid[angle].weight;
                local += weight * xc_potential[angle] * harmonics[angle][lm];

                Real two_v_sigma = 2.0 * gradient_correction[angle];
                radial_weight +=
                    weight * two_v_sigma * gradient.radial_component[angle] * harmonics[angle][lm];

                Real tangential_dot = 0.0;
                for (int axis = 0; axis < 3; axis++) {
                    tangential_dot +=
                        gradient.tangential[angle][axis] * harmonic_gradients[angle][lm][axis];
                }
                /**
                 * Bookkeeping worth spelling out, because getting it wrong is
                 * an error of r⁴. The stored tangential field is ∇_Ω n / r,
                 * and the perturbation's tangential field is
                 * (1/r) Σ g_lm ∇_Ω Y_lm, so their product carries 1/r; the
                 * volume element contributes r². Net, the weight paired with
                 * g_lm under ∫dr is r·S_lm, not S_lm/r.
                 */
                angular_weight += weight * two_v_sigma * tangential_dot * radius;
            }
            potential_lm[lm][point] = local;
            radial_weight_lm[lm][point] = radial_weight;
            angular_weight_lm[lm][point] = angular_weight;
        }
    }

    auto cumulative = cumulative_integral(radial_energy, radial_grid_derivative, mesh_size);
    energy = cumulative[mesh_size - 1];
}

/**
 * Spin-polarized exchange-correlation for one branch (LDA).
 *
 * Same grid and same projection as the unpolarized routine; the difference is
 * that the functional sees two densities and returns two potentials, so the
 * D_ij of each channel integrates against its own v_xc. The frozen core is
 * unpolarized and is split evenly between the channels, which is what makes
 * n↑ + n↓ reproduce the unpolarized total exactly.
 *
 * @param density_channels_by_spin  [spin][lm][radial point], r-weighted
 * @param potential_lm_by_spin      out, [spin][lm][radial point]
 */
inline void compute_one_center_xc_spin(
    const std::vector<std::vector<std::vector<Real>>>& density_channels_by_spin,
    const std::vector<Real>& core_density,
    const std::vector<Real>& radial_grid,
    const std::vector<Real>& radial_grid_derivative,
    int mesh_size,
    const std::vector<AngularPoint>& angular_grid,
    const std::vector<std::vector<Real>>& harmonics,
    const XCFunctional& functional,
    Real& energy,
    std::vector<std::vector<std::vector<Real>>>& potential_lm_by_spin) {
    int number_of_lm = static_cast<int>(density_channels_by_spin[0].size());
    int number_of_angles = static_cast<int>(angular_grid.size());

    potential_lm_by_spin.assign(
        2, std::vector<std::vector<Real>>(number_of_lm, std::vector<Real>(mesh_size, 0.0)));
    energy = 0.0;

    std::vector<Real> density_up(number_of_angles), density_down(number_of_angles);
    std::vector<Real> potential_up(number_of_angles), potential_down(number_of_angles);
    std::vector<Real> xc_energy_density(number_of_angles);
    std::vector<Real> radial_energy(mesh_size, 0.0);

    for (int point = 0; point < mesh_size; point++) {
        Real radius = radial_grid[point];
        Real inverse_r_squared = 1.0 / (radius * radius);
        Real half_core = core_density.empty() ? 0.0 : 0.5 * core_density[point];

        for (int angle = 0; angle < number_of_angles; angle++) {
            Real up = 0.0;
            Real down = 0.0;
            for (int lm = 0; lm < number_of_lm; lm++) {
                up += density_channels_by_spin[0][lm][point] * harmonics[angle][lm];
                down += density_channels_by_spin[1][lm][point] * harmonics[angle][lm];
            }
            density_up[angle] = up * inverse_r_squared + half_core;
            density_down[angle] = down * inverse_r_squared + half_core;
        }

        functional.compute_spin(
            density_up, density_down, potential_up, potential_down, xc_energy_density);

        /**
         * The functional returns ε per particle of the *total* density, matching
         * the unpolarized convention, so the energy weight is n↑ + n↓.
         */
        Real energy_at_radius = 0.0;
        for (int angle = 0; angle < number_of_angles; angle++) {
            energy_at_radius += angular_grid[angle].weight * xc_energy_density[angle] *
                                (density_up[angle] + density_down[angle]);
        }
        radial_energy[point] = energy_at_radius * radius * radius;

        for (int lm = 0; lm < number_of_lm; lm++) {
            Real projected_up = 0.0;
            Real projected_down = 0.0;
            for (int angle = 0; angle < number_of_angles; angle++) {
                Real weight = angular_grid[angle].weight * harmonics[angle][lm];
                projected_up += weight * potential_up[angle];
                projected_down += weight * potential_down[angle];
            }
            potential_lm_by_spin[0][lm][point] = projected_up;
            potential_lm_by_spin[1][lm][point] = projected_down;
        }
    }

    auto cumulative = cumulative_integral(radial_energy, radial_grid_derivative, mesh_size);
    energy = cumulative[mesh_size - 1];
}

/**
 * Spin-polarized gradient-corrected exchange-correlation for one branch.
 *
 * The by-parts construction of the unpolarized routine carries over with one
 * change: a GGA's dependence on ∇n_σ runs through both σ_σσ and the cross term
 * σ_↑↓, so the field conjugate to ∇δn_σ is
 *
 *     W_σ = 2 v_{σσ} ∇n_σ + v_{↑↓} ∇n_{σ'}
 *
 * rather than 2 v_σ ∇n.
 *
 * It reduces exactly, not just numerically: at n↑ = n↓ = n/2 each channel's
 * gradient is half the total, and the functional's own spin scaling supplies
 * the rest — PBE exchange has no cross term (v_{↑↓} = 0) with
 * v_{↑↑} = 2 v_σ^unpol, while PBE correlation has v_{↑↓} = 2 v_{↑↑}. Both land
 * on 2 v_σ^unpol ∇n. `OneCenterEnergySpinTest` gates the resulting energy.
 */
inline void compute_one_center_xc_gradient_corrected_spin(
    const std::vector<std::vector<std::vector<Real>>>& density_channels_by_spin,
    const std::vector<Real>& core_density,
    const std::vector<Real>& radial_grid,
    const std::vector<Real>& radial_grid_derivative,
    int mesh_size,
    const std::vector<AngularPoint>& angular_grid,
    const std::vector<std::vector<Real>>& harmonics,
    const std::vector<std::vector<std::array<Real, 3>>>& harmonic_gradients,
    const XCFunctional& functional,
    Real& energy,
    std::vector<std::vector<std::vector<Real>>>& potential_lm_by_spin,
    std::vector<std::vector<std::vector<Real>>>& radial_weight_lm_by_spin,
    std::vector<std::vector<std::vector<Real>>>& angular_weight_lm_by_spin) {
    int number_of_lm = static_cast<int>(density_channels_by_spin[0].size());
    int number_of_angles = static_cast<int>(angular_grid.size());

    // f_lm = u_lm/r² per spin, with half the spherical core in each L = 0 channel.
    std::vector<std::vector<std::vector<Real>>> channel_values(
        2, std::vector<std::vector<Real>>(number_of_lm, std::vector<Real>(mesh_size, 0.0)));
    for (int spin = 0; spin < 2; spin++) {
        for (int lm = 0; lm < number_of_lm; lm++) {
            for (int point = 0; point < mesh_size; point++) {
                Real radius = radial_grid[point];
                channel_values[spin][lm][point] =
                    density_channels_by_spin[spin][lm][point] / (radius * radius);
            }
        }
        if (!core_density.empty()) {
            const Real inverse_y00 = std::sqrt(constants::four_pi);
            for (int point = 0; point < mesh_size; point++)
                channel_values[spin][0][point] += 0.5 * core_density[point] * inverse_y00;
        }
    }

    std::vector<std::vector<std::vector<Real>>> channel_derivatives(
        2, std::vector<std::vector<Real>>(number_of_lm));
    for (int spin = 0; spin < 2; spin++)
        for (int lm = 0; lm < number_of_lm; lm++)
            channel_derivatives[spin][lm] =
                radial_derivative(channel_values[spin][lm], radial_grid_derivative, mesh_size);

    auto empty_by_spin = [&]() {
        return std::vector<std::vector<std::vector<Real>>>(
            2, std::vector<std::vector<Real>>(number_of_lm, std::vector<Real>(mesh_size, 0.0)));
    };
    potential_lm_by_spin = empty_by_spin();
    radial_weight_lm_by_spin = empty_by_spin();
    angular_weight_lm_by_spin = empty_by_spin();

    std::vector<Real> density_up(number_of_angles), density_down(number_of_angles);
    std::vector<Real> sigma_uu(number_of_angles), sigma_ud(number_of_angles),
        sigma_dd(number_of_angles);
    std::vector<Real> potential_up(number_of_angles), potential_down(number_of_angles);
    std::vector<Real> gradient_uu(number_of_angles), gradient_ud(number_of_angles),
        gradient_dd(number_of_angles);
    std::vector<Real> xc_energy_density(number_of_angles);
    std::vector<Real> radial_energy(mesh_size, 0.0);

    std::vector<Real> values_at_radius(number_of_lm), derivatives_at_radius(number_of_lm);

    for (int point = 0; point < mesh_size; point++) {
        Real radius = radial_grid[point];

        std::vector<GradientAtRadius> gradient(2);
        for (int spin = 0; spin < 2; spin++) {
            for (int lm = 0; lm < number_of_lm; lm++) {
                values_at_radius[lm] = channel_values[spin][lm][point];
                derivatives_at_radius[lm] = channel_derivatives[spin][lm][point];
            }
            gradient[spin] = build_gradient_at_radius(
                values_at_radius, derivatives_at_radius, radius, harmonics, harmonic_gradients);
        }

        for (int angle = 0; angle < number_of_angles; angle++) {
            Real up = 0.0;
            Real down = 0.0;
            for (int lm = 0; lm < number_of_lm; lm++) {
                up += channel_values[0][lm][point] * harmonics[angle][lm];
                down += channel_values[1][lm][point] * harmonics[angle][lm];
            }
            density_up[angle] = up;
            density_down[angle] = down;

            sigma_uu[angle] = gradient[0].gradient_squared[angle];
            sigma_dd[angle] = gradient[1].gradient_squared[angle];
            Real cross = gradient[0].radial_component[angle] * gradient[1].radial_component[angle];
            for (int axis = 0; axis < 3; axis++)
                cross += gradient[0].tangential[angle][axis] * gradient[1].tangential[angle][axis];
            sigma_ud[angle] = cross;
        }

        functional.compute_gradient_corrected_spin(density_up,
                                                   density_down,
                                                   sigma_uu,
                                                   sigma_ud,
                                                   sigma_dd,
                                                   potential_up,
                                                   potential_down,
                                                   gradient_uu,
                                                   gradient_ud,
                                                   gradient_dd,
                                                   xc_energy_density);

        Real energy_at_radius = 0.0;
        for (int angle = 0; angle < number_of_angles; angle++) {
            energy_at_radius += angular_grid[angle].weight * xc_energy_density[angle] *
                                (density_up[angle] + density_down[angle]);
        }
        radial_energy[point] = energy_at_radius * radius * radius;

        for (int spin = 0; spin < 2; spin++) {
            const auto& potential = spin == 0 ? potential_up : potential_down;
            const auto& same_spin_gradient = spin == 0 ? gradient_uu : gradient_dd;
            const auto& own = gradient[spin];
            const auto& other = gradient[1 - spin];

            for (int lm = 0; lm < number_of_lm; lm++) {
                Real local = 0.0;
                Real radial_weight = 0.0;
                Real angular_weight = 0.0;
                for (int angle = 0; angle < number_of_angles; angle++) {
                    Real weight = angular_grid[angle].weight;
                    local += weight * potential[angle] * harmonics[angle][lm];

                    // W_σ = 2 v_σσ ∇n_σ + v_↑↓ ∇n_σ'
                    Real conjugate_radial =
                        2.0 * same_spin_gradient[angle] * own.radial_component[angle] +
                        gradient_ud[angle] * other.radial_component[angle];
                    radial_weight += weight * conjugate_radial * harmonics[angle][lm];

                    Real tangential_dot = 0.0;
                    for (int axis = 0; axis < 3; axis++) {
                        Real conjugate_tangential =
                            2.0 * same_spin_gradient[angle] * own.tangential[angle][axis] +
                            gradient_ud[angle] * other.tangential[angle][axis];
                        tangential_dot +=
                            conjugate_tangential * harmonic_gradients[angle][lm][axis];
                    }
                    /**
                     * Same r-bookkeeping as the unpolarized routine: the stored
                     * tangential field carries 1/r and the perturbation's own
                     * carries another, against r² from the volume element, so
                     * the weight paired with g_lm under ∫dr is r·S_lm.
                     */
                    angular_weight += weight * tangential_dot * radius;
                }
                potential_lm_by_spin[spin][lm][point] = local;
                radial_weight_lm_by_spin[spin][lm][point] = radial_weight;
                angular_weight_lm_by_spin[spin][lm][point] = angular_weight;
            }
        }
    }

    auto cumulative = cumulative_integral(radial_energy, radial_grid_derivative, mesh_size);
    energy = cumulative[mesh_size - 1];
}

/**
 * Assemble both branches: Hartree and XC energies, and the summed potential
 * each branch's D_ij contribution integrates against.
 */
inline OneCenterEnergy compute_one_center_energy(
    const PawPseudopotential& pseudopotential,
    const OneCenterDensities& densities,
    const std::vector<AngularPoint>& angular_grid,
    const std::vector<std::vector<Real>>& harmonics,
    const XCFunctional& functional,
    const std::vector<std::vector<std::array<Real, 3>>>* harmonic_gradients = nullptr,
    const OneCenterOptions& options = {}) {
    OneCenterEnergy result;
    int mesh_size = pseudopotential.mesh_size;
    int number_of_lm = densities.number_of_lm;

    auto build_branch = [&](const std::vector<std::vector<Real>>& channels,
                            const std::vector<Real>& core_density,
                            bool subtract_core_exchange_correlation) {
        OneCenterBranch branch;

        std::vector<std::vector<Real>> hartree_potential(number_of_lm);
        for (int lm = 0; lm < number_of_lm; lm++) {
            solve_radial_hartree(angular_momentum_of_lm(lm),
                                 pseudopotential.radial_grid,
                                 pseudopotential.radial_grid_derivative,
                                 channels[lm],
                                 mesh_size,
                                 hartree_potential[lm]);
        }
        branch.hartree_energy = radial_hartree_energy(
            hartree_potential, channels, pseudopotential.radial_grid_derivative, mesh_size);

        std::vector<std::vector<Real>> xc_potential;
        if (harmonic_gradients != nullptr) {
            compute_one_center_xc_gradient_corrected(channels,
                                                     core_density,
                                                     pseudopotential.radial_grid,
                                                     pseudopotential.radial_grid_derivative,
                                                     mesh_size,
                                                     angular_grid,
                                                     harmonics,
                                                     *harmonic_gradients,
                                                     functional,
                                                     branch.xc_energy,
                                                     xc_potential,
                                                     branch.gradient_radial_weight_lm,
                                                     branch.gradient_angular_weight_lm);
        } else {
            compute_one_center_xc(channels,
                                  core_density,
                                  pseudopotential.radial_grid,
                                  pseudopotential.radial_grid_derivative,
                                  mesh_size,
                                  angular_grid,
                                  harmonics,
                                  functional,
                                  branch.xc_energy,
                                  xc_potential);
        }

        /**
         * Only the energy is corrected here, never `potential_lm`: E_xc[n_core]
         * does not depend on the valence occupancies, so its derivative — which
         * is what D_ij is — is identically zero, and removing it from the
         * potential too would break D ≡ ∂E/∂ρ_ij.
         */
        if (subtract_core_exchange_correlation && !core_density.empty()) {
            std::vector<std::vector<Real>> no_valence(number_of_lm,
                                                      std::vector<Real>(mesh_size, 0.0));
            std::vector<std::vector<Real>> discarded_potential;
            Real core_only_energy = 0.0;
            if (harmonic_gradients != nullptr) {
                std::vector<std::vector<Real>> discarded_radial_weight;
                std::vector<std::vector<Real>> discarded_angular_weight;
                compute_one_center_xc_gradient_corrected(no_valence,
                                                         core_density,
                                                         pseudopotential.radial_grid,
                                                         pseudopotential.radial_grid_derivative,
                                                         mesh_size,
                                                         angular_grid,
                                                         harmonics,
                                                         *harmonic_gradients,
                                                         functional,
                                                         core_only_energy,
                                                         discarded_potential,
                                                         discarded_radial_weight,
                                                         discarded_angular_weight);
            } else {
                compute_one_center_xc(no_valence,
                                      core_density,
                                      pseudopotential.radial_grid,
                                      pseudopotential.radial_grid_derivative,
                                      mesh_size,
                                      angular_grid,
                                      harmonics,
                                      functional,
                                      core_only_energy,
                                      discarded_potential);
            }
            branch.xc_energy -= core_only_energy;
        }

        branch.potential_lm.assign(number_of_lm, std::vector<Real>(mesh_size, 0.0));
        for (int lm = 0; lm < number_of_lm; lm++) {
            for (int point = 0; point < mesh_size; point++) {
                branch.potential_lm[lm][point] =
                    hartree_potential[lm][point] + xc_potential[lm][point];
            }
        }
        return branch;
    };

    result.all_electron = build_branch(densities.all_electron,
                                       pseudopotential.charge_core_all_electron.radial_values,
                                       options.subtract_core_exchange_correlation);
    /**
     * The pseudo branch keeps its core-core term: its "core" is the shallow
     * pseudized partial core, and an atomic-referenced convention removes the
     * core self-interaction from the all-electron branch alone.
     */
    result.pseudo =
        build_branch(densities.pseudo, pseudopotential.charge_core.radial_values, false);
    return result;
}

/**
 * Both branches, spin-polarized: one `OneCenterEnergy` per spin channel.
 *
 * Hartree is a functional of the *total* density alone, so it is computed once
 * from n↑ + n↓ and its potential handed to both channels — that is the physics,
 * not an approximation. Only XC splits. To keep `total()` summable over the
 * returned vector without double counting, the Hartree energy is recorded on
 * channel 0 and left at zero on channel 1, while both channels' `potential_lm`
 * carry the full Hartree potential because each channel's D_ij needs it.
 *
 * @param densities_by_spin  one `OneCenterDensities` per channel, same lm and mesh
 */
inline std::vector<OneCenterEnergy> compute_one_center_energy_spin(
    const PawPseudopotential& pseudopotential,
    const std::vector<OneCenterDensities>& densities_by_spin,
    const std::vector<AngularPoint>& angular_grid,
    const std::vector<std::vector<Real>>& harmonics,
    const XCFunctional& functional,
    const std::vector<std::vector<std::array<Real, 3>>>* harmonic_gradients = nullptr,
    const OneCenterOptions& options = {}) {
    std::vector<OneCenterEnergy> result(2);
    int mesh_size = pseudopotential.mesh_size;
    int number_of_lm = densities_by_spin[0].number_of_lm;

    auto build_branch = [&](const std::vector<std::vector<Real>>& channels_up,
                            const std::vector<std::vector<Real>>& channels_down,
                            const std::vector<Real>& core_density,
                            bool subtract_core_exchange_correlation,
                            OneCenterBranch& branch_up,
                            OneCenterBranch& branch_down) {
        std::vector<std::vector<Real>> total(number_of_lm, std::vector<Real>(mesh_size, 0.0));
        for (int lm = 0; lm < number_of_lm; lm++)
            for (int point = 0; point < mesh_size; point++)
                total[lm][point] = channels_up[lm][point] + channels_down[lm][point];

        std::vector<std::vector<Real>> hartree_potential(number_of_lm);
        for (int lm = 0; lm < number_of_lm; lm++) {
            solve_radial_hartree(angular_momentum_of_lm(lm),
                                 pseudopotential.radial_grid,
                                 pseudopotential.radial_grid_derivative,
                                 total[lm],
                                 mesh_size,
                                 hartree_potential[lm]);
        }
        branch_up.hartree_energy = radial_hartree_energy(
            hartree_potential, total, pseudopotential.radial_grid_derivative, mesh_size);
        branch_down.hartree_energy = 0.0;

        std::vector<std::vector<std::vector<Real>>> density_by_spin{channels_up, channels_down};
        std::vector<std::vector<std::vector<Real>>> xc_potential_by_spin;
        Real xc_energy = 0.0;
        if (harmonic_gradients != nullptr) {
            std::vector<std::vector<std::vector<Real>>> radial_weight_by_spin;
            std::vector<std::vector<std::vector<Real>>> angular_weight_by_spin;
            compute_one_center_xc_gradient_corrected_spin(density_by_spin,
                                                          core_density,
                                                          pseudopotential.radial_grid,
                                                          pseudopotential.radial_grid_derivative,
                                                          mesh_size,
                                                          angular_grid,
                                                          harmonics,
                                                          *harmonic_gradients,
                                                          functional,
                                                          xc_energy,
                                                          xc_potential_by_spin,
                                                          radial_weight_by_spin,
                                                          angular_weight_by_spin);
            branch_up.gradient_radial_weight_lm = radial_weight_by_spin[0];
            branch_up.gradient_angular_weight_lm = angular_weight_by_spin[0];
            branch_down.gradient_radial_weight_lm = radial_weight_by_spin[1];
            branch_down.gradient_angular_weight_lm = angular_weight_by_spin[1];
        } else {
            compute_one_center_xc_spin(density_by_spin,
                                       core_density,
                                       pseudopotential.radial_grid,
                                       pseudopotential.radial_grid_derivative,
                                       mesh_size,
                                       angular_grid,
                                       harmonics,
                                       functional,
                                       xc_energy,
                                       xc_potential_by_spin);
        }
        branch_up.xc_energy = xc_energy;
        branch_down.xc_energy = 0.0;

        /**
         * The core-only term goes through the spin routines as well.
         *
         * The frozen core carries no moment, so this is the same number the
         * unpolarized routine would give — but only if it is evaluated with an
         * unpolarized handle, and the handle here is polarized. Nickel's core
         * exchange-correlation is about 106 Ry, so taking the wrong branch put
         * the atomic reference at −107.95 Ry instead of −1.58 Ry, which then
         * appeared as a 113 Ry error in the SCF total.
         */
        if (subtract_core_exchange_correlation && !core_density.empty()) {
            std::vector<std::vector<std::vector<Real>>> no_valence(
                2, std::vector<std::vector<Real>>(number_of_lm, std::vector<Real>(mesh_size, 0.0)));
            std::vector<std::vector<std::vector<Real>>> discarded_potential;
            Real core_only_energy = 0.0;
            if (harmonic_gradients != nullptr) {
                std::vector<std::vector<std::vector<Real>>> discarded_radial_weight;
                std::vector<std::vector<std::vector<Real>>> discarded_angular_weight;
                compute_one_center_xc_gradient_corrected_spin(
                    no_valence,
                    core_density,
                    pseudopotential.radial_grid,
                    pseudopotential.radial_grid_derivative,
                    mesh_size,
                    angular_grid,
                    harmonics,
                    *harmonic_gradients,
                    functional,
                    core_only_energy,
                    discarded_potential,
                    discarded_radial_weight,
                    discarded_angular_weight);
            } else {
                compute_one_center_xc_spin(no_valence,
                                           core_density,
                                           pseudopotential.radial_grid,
                                           pseudopotential.radial_grid_derivative,
                                           mesh_size,
                                           angular_grid,
                                           harmonics,
                                           functional,
                                           core_only_energy,
                                           discarded_potential);
            }
            branch_up.xc_energy -= core_only_energy;
        }

        for (int spin = 0; spin < 2; spin++) {
            OneCenterBranch& branch = spin == 0 ? branch_up : branch_down;
            branch.potential_lm.assign(number_of_lm, std::vector<Real>(mesh_size, 0.0));
            for (int lm = 0; lm < number_of_lm; lm++)
                for (int point = 0; point < mesh_size; point++)
                    branch.potential_lm[lm][point] =
                        hartree_potential[lm][point] + xc_potential_by_spin[spin][lm][point];
        }
    };

    build_branch(densities_by_spin[0].all_electron,
                 densities_by_spin[1].all_electron,
                 pseudopotential.charge_core_all_electron.radial_values,
                 options.subtract_core_exchange_correlation,
                 result[0].all_electron,
                 result[1].all_electron);
    build_branch(densities_by_spin[0].pseudo,
                 densities_by_spin[1].pseudo,
                 pseudopotential.charge_core.radial_values,
                 false,
                 result[0].pseudo,
                 result[1].pseudo);
    return result;
}

/**
 * One-center energy for an unpolarized occupancy, with either functional mode.
 *
 * The atomic reference state and every non-magnetic SCF step are unpolarized,
 * but the functional handle they are evaluated with belongs to the calculation,
 * which may be spin-polarized. Driving a polarized LibXC handle through the
 * unpolarized routines does not fail — it misreads the interleaved density
 * array and returns a plausible wrong number. For nickel that put the atomic
 * reference at −24.12 Ry instead of −1.58 Ry, and the SCF total 113 Ry low
 * while the magnetic moment still looked right.
 *
 * So the branch is on the functional, not on the caller's density count: a
 * polarized handle gets the density split evenly and the spin routines, which
 * `OneCenterEnergySpinTest` gates as identical to the unpolarized answer.
 */
inline OneCenterEnergy compute_one_center_energy_either_mode(
    const PawPseudopotential& pseudopotential,
    const OneCenterDensities& densities,
    const std::vector<AngularPoint>& angular_grid,
    const std::vector<std::vector<Real>>& harmonics,
    const XCFunctional& functional,
    const std::vector<std::vector<std::array<Real, 3>>>* harmonic_gradients = nullptr,
    const OneCenterOptions& options = {}) {
    if (functional.number_of_spin_components() < 2) {
        return compute_one_center_energy(pseudopotential,
                                         densities,
                                         angular_grid,
                                         harmonics,
                                         functional,
                                         harmonic_gradients,
                                         options);
    }

    OneCenterDensities half = densities;
    for (auto* branch : {&half.all_electron, &half.pseudo})
        for (auto& channel : *branch)
            for (auto& value : channel)
                value *= 0.5;

    auto by_spin = compute_one_center_energy_spin(pseudopotential,
                                                  {half, half},
                                                  angular_grid,
                                                  harmonics,
                                                  functional,
                                                  harmonic_gradients,
                                                  options);
    /**
     * Channel 0 already carries the whole energy — Hartree comes from the total
     * density and channel 1's energies are left at zero — and both channels'
     * potentials are equal here, so channel 0 is the complete answer.
     */
    return by_spin[0];
}

} // namespace paw_one_center
} // namespace q3
