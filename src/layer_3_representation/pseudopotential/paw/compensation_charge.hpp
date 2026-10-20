#pragma once

/**
 * Compensation-charge synthesis for datasets that do not ship one.
 *
 * A UPF PAW dataset tabulates Q_ij^L(r) directly (`PP_QIJL`). Some formats do
 * not: they state only the augmentation *charges* q_ij and expect the code to
 * rebuild the radial functions itself. Without them such a dataset's pseudo
 * density carries no compensation at all, its multipole moments do not match the
 * all-electron ones, and the two-grid split of a PAW calculation is invalid.
 *
 * The shape is the Kresse–Joubert construction (PRB 59, 1758 (1999)): a
 * combination of spherical Bessel functions whose arguments are chosen so
 * each vanishes at the augmentation radius,
 * fixed by two conditions —
 *
 *   1. the L-th moment is exactly the value the partial waves require, so the
 *      compensation is correct by construction rather than by convergence;
 *   2. the radial derivative vanishes at the augmentation radius, so the
 *      function joins zero smoothly instead of with a kink that would need an
 *      unreasonable plane-wave cutoff to represent.
 *
 * Adopting the published shape is not cosmetic. A converged total energy is
 * independent of the shape — any function with the right moment will do — but
 * the *rate* of convergence in the augmentation cutoff is not, so building the
 * shape Kresse & Joubert published removes a finite-augmentation-cutoff
 * discrepancy from parity comparisons with codes that use it, rather than
 * leaving it to be explained later; the parity runs confirm the agreement.
 *
 * Everything here is r-weighted, matching the partial waves and the rest of
 * the one-center module: Q_ij^L(r) as stored includes the r² of the volume
 * element, so `∫ Q_ij^L(r) r^L dr` is the moment directly.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/spherical/bessel.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"

namespace q3 {
namespace paw_compensation {

/**
 * The first `count` positive zeros of j_L, by scan-and-bisect.
 *
 * A closed form exists only for L = 0 (multiples of π); for higher L the
 * zeros are found numerically, which is cheap because only two are ever
 * needed per channel.
 */
inline std::vector<Real> spherical_bessel_zeros(int degree, int count) {
    std::vector<Real> zeros;
    const Real scan_step = 0.05;
    Real previous_argument = 1e-6;
    Real previous_value = spherical_bessel(degree, previous_argument);

    for (Real argument = previous_argument + scan_step;
         zeros.size() < static_cast<size_t>(count) && argument < 500.0;
         argument += scan_step) {
        Real value = spherical_bessel(degree, argument);
        if (previous_value == 0.0 || (previous_value < 0.0) != (value < 0.0)) {
            Real low = previous_argument;
            Real high = argument;
            for (int iteration = 0; iteration < 200; iteration++) {
                Real middle = 0.5 * (low + high);
                Real middle_value = spherical_bessel(degree, middle);
                if ((spherical_bessel(degree, low) < 0.0) != (middle_value < 0.0))
                    high = middle;
                else
                    low = middle;
            }
            zeros.push_back(0.5 * (low + high));
        }
        previous_argument = argument;
        previous_value = value;
    }
    return zeros;
}

/// d/dx j_L(x), from the standard recurrence.
inline Real spherical_bessel_derivative(int degree, Real argument) {
    if (std::abs(argument) < 1e-12)
        return degree == 1 ? (1.0 / 3.0) : 0.0;
    if (degree == 0)
        return -spherical_bessel(1, argument);
    return spherical_bessel(degree - 1, argument) -
           (static_cast<Real>(degree + 1) / argument) * spherical_bessel(degree, argument);
}

/**
 * A unit-moment compensation shape for one angular momentum.
 *
 * Returns G_L(r), r-weighted and confined to r ≤ r_aug, satisfying
 * ∫ G_L(r) r^L dr = 1. Multiply by a channel's moment to obtain its
 * Q_ij^L(r).
 */
inline std::vector<Real> build_compensation_shape(int angular_momentum,
                                                  Real augmentation_radius,
                                                  const std::vector<Real>& radial_grid,
                                                  int mesh_size) {
    std::vector<Real> shape(mesh_size, 0.0);
    auto zeros = spherical_bessel_zeros(angular_momentum, 2);
    if (zeros.size() < 2 || augmentation_radius <= 0.0)
        return shape;

    const Real first_argument = zeros[0] / augmentation_radius;
    const Real second_argument = zeros[1] / augmentation_radius;

    /**
     * Moment of each Bessel component, and its slope at the boundary.
     *
     * The integral is normalized in the continuum, the modern default; a
     * legacy compatibility mode some codes keep normalizes on the discrete
     * mesh instead. Use the exact identity
     *
     *   integral_0^R j_L(q r) r^(L+2) dr
     *       = R^(L+2) j_(L+1)(q R) / q
     *
     * so the construction follows the continuous convention without adding
     * another quadrature error. The old mesh-normalized form matched the
     * legacy mode instead and was most visible for Cu d channels in the
     * absolute parity gate.
     */
    auto moment_of = [&](Real wavenumber) {
        return std::pow(augmentation_radius, angular_momentum + 2) *
               spherical_bessel(angular_momentum + 1, wavenumber * augmentation_radius) /
               wavenumber;
    };

    Real first_moment = moment_of(first_argument);
    Real second_moment = moment_of(second_argument);
    Real first_slope = first_argument * spherical_bessel_derivative(
                                            angular_momentum, first_argument * augmentation_radius);
    Real second_slope =
        second_argument *
        spherical_bessel_derivative(angular_momentum, second_argument * augmentation_radius);

    /**
     * Solve
     *     a₁·m₁ + a₂·m₂ = 1        (unit moment)
     *     a₁·s₁ + a₂·s₂ = 0        (no kink at the boundary)
     */
    Real determinant = first_moment * second_slope - second_moment * first_slope;
    if (std::abs(determinant) < 1e-30)
        return shape;
    Real first_coefficient = second_slope / determinant;
    Real second_coefficient = -first_slope / determinant;

    for (int point = 0; point < mesh_size; point++) {
        Real radius = radial_grid[point];
        if (radius > augmentation_radius)
            break;
        Real value =
            first_coefficient * spherical_bessel(angular_momentum, first_argument * radius) +
            second_coefficient * spherical_bessel(angular_momentum, second_argument * radius);
        shape[point] = value * radius * radius;
    }
    return shape;
}

/**
 * Multipole moment ∫(φ_i φ_j − φ̃_i φ̃_j) r^L dr for one radial channel pair.
 *
 * The quantity the compensation charge must reproduce. Identical to the check
 * in `paw/validation.hpp`, promoted here because it is now production input
 * rather than only a dataset sanity test.
 */
inline Real partial_wave_moment(const PawPseudopotential& pseudopotential,
                                int first_channel,
                                int second_channel,
                                int angular_momentum) {
    int mesh_size = pseudopotential.mesh_size;
    std::vector<Real> integrand(mesh_size);
    const auto& all_electron_first =
        pseudopotential.partial_waves_all_electron[first_channel].radial_values;
    const auto& all_electron_second =
        pseudopotential.partial_waves_all_electron[second_channel].radial_values;
    const auto& pseudo_first = pseudopotential.partial_waves_pseudo[first_channel].radial_values;
    const auto& pseudo_second = pseudopotential.partial_waves_pseudo[second_channel].radial_values;

    for (int point = 0; point < mesh_size; point++) {
        Real difference = all_electron_first[point] * all_electron_second[point] -
                          pseudo_first[point] * pseudo_second[point];
        integrand[point] =
            difference * std::pow(pseudopotential.radial_grid[point], angular_momentum);
    }
    return simpson_rule(integrand, pseudopotential.radial_grid_derivative, mesh_size);
}

/**
 * Fill `augmentation_function_by_angular_momentum` for a dataset that does not
 * ship it.
 *
 * A no-op when the dataset already has them: a UPF PAW file's own Q_ij^L are
 * the generator's, and replacing them with a synthesized shape would discard
 * information the file actually contains.
 */
inline void synthesize_augmentation_functions(PawPseudopotential& pseudopotential) {
    int number_of_channels = pseudopotential.number_of_projectors;
    if (number_of_channels == 0)
        return;

    bool already_present = false;
    for (const auto& pair : pseudopotential.augmentation_function_by_angular_momentum) {
        for (const auto& channel : pair) {
            if (!channel.empty())
                already_present = true;
        }
    }
    if (already_present)
        return;

    int maximum_angular_momentum = 0;
    for (const auto& wave : pseudopotential.partial_waves_pseudo)
        maximum_angular_momentum = std::max(maximum_angular_momentum, wave.angular_momentum);
    int number_of_angular_momenta = 2 * maximum_angular_momentum + 1;

    Real augmentation_radius = pseudopotential.augmentation_cutoff_radius;
    if (augmentation_radius <= 0.0)
        return;

    // One shape per L, reused across every channel pair that needs it.
    std::vector<std::vector<Real>> shapes(number_of_angular_momenta);
    for (int angular_momentum = 0; angular_momentum < number_of_angular_momenta;
         angular_momentum++) {
        shapes[angular_momentum] = build_compensation_shape(angular_momentum,
                                                            augmentation_radius,
                                                            pseudopotential.radial_grid,
                                                            pseudopotential.mesh_size);
    }

    pseudopotential.augmentation_function_by_angular_momentum.assign(
        static_cast<size_t>(number_of_channels) * number_of_channels,
        std::vector<std::vector<Real>>(number_of_angular_momenta));
    pseudopotential.number_of_q_angular_momenta = number_of_angular_momenta;

    for (int first = 0; first < number_of_channels; first++) {
        for (int second = 0; second < number_of_channels; second++) {
            int first_l = pseudopotential.partial_waves_pseudo[first].angular_momentum;
            int second_l = pseudopotential.partial_waves_pseudo[second].angular_momentum;
            size_t pair_index = static_cast<size_t>(first) * number_of_channels + second;

            for (int angular_momentum = 0; angular_momentum < number_of_angular_momenta;
                 angular_momentum++) {
                /**
                 * Channels the triangle condition forbids carry no
                 * compensation charge — the same selection rule the UPF
                 * datasets encode as exact zeros.
                 */
                bool allowed = angular_momentum >= std::abs(first_l - second_l) &&
                               angular_momentum <= first_l + second_l &&
                               (first_l + second_l + angular_momentum) % 2 == 0;
                if (!allowed)
                    continue;

                Real moment = partial_wave_moment(pseudopotential, first, second, angular_momentum);
                auto& destination =
                    pseudopotential
                        .augmentation_function_by_angular_momentum[pair_index][angular_momentum];
                destination.assign(pseudopotential.mesh_size, 0.0);
                for (int point = 0; point < pseudopotential.mesh_size; point++)
                    destination[point] = moment * shapes[angular_momentum][point];
            }
        }
    }
}

} // namespace paw_compensation
} // namespace q3
