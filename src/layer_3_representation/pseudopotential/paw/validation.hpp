#pragma once

/**
 * PAW dataset self-consistency checks.
 *
 * A PAW dataset overdetermines itself: the multipole moments, the augmentation
 * charges and the partial waves are three views of the same one-center data, so
 * they can be checked against each other without any external reference. These
 * checks are what let a reader of an undocumented format be trusted later
 * on: a dataset that satisfies all of them was almost
 * certainly parsed correctly, whatever format it arrived in.
 *
 * Each check returns the worst absolute deviation it found, so callers can
 * report a number rather than a bare pass/fail. The tolerances used by the
 * tests were established by evaluating these identities on the reference
 * datasets rather than assumed.
 *
 * All partial waves are r-weighted (r·φ), so products already carry the r² of
 * the volume element and integrate against the radial-grid derivative directly.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"

namespace q3 {
namespace paw_validation {

/**
 * Whether an augmentation channel (l_i, l_j, L) is allowed by the
 * angular-momentum triangle condition. Forbidden channels carry no
 * compensation charge and are stored as exact zeros.
 */
inline bool is_allowed_augmentation_channel(int angular_momentum_1,
                                            int angular_momentum_2,
                                            int total_angular_momentum) {
    if (total_angular_momentum < std::abs(angular_momentum_1 - angular_momentum_2))
        return false;
    if (total_angular_momentum > angular_momentum_1 + angular_momentum_2)
        return false;
    return (angular_momentum_1 + angular_momentum_2 + total_angular_momentum) % 2 == 0;
}

/**
 * Multipole identity: for every triangle-allowed channel,
 *
 *     multipole_moments[L][i][j] = ∫ (φ_i φ_j − φ̃_i φ̃_j) r^L dr
 *
 * This is the tightest available check that the all-electron partial waves,
 * the pseudo partial waves and the stored moments describe one dataset.
 */
inline Real worst_multipole_moment_deviation(const PawPseudopotential& pseudopotential) {
    Real worst_deviation = 0.0;
    int number_of_channels = pseudopotential.number_of_projectors;
    int mesh_size = pseudopotential.mesh_size;

    for (int angular_momentum = 0;
         angular_momentum < pseudopotential.number_of_augmentation_angular_momenta;
         angular_momentum++) {
        for (int channel_1 = 0; channel_1 < number_of_channels; channel_1++) {
            for (int channel_2 = 0; channel_2 < number_of_channels; channel_2++) {
                const auto& all_electron_1 = pseudopotential.partial_waves_all_electron[channel_1];
                const auto& all_electron_2 = pseudopotential.partial_waves_all_electron[channel_2];
                if (!is_allowed_augmentation_channel(all_electron_1.angular_momentum,
                                                     all_electron_2.angular_momentum,
                                                     angular_momentum))
                    continue;

                const auto& pseudo_1 = pseudopotential.partial_waves_pseudo[channel_1];
                const auto& pseudo_2 = pseudopotential.partial_waves_pseudo[channel_2];

                std::vector<Real> integrand(mesh_size);
                for (int radial_index = 0; radial_index < mesh_size; radial_index++) {
                    Real difference =
                        all_electron_1.radial_values[radial_index] *
                            all_electron_2.radial_values[radial_index] -
                        pseudo_1.radial_values[radial_index] * pseudo_2.radial_values[radial_index];
                    integrand[radial_index] =
                        difference *
                        std::pow(pseudopotential.radial_grid[radial_index], angular_momentum);
                }

                Real computed =
                    simpson_rule(integrand, pseudopotential.radial_grid_derivative, mesh_size);
                Real stored =
                    pseudopotential.multipole_moments[angular_momentum][channel_1][channel_2];
                worst_deviation = std::max(worst_deviation, std::abs(computed - stored));
            }
        }
    }
    return worst_deviation;
}

/**
 * The L = 0 multipole block is the augmentation charge matrix: the monopole of
 * the compensation charge is exactly what the ultrasoft apparatus calls q_ij.
 * Datasets store both, and they must agree.
 */
inline Real worst_augmentation_charge_deviation(const PawPseudopotential& pseudopotential) {
    Real worst_deviation = 0.0;
    int number_of_channels = pseudopotential.number_of_projectors;
    if (pseudopotential.multipole_moments.empty())
        return worst_deviation;

    for (int channel_1 = 0; channel_1 < number_of_channels; channel_1++) {
        for (int channel_2 = 0; channel_2 < number_of_channels; channel_2++) {
            Real monopole = pseudopotential.multipole_moments[0][channel_1][channel_2];
            Real augmentation_charge =
                pseudopotential.augmentation_overlap_integrals[channel_1][channel_2];
            worst_deviation = std::max(worst_deviation, std::abs(monopole - augmentation_charge));
        }
    }
    return worst_deviation;
}

/**
 * Projector duality: ⟨p̃_i|φ̃_j⟩ = δ_ij within the same angular-momentum
 * channel. This is the defining property of the PAW projectors — if it fails,
 * the projectors and the pseudo partial waves came from different datasets or
 * were paired up in the wrong order.
 *
 * Channels with different l are orthogonal by symmetry and carry no
 * information about the radial tabulation, so they are skipped.
 */
inline Real worst_projector_duality_deviation(const PawPseudopotential& pseudopotential) {
    Real worst_deviation = 0.0;
    int number_of_channels = pseudopotential.number_of_projectors;
    int mesh_size = pseudopotential.mesh_size;

    for (int projector_index = 0; projector_index < number_of_channels; projector_index++) {
        for (int wave_index = 0; wave_index < number_of_channels; wave_index++) {
            const auto& projector = pseudopotential.projectors[projector_index];
            const auto& pseudo_wave = pseudopotential.partial_waves_pseudo[wave_index];
            if (projector.angular_momentum != pseudo_wave.angular_momentum)
                continue;

            std::vector<Real> integrand(mesh_size);
            for (int radial_index = 0; radial_index < mesh_size; radial_index++) {
                integrand[radial_index] =
                    projector.radial_values[radial_index] * pseudo_wave.radial_values[radial_index];
            }
            Real overlap =
                simpson_rule(integrand, pseudopotential.radial_grid_derivative, mesh_size);
            Real expected = (projector_index == wave_index) ? 1.0 : 0.0;
            worst_deviation = std::max(worst_deviation, std::abs(overlap - expected));
        }
    }
    return worst_deviation;
}

/**
 * Integrated all-electron core charge, 4π ∫ ρ_core(r) r² dr. Equals the number
 * of core electrons the dataset froze out, Z − Z_valence, which is how a caller
 * can confirm the all-electron core density was read on the right grid and with
 * the right normalization without knowing Z beforehand.
 */
inline Real integrated_all_electron_core_charge(const PawPseudopotential& pseudopotential) {
    int mesh_size = pseudopotential.mesh_size;
    const auto& core_density = pseudopotential.charge_core_all_electron.radial_values;
    if (core_density.empty())
        return 0.0;

    std::vector<Real> integrand(mesh_size);
    for (int radial_index = 0; radial_index < mesh_size; radial_index++) {
        Real radius = pseudopotential.radial_grid[radial_index];
        integrand[radial_index] = core_density[radial_index] * radius * radius;
    }
    constexpr Real four_pi = 4.0 * M_PI;
    return four_pi * simpson_rule(integrand, pseudopotential.radial_grid_derivative, mesh_size);
}

/**
 * Nuclear charge implied by the all-electron local potential, read off its
 * Coulomb tail toward the origin: V_AE(r) → −2Z/r in Rydberg units, so
 * −r·V(r)/2 → Z. Evaluated at the innermost grid point, which sits close
 * enough to the origin on a logarithmic mesh to recover Z to a fraction of a
 * percent.
 */
inline Real nuclear_charge_from_all_electron_potential(const PawPseudopotential& pseudopotential) {
    const auto& potential = pseudopotential.potential_local_all_electron.radial_values;
    if (potential.empty() || pseudopotential.radial_grid.empty())
        return 0.0;
    return -0.5 * pseudopotential.radial_grid[0] * potential[0];
}

} // namespace paw_validation
} // namespace q3
