#pragma once

/**
 * The one-center contribution to D_ij.
 *
 * D_ij is assembled from three pieces, and this file owns exactly one of them.
 * The split is the formalism's own (Blöchl, PRB 50, 17953 (1994); Kresse &
 * Joubert, PRB 59, 1758 (1999)): only the Hxc part of D¹ − D̃¹ depends on the
 * current density, so it is the only piece an SCF step recomputes, and it adds
 * to the D the ultrasoft pathway already produced rather than replacing any of
 * it:
 *
 *     D_total = D⁰ (dataset)  +  ∫ v_eff Q_ij (grid side)  +  (D¹ − D̃¹)_Hxc
 *                                                             ^^^^^^^^^^^^^^ here
 *
 * The kinetic and core-reference terms are baked into `PP_DIJ` / `DION` at
 * generation time and are **not** recomputed — doing so would double-count
 * them silently. So this is a decoration of the ultrasoft result, never a
 * replacement for it.
 *
 * The construction is the derivative written out. The one-center energy depends
 * on ρ_ij only through the two densities, so ∂E/∂ρ_ij is that pair's own density
 * — φ_i φ_j on the all-electron side, φ̃_i φ̃_j + Q_ij on the pseudo side —
 * integrated against that branch's potential, the pseudo branch entering
 * negatively. No other closed form exists; the alternative is finite
 * differences, which is what the gate uses as an independent check rather than
 * as the implementation.
 *
 * That makes D_ij the derivative of the one-center energy with respect to
 * ρ_ij by construction, which is what the finite-difference gate asserts. The
 * gate covers everything inside this module; it cannot see the grid-side term
 * or the split between grid and sphere, which the A2.0 memo owns instead.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/paw/one_center/densities.hpp"
#include "layer_4_computation/paw/one_center/energy.hpp"
#include "layer_4_computation/paw/one_center/gradient.hpp"
#include "layer_4_computation/paw/one_center/radial_poisson.hpp"

namespace q3 {
namespace paw_one_center {

/**
 * One-center D_ij correction, as a dense (nh × nh) matrix for one spin.
 *
 * @param energy  the branches produced by compute_one_center_energy, whose
 *                `potential_lm` are integrated against
 */
inline std::vector<Real> compute_one_center_d_matrix(const PawPseudopotential& pseudopotential,
                                                     const ChannelExpansion& expansion,
                                                     const GauntCoefficientTable& gaunt,
                                                     const OneCenterEnergy& energy) {
    int number_of_channels = expansion.size();
    int number_of_radial = pseudopotential.number_of_projectors;
    int mesh_size = pseudopotential.mesh_size;
    int number_of_lm = density_lm_count(pseudopotential);

    std::vector<Real> d_matrix(static_cast<size_t>(number_of_channels) * number_of_channels, 0.0);

    std::vector<Real> integrand(mesh_size);
    for (int first = 0; first < number_of_channels; first++) {
        for (int second = 0; second < number_of_channels; second++) {
            int radial_first = expansion.radial_channel[first];
            int radial_second = expansion.radial_channel[second];
            int lm_first = expansion.lm_index[first];
            int lm_second = expansion.lm_index[second];
            int pair_index = radial_first * number_of_radial + radial_second;

            const auto& all_electron_first =
                pseudopotential.partial_waves_all_electron[radial_first].radial_values;
            const auto& all_electron_second =
                pseudopotential.partial_waves_all_electron[radial_second].radial_values;
            const auto& pseudo_first =
                pseudopotential.partial_waves_pseudo[radial_first].radial_values;
            const auto& pseudo_second =
                pseudopotential.partial_waves_pseudo[radial_second].radial_values;

            Real accumulated = 0.0;

            /**
             * For a gradient-corrected functional the perturbation's own
             * channels are needed, not just their values: g_lm(r) = G·p(r)/r²
             * and its radial derivative, paired with the by-parts weights the
             * energy routine produced.
             */
            const bool gradient_corrected = energy.all_electron.has_gradient_correction();
            std::vector<Real> all_electron_channel;
            std::vector<Real> pseudo_channel;
            std::vector<Real> all_electron_channel_derivative;
            std::vector<Real> pseudo_channel_derivative;

            for (int lm = 0; lm < number_of_lm; lm++) {
                Real angular = gaunt.coefficients[lm][lm_first][lm_second];
                if (angular == 0.0)
                    continue;
                int angular_momentum = angular_momentum_of_lm(lm);

                const std::vector<Real>* augmentation = nullptr;
                if (pair_index <
                        static_cast<int>(
                            pseudopotential.augmentation_function_by_angular_momentum.size()) &&
                    angular_momentum <
                        static_cast<int>(
                            pseudopotential.augmentation_function_by_angular_momentum[pair_index]
                                .size())) {
                    const auto& candidate =
                        pseudopotential.augmentation_function_by_angular_momentum[pair_index]
                                                                                 [angular_momentum];
                    if (!candidate.empty())
                        augmentation = &candidate;
                }

                // All-electron branch, positive.
                for (int point = 0; point < mesh_size; point++) {
                    integrand[point] = angular * all_electron_first[point] *
                                       all_electron_second[point] *
                                       energy.all_electron.potential_lm[lm][point];
                }
                auto cumulative = cumulative_integral(
                    integrand, pseudopotential.radial_grid_derivative, mesh_size);
                accumulated += cumulative[mesh_size - 1];

                if (gradient_corrected) {
                    all_electron_channel.assign(mesh_size, 0.0);
                    for (int point = 0; point < mesh_size; point++) {
                        Real radius = pseudopotential.radial_grid[point];
                        all_electron_channel[point] = angular * all_electron_first[point] *
                                                      all_electron_second[point] /
                                                      (radius * radius);
                    }
                    all_electron_channel_derivative = radial_derivative(
                        all_electron_channel, pseudopotential.radial_grid_derivative, mesh_size);
                    for (int point = 0; point < mesh_size; point++) {
                        Real radius = pseudopotential.radial_grid[point];
                        integrand[point] =
                            radius * radius * all_electron_channel_derivative[point] *
                                energy.all_electron.gradient_radial_weight_lm[lm][point] +
                            all_electron_channel[point] *
                                energy.all_electron.gradient_angular_weight_lm[lm][point];
                    }
                    cumulative = cumulative_integral(
                        integrand, pseudopotential.radial_grid_derivative, mesh_size);
                    accumulated += cumulative[mesh_size - 1];
                }

                // Pseudo branch, negative, compensation charge included.
                for (int point = 0; point < mesh_size; point++) {
                    Real pseudo_value = pseudo_first[point] * pseudo_second[point];
                    if (augmentation != nullptr)
                        pseudo_value += (*augmentation)[point];
                    integrand[point] =
                        angular * pseudo_value * energy.pseudo.potential_lm[lm][point];
                }
                cumulative = cumulative_integral(
                    integrand, pseudopotential.radial_grid_derivative, mesh_size);
                accumulated -= cumulative[mesh_size - 1];

                if (gradient_corrected) {
                    pseudo_channel.assign(mesh_size, 0.0);
                    for (int point = 0; point < mesh_size; point++) {
                        Real radius = pseudopotential.radial_grid[point];
                        Real pseudo_value = pseudo_first[point] * pseudo_second[point];
                        if (augmentation != nullptr)
                            pseudo_value += (*augmentation)[point];
                        pseudo_channel[point] = angular * pseudo_value / (radius * radius);
                    }
                    pseudo_channel_derivative = radial_derivative(
                        pseudo_channel, pseudopotential.radial_grid_derivative, mesh_size);
                    for (int point = 0; point < mesh_size; point++) {
                        Real radius = pseudopotential.radial_grid[point];
                        integrand[point] = radius * radius * pseudo_channel_derivative[point] *
                                               energy.pseudo.gradient_radial_weight_lm[lm][point] +
                                           pseudo_channel[point] *
                                               energy.pseudo.gradient_angular_weight_lm[lm][point];
                    }
                    cumulative = cumulative_integral(
                        integrand, pseudopotential.radial_grid_derivative, mesh_size);
                    accumulated -= cumulative[mesh_size - 1];
                }
            }
            d_matrix[static_cast<size_t>(first) * number_of_channels + second] = accumulated;
        }
    }
    return d_matrix;
}

} // namespace paw_one_center
} // namespace q3
