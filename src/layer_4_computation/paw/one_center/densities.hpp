#pragma once

/**
 * One-center densities from on-site occupancies.
 *
 * Given ρ_ij — the cross-band occupancies the ultrasoft machinery already
 * accumulates as its projector overlap coefficients — this builds the two
 * lm-decomposed radial densities a PAW energy needs:
 *
 *     n¹  = Σ_ij ρ_ij φ_i φ_j                    (all-electron)
 *     ñ¹  = Σ_ij ρ_ij [ φ̃_i φ̃_j + Q_ij^L(r) ]   (pseudo, compensated)
 *
 * The compensation charge is part of the pseudo density rather than a separate
 * term, because that is what it is defined to be: n̂ exists so that ñ¹ + n̂
 * carries the same multipole moments as n¹ (Blöchl, PRB 50, 17953 (1994)), which
 * puts it inside the pseudo density and nowhere else. That placement is what
 * makes the electrostatics cancel between the two grids, and it is exactly
 * what the gate on this file checks.
 *
 * In equation numbers: n¹ and ñ¹ are Eqs. (16)–(17) of P. E. Blöchl, Phys.
 * Rev. B 50, 17953 (1994), and Eqs. (5)–(7) of G. Kresse & D. Joubert, Phys.
 * Rev. B 59, 1758 (1999), with ρ_ij their Eq. (7); the compensated ñ¹ above
 * is their ñ¹ + n̂. The moment condition is Kresse & Joubert's Eq. (24), and
 * the compensation charge their Eq. (27).
 * Blöchl's own moments, his Eq. (24), also include the nuclear charge; the
 * Kresse–Joubert form used here leaves it out and treats the ion-ion term by
 * an Ewald sum instead.
 *
 * Everything is r-weighted: partial waves are stored as r·φ, so a product of
 * two already carries the r² of the volume element, and the result is the
 * `u_L(r) = r² ρ_L(r)` form the radial Poisson solve expects. No factor of r²
 * is inserted or removed anywhere along the chain.
 *
 * Pure over dataset + occupancies: no basis, crystal, or iterator types
 * appear, which is what lets the gate below run without an SCF.
 */

#include <cmath>
#include <memory>
#include <vector>

#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/spherical/coefficients.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"

namespace q3 {
namespace paw_one_center {

/**
 * The dataset's projector channels expanded over magnetic quantum numbers.
 *
 * A dataset states `number_of_projectors` radial channels, each carrying
 * 2l+1 magnetic components; occupancies are indexed over that expanded set.
 * The expansion follows the same convention the plane-wave projector mapping
 * uses, so an occupancy matrix built by one is readable by the other.
 */
struct ChannelExpansion {
    std::vector<int> radial_channel;   ///< index into partial_waves_*
    std::vector<int> angular_momentum; ///< l of that channel
    std::vector<int> lm_index;         ///< flat (l,m) index

    [[nodiscard]] int size() const {
        return static_cast<int>(radial_channel.size());
    }
};

inline ChannelExpansion expand_channels(const PawPseudopotential& pseudopotential) {
    ChannelExpansion expansion;
    for (int channel = 0; channel < pseudopotential.number_of_projectors; channel++) {
        int angular_momentum = pseudopotential.partial_waves_pseudo[channel].angular_momentum;
        for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
            expansion.radial_channel.push_back(channel);
            expansion.angular_momentum.push_back(angular_momentum);
            expansion.lm_index.push_back(
                harmonic_combined_degree_order_index(angular_momentum, magnetic_projection));
        }
    }
    return expansion;
}

/**
 * On-site occupancies ρ_ij for one atom.
 *
 * Carries a spin index from the outset even though the first validated path is
 * unpolarized: adding it later would touch every signature the one-center
 * module exposes.
 */
struct OnSiteOccupancies {
    int number_of_spin_components{1};
    int number_of_channels{}; ///< expanded channel count (nh)
    std::vector<Real> values; ///< [spin][i * nh + j], symmetric in (i, j)

    [[nodiscard]] Real at(int spin, int first, int second) const {
        return values[static_cast<size_t>(spin) * number_of_channels * number_of_channels +
                      static_cast<size_t>(first) * number_of_channels + second];
    }
    Real& at(int spin, int first, int second) {
        return values[static_cast<size_t>(spin) * number_of_channels * number_of_channels +
                      static_cast<size_t>(first) * number_of_channels + second];
    }

    static OnSiteOccupancies zeros(int number_of_channels, int number_of_spin_components = 1) {
        OnSiteOccupancies occupancies;
        occupancies.number_of_spin_components = number_of_spin_components;
        occupancies.number_of_channels = number_of_channels;
        occupancies.values.assign(static_cast<size_t>(number_of_spin_components) *
                                      number_of_channels * number_of_channels,
                                  0.0);
        return occupancies;
    }
};

/// lm-decomposed one-center densities, r-weighted, for one spin channel.
struct OneCenterDensities {
    std::vector<std::vector<Real>> all_electron; ///< [lm][radial point]
    std::vector<std::vector<Real>> pseudo;       ///< [lm][radial point], compensated
    int number_of_lm{};
    int mesh_size{};
};

/// Number of lm channels a density needs: products of two l_max harmonics.
inline int density_lm_count(const PawPseudopotential& pseudopotential) {
    int maximum = 0;
    for (const auto& wave : pseudopotential.partial_waves_pseudo)
        maximum = std::max(maximum, wave.angular_momentum);
    int product_degree = 2 * maximum;
    return (product_degree + 1) * (product_degree + 1);
}

/**
 * Build n¹ and ñ¹ (+ compensation) for one spin channel.
 *
 * @param gaunt  coefficients[lm_product][lm_i][lm_j] = ∫ Y Y Y dΩ
 */
inline OneCenterDensities build_one_center_densities(const PawPseudopotential& pseudopotential,
                                                     const ChannelExpansion& expansion,
                                                     const OnSiteOccupancies& occupancies,
                                                     const GauntCoefficientTable& gaunt,
                                                     int spin) {
    OneCenterDensities densities;
    densities.mesh_size = pseudopotential.mesh_size;
    densities.number_of_lm = density_lm_count(pseudopotential);
    densities.all_electron.assign(densities.number_of_lm,
                                  std::vector<Real>(densities.mesh_size, 0.0));
    densities.pseudo.assign(densities.number_of_lm, std::vector<Real>(densities.mesh_size, 0.0));

    int number_of_channels = expansion.size();
    int number_of_radial = pseudopotential.number_of_projectors;

    for (int first = 0; first < number_of_channels; first++) {
        for (int second = 0; second < number_of_channels; second++) {
            Real occupancy = occupancies.at(spin, first, second);
            if (occupancy == 0.0)
                continue;

            int radial_first = expansion.radial_channel[first];
            int radial_second = expansion.radial_channel[second];
            int lm_first = expansion.lm_index[first];
            int lm_second = expansion.lm_index[second];

            const auto& all_electron_first =
                pseudopotential.partial_waves_all_electron[radial_first].radial_values;
            const auto& all_electron_second =
                pseudopotential.partial_waves_all_electron[radial_second].radial_values;
            const auto& pseudo_first =
                pseudopotential.partial_waves_pseudo[radial_first].radial_values;
            const auto& pseudo_second =
                pseudopotential.partial_waves_pseudo[radial_second].radial_values;

            int pair_index = radial_first * number_of_radial + radial_second;

            for (int lm = 0; lm < densities.number_of_lm; lm++) {
                Real angular = gaunt.coefficients[lm][lm_first][lm_second];
                if (angular == 0.0)
                    continue;
                Real weight = occupancy * angular;

                // L of this lm channel selects the augmentation function.
                int angular_momentum = 0;
                while ((angular_momentum + 1) * (angular_momentum + 1) <= lm)
                    angular_momentum++;

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

                for (int point = 0; point < densities.mesh_size; point++) {
                    densities.all_electron[lm][point] +=
                        weight * all_electron_first[point] * all_electron_second[point];
                    Real pseudo_value = pseudo_first[point] * pseudo_second[point];
                    if (augmentation != nullptr)
                        pseudo_value += (*augmentation)[point];
                    densities.pseudo[lm][point] += weight * pseudo_value;
                }
            }
        }
    }
    return densities;
}

/**
 * Multipole moment ∫ u_LM(r) r^L dr of one density channel.
 *
 * The quantity the compensation charge exists to make equal between the
 * all-electron and pseudo densities — outside the augmentation sphere the two
 * then produce identical electrostatics, which is what lets a PAW calculation
 * be split across two grids at all.
 */
inline Real density_channel_moment(const std::vector<Real>& density_channel,
                                   int angular_momentum,
                                   const std::vector<Real>& radial_grid,
                                   const std::vector<Real>& radial_grid_derivative,
                                   int mesh_size) {
    std::vector<Real> integrand(mesh_size);
    for (int point = 0; point < mesh_size; point++) {
        integrand[point] = density_channel[point] * std::pow(radial_grid[point], angular_momentum);
    }
    return simpson_rule(integrand, radial_grid_derivative, mesh_size);
}

/// The L of a flat lm index.
inline int angular_momentum_of_lm(int lm_index) {
    int angular_momentum = 0;
    while ((angular_momentum + 1) * (angular_momentum + 1) <= lm_index)
        angular_momentum++;
    return angular_momentum;
}

} // namespace paw_one_center
} // namespace q3
