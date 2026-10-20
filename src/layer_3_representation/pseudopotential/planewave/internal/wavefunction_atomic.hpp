#pragma once

/**
 * Planewave representation of atomic wavefunctions.
 *
 * Provides the Bessel-transformed atomic wavefunction χ(q) interpolation table
 * for wavefunction initialization.  This is the canonical location for the
 * radial Fourier transform of pseudopotential atomic wavefunctions onto a
 * uniform q-grid, which is then interpolated at |k+G| values.
 *
 * The tabulated quantity is the radial part of the Rayleigh expansion of a
 * plane wave in the cell-normalised basis:
 *   table[species_type][wavefunction_index][q_index]
 *       = 4π/√Ω ∫ χ(r) j_l(q·r) r dr
 * with the 4π and j_l from the expansion, the 1/√Ω from the plane-wave
 * normalisation, and the single power of r because UPF stores r·R_nl(r).
 *
 * The q axis is innermost so the 4-point interpolation stencil reads four
 * consecutive entries of one contiguous vector; species is outermost because
 * the shell count varies per species.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/spherical/bessel.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

/**
 * Tabulate atomic wavefunction radial Fourier transforms on a uniform q-grid.
 *
 * For each species type and each atomic wavefunction, computes:
 *   table[species_type][wavefunction_index][q_index]
 *       = (4π / √Ω) ∫₀^∞ χ(r) · j_l(q·r) · r · dr
 * using Simpson's rule on the radial mesh (truncated at the radial cutoff).
 *
 * @param pseudopotential_set  The pseudopotential set (one per species type).
 * @param cell_volume          Unit cell volume Ω in Bohr³.
 * @param q_spacing            Spacing between q-grid points.
 * @param number_of_q_points   Number of q-grid points.
 * @return table[species_type][wavefunction_index][q_index]
 */
inline std::vector<std::vector<std::vector<Real>>>
tabulate_atomic_wavefunctions_in_q_space(const PseudopotentialSet& pseudopotential_set,
                                         Real cell_volume,
                                         Real q_spacing,
                                         int number_of_q_points) {

    int number_of_species = static_cast<int>(pseudopotential_set.size());
    Real prefactor = constants::four_pi / std::sqrt(cell_volume);

    // Compute mesh truncation sizes for each species
    std::vector<int> mesh_truncation_sizes(number_of_species);
    for (int species_index = 0; species_index < number_of_species; species_index++) {
        const auto& pseudopotential = pseudopotential_set[species_index];
        if (math::tolerances::use_radial_integration_truncation) {
            mesh_truncation_sizes[species_index] = pseudopotential.mesh_size;
            for (int radial_index = pseudopotential.mesh_size - 1; radial_index >= 0;
                 radial_index--) {
                if (pseudopotential.radial_grid[radial_index] <
                    math::tolerances::radial_integration_truncation_radius) {
                    mesh_truncation_sizes[species_index] = radial_index + 1;
                    if (mesh_truncation_sizes[species_index] % 2 == 0)
                        mesh_truncation_sizes[species_index]++; // force odd for Simpson
                    break;
                }
            }
            if (mesh_truncation_sizes[species_index] > pseudopotential.mesh_size) {
                mesh_truncation_sizes[species_index] = pseudopotential.mesh_size;
                if (mesh_truncation_sizes[species_index] % 2 == 0) {
                    mesh_truncation_sizes[species_index]--;
                }
            }
        } else {
            mesh_truncation_sizes[species_index] = pseudopotential.mesh_size;
            if (mesh_truncation_sizes[species_index] % 2 == 0) {
                mesh_truncation_sizes[species_index]--;
            }
        }
    }

    // Build tabulation
    std::vector<std::vector<std::vector<Real>>> table(number_of_species);
    for (int type_index = 0; type_index < number_of_species; type_index++) {
        const auto& pseudopotential = pseudopotential_set[type_index];
        table[type_index].resize(pseudopotential.wavefunctions_atomic.size());

        for (int wavefunction_index = 0;
             wavefunction_index < static_cast<int>(pseudopotential.wavefunctions_atomic.size());
             wavefunction_index++) {
            table[type_index][wavefunction_index].resize(number_of_q_points, 0.0);

            if (pseudopotential.wavefunctions_atomic[wavefunction_index].occupations < 0.0)
                continue; // skip unoccupied

            int angular_momentum =
                pseudopotential.wavefunctions_atomic[wavefunction_index].angular_momentum;

            for (int q_index = 0; q_index < number_of_q_points; q_index++) {
                Real q_value = q_spacing * q_index;
                // Compute ∫ χ(r) * j_l(q*r) * r * dr using Simpson
                std::vector<Real> integrand(mesh_truncation_sizes[type_index]);
                for (int radial_index = 0; radial_index < mesh_truncation_sizes[type_index];
                     radial_index++) {
                    Real bessel_value = spherical_bessel(
                        angular_momentum, q_value * pseudopotential.radial_grid[radial_index]);
                    integrand[radial_index] =
                        pseudopotential.wavefunctions_atomic[wavefunction_index]
                            .radial_values[radial_index] *
                        bessel_value * pseudopotential.radial_grid[radial_index];
                }
                table[type_index][wavefunction_index][q_index] =
                    prefactor * simpson_rule(integrand,
                                             pseudopotential.radial_grid_derivative,
                                             mesh_truncation_sizes[type_index]);
            }
        }
    }

    return table;
}

} // namespace q3
