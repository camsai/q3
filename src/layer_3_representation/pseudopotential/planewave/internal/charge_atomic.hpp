#pragma once

/**
 * Atomic charge density tabulation free function.
 *
 * Pure function that computes atomic charge density in G-space via
 * radial Bessel transform on a uniform q-grid, then Lagrange interpolation
 * to the actual G-vector magnitudes.
 */

#include <cmath>
#include <stdexcept>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/reciprocal_projector_resampling.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

/**
 * Tabulate atomic charge density in G-space for all species.
 *
 * Uses Simpson's rule on a uniform q-grid, then 4-point Lagrange interpolation
 * to evaluate at each G-vector magnitude.
 *
 * @return atomic_charge_density_g[species_type][g_index]
 */
inline std::vector<std::vector<Real>>
tabulate_atomic_charge_density_in_g_space(const PseudopotentialSet& pseudopotential_set,
                                          const std::vector<int>& mesh_sizes,
                                          const std::vector<Real>& g_vector_magnitudes,
                                          Real q_spacing,
                                          int number_of_q_points) {

    int number_of_species_types = static_cast<int>(pseudopotential_set.size());
    int number_of_g_vectors = static_cast<int>(g_vector_magnitudes.size());

    // Build q-grid for tabulation
    std::vector<Real> q_point_values(number_of_q_points);
    for (int q_index = 0; q_index < number_of_q_points; q_index++) {
        q_point_values[q_index] = q_index * q_spacing;
    }

    // Tabulate on q-grid via Simpson's rule
    std::vector<std::vector<Real>> atomic_charge_density_table(number_of_species_types);
    for (int type_index = 0; type_index < number_of_species_types; type_index++) {
        /**
         * Some datasets state this quantity in reciprocal space only, so they
         * arrive with nothing radial to transform, and their shipped table is
         * resampled onto Q3's q-grid instead. Same situation as the
         * projectors, same resolution: the conventions already agree (both
         * give the valence charge at q = 0), so this is a change of grid and
         * nothing else.
         *
         * Reconstructing ρ_at from the partial waves is not an alternative: a
         * dataset's reference occupancies need not be the neutral atom's.
         */
        const auto* paw = dynamic_cast<const PawPseudopotential*>(&pseudopotential_set[type_index]);
        if (paw != nullptr && paw->has_reciprocal_atomic_charge()) {
            resample_uniform_reciprocal_table(paw->charge_atomic_reciprocal,
                                              paw->reciprocal_charge_grid_spacing,
                                              0,
                                              q_point_values,
                                              atomic_charge_density_table[type_index]);
            continue;
        }
        if (pseudopotential_set[type_index].charge_atomic.radial_values.empty()) {
            throw std::runtime_error(
                "Pseudopotential for " + pseudopotential_set[type_index].element +
                " carries no atomic charge density in either representation, so no starting "
                "density can be built from it.");
        }
        const auto& pseudopotential = pseudopotential_set[type_index];
        int radial_mesh_size = mesh_sizes[type_index];
        atomic_charge_density_table[type_index].resize(number_of_q_points);
        for (int q_index = 0; q_index < number_of_q_points; q_index++) {
            Real q_value = q_point_values[q_index];
            Real sum = 0.0;
            for (int radial_index = 0; radial_index < radial_mesh_size; radial_index++) {
                Real bessel_value =
                    spherical_bessel(0, q_value * pseudopotential.radial_grid[radial_index]);
                Real integrand = pseudopotential.charge_atomic.radial_values[radial_index] *
                                 bessel_value *
                                 pseudopotential.radial_grid_derivative[radial_index];
                Real weight;
                if (radial_index == 0 || radial_index == radial_mesh_size - 1)
                    weight = 1.0;
                else if (radial_index % 2 == 1)
                    weight = 4.0;
                else
                    weight = 2.0;
                sum += weight * integrand;
            }
            atomic_charge_density_table[type_index][q_index] = sum / 3.0;
        }
    }

    // Interpolate to G-vectors using 4-point Lagrange
    std::vector<std::vector<Real>> atomic_charge_density_g(number_of_species_types);
    for (int type_index = 0; type_index < number_of_species_types; type_index++) {
        atomic_charge_density_g[type_index].resize(number_of_g_vectors);
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
            Real q = g_vector_magnitudes[g_vector_index];
            if (q < math::tolerances::zero_vector_magnitude_threshold) {
                atomic_charge_density_g[type_index][g_vector_index] =
                    atomic_charge_density_table[type_index][0];
            } else {
                atomic_charge_density_g[type_index][g_vector_index] = lagrange_interpolation_4point(
                    atomic_charge_density_table[type_index], q, q_spacing, number_of_q_points);
            }
            if (std::isnan(atomic_charge_density_g[type_index][g_vector_index])) {
                log::error("Atomic charge density G-space coefficient is NaN: "
                           "species_type={}, g_index={}, q_value={}",
                           type_index,
                           g_vector_index,
                           q);
            }
        }
    }

    return atomic_charge_density_g;
}

} // namespace q3
