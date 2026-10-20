#pragma once

/**
 * NLCC core charge density tabulation free functions.
 *
 * Pure functions that compute the non-linear core correction (NLCC)
 * charge density in G-space via spherical Bessel transform, and evaluate
 * the core charge density at a single q value via direct radial integration.
 */

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/math/differentiation/gradient.hpp"
#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/bessel.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/math/transform/spherical_bessel.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/reciprocal_projector_resampling.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

/**
 * Compute core charge density at a single q value via direct radial integration.
 *
 * Returns 4π ∫ ρ_core(r) * r² * j₀(qr) dr.
 */
inline Real calculate_core_charge_density_at_q(const Pseudopotential& pseudopotential,
                                               int mesh_size,
                                               Real q_value) {
    /**
     * A dataset that supplies ñ_c reciprocally means that table for the
     * plane-wave grid, not a radial pseudized core density it may also carry.
     * The two are different functions carrying different amounts of charge, so
     * this is a branch on what the dataset holds.
     */
    const auto* paw = dynamic_cast<const PawPseudopotential*>(&pseudopotential);
    if (paw != nullptr && paw->has_reciprocal_core_charge()) {
        std::vector<Real> single_q{q_value};
        std::vector<Real> tabulated;
        resample_uniform_reciprocal_table(
            paw->charge_core_reciprocal, paw->reciprocal_core_grid_spacing, 0, single_q, tabulated);
        return tabulated[0];
    }

    std::vector<Real> integrand(mesh_size);
    for (int radial_index = 0; radial_index < mesh_size; radial_index++) {
        Real bessel_value =
            spherical_bessel(0, q_value * pseudopotential.radial_grid[radial_index]);
        integrand[radial_index] = pseudopotential.charge_core.radial_values[radial_index] *
                                  pseudopotential.radial_grid[radial_index] *
                                  pseudopotential.radial_grid[radial_index] * bessel_value;
    }
    Real integrated_sum =
        simpson_rule(integrand, pseudopotential.radial_grid_derivative, mesh_size);
    return constants::four_pi * integrated_sum;
}

/**
 * Compute the derivative of the core charge density on the radial grid.
 *
 * Uses the 3-point asymmetric formula from gradient_nonuniform to compute
 * d(ρ_core)/dr on the non-uniform radial grid. This is needed for
 * GGA exchange-correlation with nonlinear core correction.
 */
inline void calculate_core_charge_density_derivative(const Pseudopotential& pseudopotential,
                                                     int mesh_size,
                                                     std::vector<Real>& core_charge_derivative) {
    core_charge_derivative.resize(mesh_size, 0.0);
    gradient_nonuniform(pseudopotential.charge_core.radial_values.data(),
                        core_charge_derivative.data(),
                        pseudopotential.radial_grid.data(),
                        mesh_size);
}

/// Result of core charge density tabulation.
struct CoreChargeTabulationResult {
    bool has_nonlinear_core_correction = false;
    std::vector<std::vector<Real>> interpolation_table;  ///< [species_type][q_index]
    std::vector<Complex> core_charge_density_reciprocal; ///< ρ_core(G) with structure factors
};

/**
 * Tabulate core charge density in G-space for all species.
 *
 * Checks whether any species has NLCC; if so, builds the q-grid interpolation
 * table via spherical Bessel transform, then computes ρ_core(G) with structure factors.
 */
inline CoreChargeTabulationResult
tabulate_core_charge_density_in_g_space(const Crystal& crystal_structure,
                                        const PlanewaveBasis& planewave_basis,
                                        const PseudopotentialSet& pseudopotential_set,
                                        const std::vector<int>& mesh_sizes,
                                        const std::vector<Real>& g_vector_magnitudes,
                                        Real cell_volume,
                                        Real q_spacing,
                                        int number_of_q_points) {

    int number_of_species_types = static_cast<int>(pseudopotential_set.size());
    int number_of_g_vectors = static_cast<int>(g_vector_magnitudes.size());

    CoreChargeTabulationResult result;

    // Check if any species has NLCC
    for (int type_index = 0; type_index < number_of_species_types; type_index++) {
        if (pseudopotential_set[type_index].has_nonlinear_core_correction) {
            result.has_nonlinear_core_correction = true;
            break;
        }
    }

    if (!result.has_nonlinear_core_correction)
        return result;

    // Build q-grid
    std::vector<Real> q_point_values(number_of_q_points);
    for (int q_index = 0; q_index < number_of_q_points; q_index++) {
        q_point_values[q_index] = q_index * q_spacing;
    }

    // Build interpolation table for each species
    result.interpolation_table.resize(number_of_species_types);
    for (int element_type = 0; element_type < number_of_species_types; element_type++) {
        const auto& pseudopotential = pseudopotential_set[element_type];
        if (!pseudopotential.has_nonlinear_core_correction) {
            result.interpolation_table[element_type].assign(number_of_q_points, 0.0);
            continue;
        }
        int radial_mesh_size = mesh_sizes[element_type];
        std::vector<Real> radial_function(pseudopotential.mesh_size, 0.0);
        for (int radial_index = 0; radial_index < pseudopotential.mesh_size; radial_index++) {
            radial_function[radial_index] =
                pseudopotential.charge_core.radial_values[radial_index] *
                pseudopotential.radial_grid[radial_index];
        }
        transform_spherical_bessel(pseudopotential.radial_grid,
                                   pseudopotential.radial_grid_derivative,
                                   radial_function,
                                   0,
                                   radial_mesh_size,
                                   q_point_values,
                                   result.interpolation_table[element_type]);
    }

    // Compute core charge density in reciprocal space with structure factors
    result.core_charge_density_reciprocal.assign(number_of_g_vectors, Complex(0.0, 0.0));
    for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
        for (int element_type = 0; element_type < number_of_species_types; element_type++) {
            if (!pseudopotential_set[element_type].has_nonlinear_core_correction)
                continue;
            Real q_value = g_vector_magnitudes[g_vector_index];
            Real core_charge_density_value = calculate_core_charge_density_at_q(
                pseudopotential_set[element_type], mesh_sizes[element_type], q_value);

            Complex structure_factor(0.0, 0.0);
            for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms();
                 atom_index++) {
                if (crystal_structure.species_type_index(atom_index) == element_type) {
                    Vector3D atomic_position =
                        crystal_structure.convert_from_crystal_to_cartesian(atom_index);
                    const auto& g_vector = planewave_basis.g_vectors()[g_vector_index];
                    Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                    structure_factor += compute_phase_factor(phase);
                }
            }
            result.core_charge_density_reciprocal[g_vector_index] +=
                core_charge_density_value * structure_factor / cell_volume;
        }
    }

    return result;
}

} // namespace q3
