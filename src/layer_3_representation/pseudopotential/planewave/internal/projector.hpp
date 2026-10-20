#pragma once

/**
 * Kleinman-Bylander projector computation.
 *
 * Free functions for computing <β|k+G> projectors and the S matrix
 * (overlap) for US-PP generalized eigenvalue problems.
 */

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

/**
 * Compute <β_ib|k+G> for all atoms and projectors at a given k-point.
 * kleinman_bylander_potential[ikb][ig] = (1/√Ω) * β(|k+G|) * Y_lm(k̂+Ĝ) * S(atom) * (-i)^l
 *
 * Uses pre-tabulated β(q) from the BetaFunctionTable for O(1) lookup
 * instead of recomputing the full radial Bessel integral per G-vector.
 *
 * Optimized to precompute per-G quantities (|k+G|, Y_lm, structure factors)
 * once and reuse across all atoms and projectors.
 */
inline void compute_kleinman_bylander_projectors(const Vector3D& k_crystal,
                                                 const Crystal& crystal_structure,
                                                 const PlanewaveBasis& planewave_basis,
                                                 const PseudopotentialSet& pseudopotential_set,
                                                 const BetaFunctionTable& beta_function_table,
                                                 const std::vector<int>& g_vector_indices,
                                                 int number_of_plane_waves,
                                                 int total_number_of_projectors,
                                                 Matrix<Complex>& kleinman_bylander_potential) {

    kleinman_bylander_potential.resize(
        total_number_of_projectors, number_of_plane_waves, Complex(0.0, 0.0));

    const Matrix3x3& reciprocal_lattice_vectors = crystal_structure.reciprocal_lattice_vectors();
    Vector3D kpoint_cartesian;
    for (int dimension = 0; dimension < 3; dimension++) {
        kpoint_cartesian[dimension] = k_crystal.x * reciprocal_lattice_vectors[0][dimension] +
                                      k_crystal.y * reciprocal_lattice_vectors[1][dimension] +
                                      k_crystal.z * reciprocal_lattice_vectors[2][dimension];
    }

    Real two_pi_over_lattice_parameter = crystal_structure.two_pi_over_lattice_parameter();
    Real cell_volume = crystal_structure.cell_volume();
    Real inverse_sqrt_cell_volume = 1.0 / std::sqrt(cell_volume);

    // --- Precompute per-G quantities once ---

    // 1. k+G cartesian vectors and |k+G|² for all G-vectors
    std::vector<std::array<Real, 3>> k_plus_g_cartesian_array(number_of_plane_waves);
    std::vector<Real> k_plus_g_magnitude_squared(number_of_plane_waves);
    std::vector<Real> k_plus_g_magnitude(number_of_plane_waves);

    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        Vector3D k_plus_g =
            kpoint_cartesian + planewave_basis.g_vectors()[g_vector_indices[g_index]].g_cart;
        k_plus_g_cartesian_array[g_index] = {k_plus_g.x, k_plus_g.y, k_plus_g.z};
        Real magnitude_squared =
            k_plus_g.x * k_plus_g.x + k_plus_g.y * k_plus_g.y + k_plus_g.z * k_plus_g.z;
        k_plus_g_magnitude_squared[g_index] = magnitude_squared;
        k_plus_g_magnitude[g_index] = std::sqrt(magnitude_squared) * two_pi_over_lattice_parameter;
    }

    /**
     * Precompute structure factors per atom for all G-vectors
     *    structure_factor[atom][g] = exp(-i * 2π * (k+G)·r_atom)
     *    = exp(-i * 2π * G·r_atom) * exp(-i * 2π * k·r_atom)
     *    The k·r part is constant per atom, only G·r varies.
     */
    int number_of_atoms = crystal_structure.number_of_atoms();
    std::vector<std::vector<Complex>> structure_factors(number_of_atoms);

    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        Vector3D atomic_position = crystal_structure.convert_from_crystal_to_cartesian(atom_index);
        Real k_dot_r = kpoint_cartesian.dot(atomic_position);
        structure_factors[atom_index].resize(number_of_plane_waves);

        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            Real phase_angle = constants::two_pi *
                               (planewave_basis.g_vectors()[g_vector_indices[g_index]].g_cart.dot(
                                    atomic_position) +
                                k_dot_r);
            structure_factors[atom_index][g_index] = compute_phase_factor(phase_angle);
        }
    }

    // Determine maximum angular momentum across all species
    int maximum_angular_momentum = 0;
    for (int species_index = 0; species_index < static_cast<int>(pseudopotential_set.size());
         species_index++) {
        for (const auto& projector : pseudopotential_set[species_index].projectors) {
            maximum_angular_momentum =
                std::max(maximum_angular_momentum, projector.angular_momentum);
        }
    }
    int number_of_lm_components = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);

    // Batch-compute all Y_lm(k+G) using the optimized batch function
    std::vector<std::vector<Real>> spherical_harmonics_values;
    compute_real_spherical_harmonics_batch(number_of_lm_components,
                                           number_of_plane_waves,
                                           k_plus_g_cartesian_array,
                                           k_plus_g_magnitude_squared,
                                           spherical_harmonics_values);

    // --- Build projectors using precomputed data ---

    int combined_projector_index = 0;
    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        int species_type_index = crystal_structure.species_type_index(atom_index);
        const auto& atom_structure_factors = structure_factors[atom_index];

        for (int beta_index = 0;
             beta_index <
             static_cast<int>(pseudopotential_set[species_type_index].projectors.size());
             beta_index++) {
            int angular_momentum =
                pseudopotential_set[species_type_index].projectors[beta_index].angular_momentum;
            Complex angular_phase_factor = compute_angular_phase_factor(angular_momentum);
            Complex combined_prefactor = inverse_sqrt_cell_volume * angular_phase_factor;

            for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                int lm_index =
                    harmonic_combined_degree_order_index(angular_momentum, magnetic_projection);

                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    Real beta_value_at_q = beta_function_table.interpolate(
                        species_type_index, beta_index, k_plus_g_magnitude[g_index]);

                    kleinman_bylander_potential[combined_projector_index][g_index] =
                        beta_value_at_q * spherical_harmonics_values[g_index][lm_index] *
                        atom_structure_factors[g_index] * combined_prefactor;
                }
                combined_projector_index++;
            }
        }
    }
}
} // namespace q3
