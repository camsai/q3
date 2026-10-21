#pragma once

/**
 * Ultrasoft pseudopotential augmentation stress contribution.
 *
 * Computes the stress tensor contribution from the augmentation charges
 * that arise in ultrasoft pseudopotentials. For norm-conserving
 * pseudopotentials, this contribution is identically zero.
 *
 * Uses analytical derivatives of Q_ij(G) via the product rule applied to
 *   Q_ij(G) = sum_lm (-i)^l * c_lm * Ylm(G^) * Q_ij^l(|G|)
 * (Vanderbilt, PRB 41, 7892 (1990)), which has exactly two terms:
 *   dQ_ij/dG_alpha = sum_lm (-i)^l * c_lm * [dYlm/dG_alpha * Q_ij^l(|G|)
 *                    + Ylm * dQ_ij^l/d|G| * G_alpha/|G|]
 * where Q_ij^l(|G|) is the radial Fourier transform of Q_ij in channel l.
 *
 * Two stages, in this order because Q_ij(G) carries no atom dependence while
 * the occupancies and the structure factor carry nothing else:
 *   1. Pre-compute dQ_ij(G) for ALL G-vectors per (pair, direction), once per
 *      species rather than once per atom
 *   2. Weight by the projector occupancies and contract with
 *      V*(G)*e^{-iGτ}*G_alpha
 *
 * The expansion differentiated is the Fourier transform of Eq. (27) of
 * K. Laasonen, A. Pasquarello, R. Car, C. Lee & D. Vanderbilt, Phys. Rev. B
 * 47, 10142 (1993), with the radial transform of their Eq. (32). The strain
 * derivative is taken in the stress formalism of O. H. Nielsen &
 * R. M. Martin, Phys. Rev. B 32, 3780 (1985).
 */

#include <cmath>
#include <stdexcept>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/matrix.hpp"

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"

namespace q3 {

class UltrasoftAugmentationStressTensorCalculator {
  public:
    UltrasoftAugmentationStressTensorCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveKohnShamQuantumSystem& quantum_system,
        const std::vector<std::vector<Complex>>& potential_g_by_spin,
        const std::vector<Matrix<Real>>& coefficients_by_spin,
        StressTensorContributions& result)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , quantum_system_(quantum_system)
        , potential_g_by_spin_(potential_g_by_spin)
        , coefficients_by_spin_(coefficients_by_spin)
        , result_(result) {}

    void compute() {
        for (int index_a = 0; index_a < 3; index_a++)
            for (int index_b = 0; index_b < 3; index_b++)
                result_.augmentation[index_a][index_b] = 0.0;

        if (!quantum_system_.has_ultrasoft())
            return;
        if (potential_g_by_spin_.size() != coefficients_by_spin_.size())
            throw std::runtime_error(
                "augmentation stress: " + std::to_string(potential_g_by_spin_.size()) +
                " potential channels against " + std::to_string(coefficients_by_spin_.size()) +
                " occupancy channels; they must pair up spin by spin");

        for (size_t spin_index = 0; spin_index < potential_g_by_spin_.size(); spin_index++)
            accumulate_spin_channel(potential_g_by_spin_[spin_index],
                                    coefficients_by_spin_[spin_index]);

        /**
         * Symmetrized once over the spin-summed tensor rather than per channel.
         * Symmetrization is linear, so the two are identical; doing it here
         * keeps the accumulation loop free of it.
         */
        for (int index_a = 0; index_a < 3; index_a++) {
            for (int index_b = index_a + 1; index_b < 3; index_b++) {
                Real average_stress_component = (result_.augmentation[index_a][index_b] +
                                                 result_.augmentation[index_b][index_a]) /
                                                2.0;
                result_.augmentation[index_a][index_b] = average_stress_component;
                result_.augmentation[index_b][index_a] = average_stress_component;
            }
        }
    }

  private:
    /**
     * One spin channel's contribution, accumulated into `result_.augmentation`.
     *
     * Spin has to be resolved here for the same reason it does in the
     * augmentation *force*: contracting the spin-averaged potential against the
     * spin-summed occupancies drops the (V↑−V↓)(ρ↑−ρ↓) cross term, which is
     * zero only for a non-magnetic cell.
     */
    void accumulate_spin_channel(const std::vector<Complex>& total_potential_g,
                                 const Matrix<Real>& projector_overlap_coefficients) {
        int number_of_atoms = crystal_structure_.number_of_atoms();
        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        Real two_pi_over_lattice_parameter = crystal_structure_.two_pi_over_lattice_parameter();
        int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();

        auto& ultrasoft_system =
            static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(quantum_system_);
        const auto& augmentation_q_table =
            ultrasoft_system.ultrasoft_basis().augmentation_q_table();
        const auto& spherical_harmonics_table =
            ultrasoft_system.ultrasoft_basis().spherical_harmonics_table();
        const auto& gaunt_table = ultrasoft_system.ultrasoft_basis().gaunt_table();
        int number_of_lm_components = spherical_harmonics_table.number_of_lm_components();
        const auto& projector_mapping_table =
            quantum_system_.pseudopotential_basis().projector_mapping_table();

        // Pre-compute G-vector data
        std::vector<std::array<Real, 3>> g_cartesian_vectors(number_of_g_vectors);
        std::vector<Real> g_squared_values(number_of_g_vectors);
        const auto& g_vector_magnitudes = planewave_basis_.g_vector_magnitudes();

        for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
            const auto& g_vector = planewave_basis_.g_vectors()[g_index];
            g_cartesian_vectors[g_index] = {g_vector.g_cart.x * two_pi_over_lattice_parameter,
                                            g_vector.g_cart.y * two_pi_over_lattice_parameter,
                                            g_vector.g_cart.z * two_pi_over_lattice_parameter};
            g_squared_values[g_index] =
                g_cartesian_vectors[g_index][0] * g_cartesian_vectors[g_index][0] +
                g_cartesian_vectors[g_index][1] * g_cartesian_vectors[g_index][1] +
                g_cartesian_vectors[g_index][2] * g_cartesian_vectors[g_index][2];
        }

        // Pre-compute |G| values and inverse |G| for chain rule
        std::vector<Real> inverse_g_magnitude(number_of_g_vectors, 0.0);
        for (int g_index = first_nonzero_g_index; g_index < number_of_g_vectors; g_index++) {
            if (g_vector_magnitudes[g_index] > math::tolerances::zero_vector_magnitude_threshold) {
                inverse_g_magnitude[g_index] = 1.0 / g_vector_magnitudes[g_index];
            }
        }

        /**
         * One species type at a time, to bound the per-species dQ buffer below.
         * Note that Q_ij^l(|G|) and its |G|-derivative are NOT pre-tabulated: both
         * are interpolated inside the G loop, once per (pair, LM, G).
         */

        // For each polarization direction beta:
        for (int polarization_index_b = 0; polarization_index_b < 3; polarization_index_b++) {

            // Pre-compute dYlm/dG_beta for all G-vectors (one batch call)
            std::vector<std::vector<Real>> spherical_harmonic_derivatives_beta;
            compute_spherical_harmonic_derivatives_batch(number_of_lm_components,
                                                         number_of_g_vectors,
                                                         g_cartesian_vectors,
                                                         g_squared_values,
                                                         polarization_index_b,
                                                         spherical_harmonic_derivatives_beta);

            // Process each unique species type once
            int number_of_species = static_cast<int>(
                quantum_system_.pseudopotential_basis().pseudopotential_set().size());
            for (int species_type_index = 0; species_type_index < number_of_species;
                 species_type_index++) {
                if (!quantum_system_.pseudopotential_basis()
                         .pseudopotential_set()[species_type_index]
                         .has_augmentation_charges())
                    continue;

                int number_of_projectors =
                    projector_mapping_table.number_of_channels(species_type_index);
                int number_of_beta_functions = quantum_system_.pseudopotential_basis()
                                                   .pseudopotential_set()[species_type_index]
                                                   .number_of_projectors;
                int maximum_angular_momentum_q_plus_one =
                    augmentation_q_table.maximum_angular_momentum_q() + 1;
                int number_of_lm_projectors =
                    maximum_angular_momentum_q_plus_one * maximum_angular_momentum_q_plus_one;

                /**
                 * Compute dQ_ij(G)/dG_beta for all (i,j) pairs and all G-vectors:
                 * one complex vector of length Ng per pair. The contraction below
                 * needs every G of a pair at once, and the pair set is fixed per
                 * species while the atoms that consume it are not.
                 */
                int number_of_triangle_pairs =
                    number_of_projectors * (number_of_projectors + 1) / 2;
                // Flat storage: dqg[pair_triangle_index * Ng + g_index]
                std::vector<Complex> dqg(static_cast<size_t>(number_of_triangle_pairs) *
                                             number_of_g_vectors,
                                         Complex(0.0, 0.0));

                int triangle_pair_index = 0;
                for (int projector_index_i = 0; projector_index_i < number_of_projectors;
                     projector_index_i++) {
                    for (int projector_index_j = projector_index_i;
                         projector_index_j < number_of_projectors;
                         projector_index_j++, triangle_pair_index++) {

                        int combined_lm_index_i = projector_mapping_table.combined_lm_index(
                            species_type_index, projector_index_i);
                        int combined_lm_index_j = projector_mapping_table.combined_lm_index(
                            species_type_index, projector_index_j);
                        int angular_momentum_i = projector_mapping_table.angular_momentum(
                            species_type_index, projector_index_i);
                        int angular_momentum_j = projector_mapping_table.angular_momentum(
                            species_type_index, projector_index_j);
                        int beta_index_i = projector_mapping_table.to_beta_index(species_type_index,
                                                                                 projector_index_i);
                        int beta_index_j = projector_mapping_table.to_beta_index(species_type_index,
                                                                                 projector_index_j);
                        int projector_pair_index =
                            beta_index_i * number_of_beta_functions + beta_index_j;

                        if (combined_lm_index_i >= number_of_lm_projectors ||
                            combined_lm_index_j >= number_of_lm_projectors)
                            continue;
                        if (gaunt_table.number_of_nonzero_product_indices[combined_lm_index_i]
                                                                         [combined_lm_index_j] == 0)
                            continue;

                        Complex* dqg_pair =
                            &dqg[static_cast<size_t>(triangle_pair_index) * number_of_g_vectors];

                        // For each Gaunt term, sweep over ALL G-vectors
                        for (int gaunt_index = 0;
                             gaunt_index <
                             gaunt_table.number_of_nonzero_product_indices[combined_lm_index_i]
                                                                          [combined_lm_index_j];
                             gaunt_index++) {
                            int product_lm_index =
                                gaunt_table
                                    .nonzero_product_indices[combined_lm_index_i]
                                                            [combined_lm_index_j][gaunt_index];
                            Real gaunt_coefficient_value =
                                gaunt_table.coefficients[product_lm_index][combined_lm_index_i]
                                                        [combined_lm_index_j];
                            int angular_momentum =
                                static_cast<int>(std::sqrt(static_cast<double>(product_lm_index)));
                            if (angular_momentum <
                                    std::abs(angular_momentum_i - angular_momentum_j) ||
                                angular_momentum > angular_momentum_i + angular_momentum_j ||
                                (angular_momentum + angular_momentum_i + angular_momentum_j) % 2 !=
                                    0)
                                continue;

                            int phase_type = angular_momentum % 4;

                            // Vectorized inner loop over G-vectors
                            for (int g_index = first_nonzero_g_index; g_index < number_of_g_vectors;
                                 g_index++) {
                                Real g_abs = g_vector_magnitudes[g_index];
                                if (g_abs < math::tolerances::zero_vector_magnitude_threshold)
                                    continue;

                                Real augmentation_function_value =
                                    augmentation_q_table.interpolate_augmentation_function(
                                        species_type_index,
                                        projector_pair_index,
                                        angular_momentum,
                                        g_abs);

                                Real augmentation_function_derivative_value =
                                    augmentation_q_table
                                        .interpolate_augmentation_function_derivative(
                                            species_type_index,
                                            projector_pair_index,
                                            angular_momentum,
                                            g_abs);

                                Real ylm_value =
                                    spherical_harmonics_table.value(g_index, product_lm_index);
                                Real dylm_value =
                                    spherical_harmonic_derivatives_beta[g_index][product_lm_index];

                                Real g_beta = g_cartesian_vectors[g_index][polarization_index_b];
                                Real derivative_contribution =
                                    dylm_value * augmentation_function_value +
                                    ylm_value * augmentation_function_derivative_value * g_beta *
                                        inverse_g_magnitude[g_index];

                                Real weighted = gaunt_coefficient_value * derivative_contribution;

                                switch (phase_type) {
                                    case 0:
                                        dqg_pair[g_index] += Complex(weighted, 0.0);
                                        break;
                                    case 1:
                                        dqg_pair[g_index] += Complex(0.0, -weighted);
                                        break;
                                    case 2:
                                        dqg_pair[g_index] += Complex(-weighted, 0.0);
                                        break;
                                    case 3:
                                        dqg_pair[g_index] += Complex(0.0, weighted);
                                        break;
                                }
                            }
                        }
                    }
                }

                // Now for each atom of this species, contract dqg with ρ_ij and V*(G)*SF
                for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                    if (crystal_structure_.species_type_index(atom_index) != species_type_index)
                        continue;

                    // Compute ρ_ij-weighted dQ(G): aux(G) = sum_{ij} D_ij * dQ_ij(G)/dG_beta
                    std::vector<Complex> aux(number_of_g_vectors, Complex(0.0, 0.0));

                    triangle_pair_index = 0;
                    for (int projector_index_i = 0; projector_index_i < number_of_projectors;
                         projector_index_i++) {
                        for (int projector_index_j = projector_index_i;
                             projector_index_j < number_of_projectors;
                             projector_index_j++, triangle_pair_index++) {
                            int projector_pair_linear =
                                projector_index_i * number_of_projectors + projector_index_j;
                            Real projector_overlap_coefficient =
                                (projector_pair_linear <
                                 static_cast<int>(projector_overlap_coefficients.cols()))
                                    ? projector_overlap_coefficients(atom_index,
                                                                     projector_pair_linear)
                                    : 0.0;
                            if (std::abs(projector_overlap_coefficient) <
                                math::tolerances::numerical_zero_threshold)
                                continue;
                            Real symmetry_factor =
                                (projector_index_i == projector_index_j) ? 1.0 : 2.0;
                            Real coefficient = projector_overlap_coefficient * symmetry_factor;

                            const Complex* dqg_pair =
                                &dqg[static_cast<size_t>(triangle_pair_index) *
                                     number_of_g_vectors];
                            for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
                                aux[g_index] += coefficient * dqg_pair[g_index];
                            }
                        }
                    }

                    // Multiply by V*(G) * structure_factor
                    Vector3D atomic_position =
                        crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
                    for (int g_index = first_nonzero_g_index; g_index < number_of_g_vectors;
                         g_index++) {
                        const auto& g_vector = planewave_basis_.g_vectors()[g_index];
                        Real phase_angle = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                        Complex structure_factor = compute_phase_factor(phase_angle);
                        aux[g_index] *= std::conj(total_potential_g[g_index]) * structure_factor;
                    }

                    // Contract: sigma_{alpha, beta} += sum_G G_alpha * aux(G)
                    for (int polarization_index_a = 0; polarization_index_a < 3;
                         polarization_index_a++) {
                        Real sigma_component = 0.0;
                        Real reciprocal_space_multiplicity =
                            planewave_basis_.is_gamma_only() ? 2.0 : 1.0;
                        for (int g_index = first_nonzero_g_index; g_index < number_of_g_vectors;
                             g_index++) {
                            Real g_alpha = g_cartesian_vectors[g_index][polarization_index_a];
                            sigma_component += reciprocal_space_multiplicity *
                                               (Complex(g_alpha, 0.0) * aux[g_index]).real();
                        }
                        result_.augmentation[polarization_index_a][polarization_index_b] +=
                            sigma_component;
                    }
                }
            }
        }
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveKohnShamQuantumSystem& quantum_system_;
    const std::vector<std::vector<Complex>>& potential_g_by_spin_;
    const std::vector<Matrix<Real>>& coefficients_by_spin_;
    StressTensorContributions& result_;
};

} // namespace q3
