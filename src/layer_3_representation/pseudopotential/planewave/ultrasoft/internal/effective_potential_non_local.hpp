#pragma once

/**
 * PlanewaveEffectiveNonlocalPotential — planewave implementation of the effective
 * D-matrix computation (QE's "newd").
 *
 * Inherits from the abstract EffectiveNonlocalPotential interface.
 * Stores the ultrasoft-specific dependencies (augmentation tables, spherical
 * harmonics, Gaunt coefficients, projector mappings) as members and delegates
 * the actual computation to the existing free function.
 *
 * Computes the effective D-matrix for all atoms by combining the bare
 * D-ion matrix with the G-space integral of V_eff(G) * Q_ij(G).
 *
 * For norm-conserving species, copies the bare d_ion_matrix.
 * For ultrasoft species, adds ∫ V_eff(G) * Q_ij(G) * e^{-iG·τ} dG.
 */

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/coefficients.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/spherical_harmonics_table.hpp"
#include "layer_3_representation/pseudopotential/planewave/ultrasoft/internal/charge_density_augmentation.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/internal/effective_potential_non_local.hpp"

namespace q3 {

// Forward declaration of free function used by the wrapper class below.
inline std::vector<std::vector<Real>> compute_ultrasoft_effective_d_matrix(
    const Crystal& crystal_structure,
    const PlanewaveBasis& planewave_basis,
    const PseudopotentialSet& pseudopotential_set,
    const std::vector<Complex>& total_potential_g,
    const AugmentationQTable& augmentation_q_table,
    const SphericalHarmonicsTable& spherical_harmonics_table,
    const GauntCoefficientTable& gaunt_table,
    const std::vector<int>& projector_channels_per_type,
    const std::vector<std::vector<int>>& projector_to_beta_index,
    const std::vector<std::vector<int>>& projector_combined_lm_index,
    const std::vector<std::vector<int>>& projector_angular_momentum);

/**
 * Planewave implementation of the effective D-matrix computation.
 *
 * Stores ultrasoft-specific dependencies as members and implements
 * the abstract EffectiveNonlocalPotential::compute() interface.
 */
class PlanewaveEffectiveNonlocalPotential : public EffectiveNonlocalPotential {
  public:
    PlanewaveEffectiveNonlocalPotential(
        const PseudopotentialSet& pseudopotential_set,
        const AugmentationQTable& augmentation_q_table,
        const SphericalHarmonicsTable& spherical_harmonics_table,
        const GauntCoefficientTable& gaunt_table,
        const std::vector<int>& projector_channels_per_type,
        const std::vector<std::vector<int>>& projector_to_beta_index,
        const std::vector<std::vector<int>>& projector_combined_lm_index,
        const std::vector<std::vector<int>>& projector_angular_momentum)
        : pseudopotential_set_(pseudopotential_set)
        , augmentation_q_table_(augmentation_q_table)
        , spherical_harmonics_table_(spherical_harmonics_table)
        , gaunt_table_(gaunt_table)
        , projector_channels_per_type_(projector_channels_per_type)
        , projector_to_beta_index_(projector_to_beta_index)
        , projector_combined_lm_index_(projector_combined_lm_index)
        , projector_angular_momentum_(projector_angular_momentum) {}

    ~PlanewaveEffectiveNonlocalPotential() override = default;

    /// Compute the effective D-matrix for all atoms.
    [[nodiscard]] std::vector<std::vector<Real>>
    compute(const Crystal& crystal_structure,
            const PlanewaveBasis& planewave_basis,
            const std::vector<Complex>& total_potential_g) const override {
        return compute_ultrasoft_effective_d_matrix(crystal_structure,
                                                    planewave_basis,
                                                    pseudopotential_set_,
                                                    total_potential_g,
                                                    augmentation_q_table_,
                                                    spherical_harmonics_table_,
                                                    gaunt_table_,
                                                    projector_channels_per_type_,
                                                    projector_to_beta_index_,
                                                    projector_combined_lm_index_,
                                                    projector_angular_momentum_);
    }

  private:
    const PseudopotentialSet& pseudopotential_set_;
    const AugmentationQTable& augmentation_q_table_;
    const SphericalHarmonicsTable& spherical_harmonics_table_;
    const GauntCoefficientTable& gaunt_table_;
    const std::vector<int>& projector_channels_per_type_;
    const std::vector<std::vector<int>>& projector_to_beta_index_;
    const std::vector<std::vector<int>>& projector_combined_lm_index_;
    const std::vector<std::vector<int>>& projector_angular_momentum_;
};

/**
 * Compute the effective D-matrix (QE's "newd") for all atoms.
 *
 * @return effective_d_matrix[atom_index][projector_i * number_of_projectors + projector_j]
 */
inline std::vector<std::vector<Real>> compute_ultrasoft_effective_d_matrix(
    const Crystal& crystal_structure,
    const PlanewaveBasis& planewave_basis,
    const PseudopotentialSet& pseudopotential_set,
    const std::vector<Complex>& total_potential_g,
    const AugmentationQTable& augmentation_q_table,
    const SphericalHarmonicsTable& spherical_harmonics_table,
    const GauntCoefficientTable& gaunt_table,
    const std::vector<int>& projector_channels_per_type,
    const std::vector<std::vector<int>>& projector_to_beta_index,
    const std::vector<std::vector<int>>& projector_combined_lm_index,
    const std::vector<std::vector<int>>& projector_angular_momentum) {

    int number_of_atoms = crystal_structure.number_of_atoms();
    int number_of_g_vectors = planewave_basis.number_of_g_vectors();
    Real cell_volume = crystal_structure.cell_volume();
    const auto& g_vector_magnitudes = planewave_basis.g_vector_magnitudes();

    int maximum_angular_momentum_q_plus_one = augmentation_q_table.maximum_angular_momentum_q() + 1;
    int number_of_lm_projectors =
        maximum_angular_momentum_q_plus_one * maximum_angular_momentum_q_plus_one;

    std::vector<std::vector<Real>> effective_d_matrix(number_of_atoms);

    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        int species_type_index = crystal_structure.species_type_index(atom_index);
        int number_of_projectors_per_species =
            pseudopotential_set[species_type_index].number_of_projectors;
        int number_of_projectors = projector_channels_per_type[species_type_index];
        effective_d_matrix[atom_index].assign(number_of_projectors * number_of_projectors, 0.0);

        Vector3D atomic_position = crystal_structure.convert_from_crystal_to_cartesian(atom_index);

        // Initialize with dvan (bare D matrix)
        for (int projector_index_i = 0; projector_index_i < number_of_projectors;
             projector_index_i++) {
            for (int projector_index_j = 0; projector_index_j < number_of_projectors;
                 projector_index_j++) {
                if (projector_combined_lm_index[species_type_index][projector_index_i] ==
                    projector_combined_lm_index[species_type_index][projector_index_j]) {
                    effective_d_matrix[atom_index][projector_index_i * number_of_projectors +
                                                   projector_index_j] =
                        pseudopotential_set[species_type_index].d_ion_matrix
                            [projector_to_beta_index[species_type_index][projector_index_i]]
                            [projector_to_beta_index[species_type_index][projector_index_j]];
                }
            }
        }

        // Precompute V(G) × structure_factor(G) for this atom
        std::vector<Complex> potential_g_structure_factor_product(number_of_g_vectors);
        for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
            const auto& g_vector = planewave_basis.g_vectors()[g_index];
            Real phase_angle = constants::two_pi * g_vector.g_cart.dot(atomic_position);
            potential_g_structure_factor_product[g_index] =
                total_potential_g[g_index] * compute_phase_factor(phase_angle);
        }

        // Add G-space integral: ∫ V_eff(G) * Q_ij(G) * e^{-iG·τ} dG
        for (int projector_index_i = 0; projector_index_i < number_of_projectors;
             projector_index_i++) {
            for (int projector_index_j = projector_index_i;
                 projector_index_j < number_of_projectors;
                 projector_index_j++) {
                int beta_index_i = projector_to_beta_index[species_type_index][projector_index_i];
                int beta_index_j = projector_to_beta_index[species_type_index][projector_index_j];
                int combined_lm_index_i =
                    projector_combined_lm_index[species_type_index][projector_index_i];
                int combined_lm_index_j =
                    projector_combined_lm_index[species_type_index][projector_index_j];
                int angular_momentum_i =
                    projector_angular_momentum[species_type_index][projector_index_i];
                int angular_momentum_j =
                    projector_angular_momentum[species_type_index][projector_index_j];

                int projector_pair_index =
                    beta_index_i * number_of_projectors_per_species + beta_index_j;

                if (combined_lm_index_i >= number_of_lm_projectors ||
                    combined_lm_index_j >= number_of_lm_projectors) {
                    continue;
                }
                if (gaunt_table.number_of_nonzero_product_indices[combined_lm_index_i]
                                                                 [combined_lm_index_j] == 0) {
                    continue;
                }

                // Pre-compute valid Gaunt contributions outside the G-vector loop
                struct GauntTerm {
                    int product_lm_index;
                    int angular_momentum;
                    int projector_pair_index_for_augmentation_function;
                    Real gaunt_coefficient_value;
                };
                std::vector<GauntTerm> valid_gaunt_terms;
                for (int gaunt_index = 0;
                     gaunt_index <
                     gaunt_table.number_of_nonzero_product_indices[combined_lm_index_i]
                                                                  [combined_lm_index_j];
                     gaunt_index++) {
                    int product_lm_index =
                        gaunt_table.nonzero_product_indices[combined_lm_index_i]
                                                           [combined_lm_index_j][gaunt_index];
                    Real gaunt_coefficient_value =
                        gaunt_table.coefficients[product_lm_index][combined_lm_index_i]
                                                [combined_lm_index_j];
                    int angular_momentum =
                        static_cast<int>(std::sqrt(static_cast<double>(product_lm_index)));
                    if (angular_momentum < std::abs(angular_momentum_i - angular_momentum_j) ||
                        angular_momentum > angular_momentum_i + angular_momentum_j ||
                        (angular_momentum + angular_momentum_i + angular_momentum_j) % 2 != 0) {
                        continue;
                    }
                    valid_gaunt_terms.push_back({product_lm_index,
                                                 angular_momentum,
                                                 projector_pair_index,
                                                 gaunt_coefficient_value});
                }

                if (valid_gaunt_terms.empty()) {
                    continue;
                }

                Complex sum_integral(0.0, 0.0);
                for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
                    Real q_g_real = 0.0;
                    Real q_g_imag = 0.0;
                    Real g_magnitude = g_vector_magnitudes[g_index];
                    for (const auto& term : valid_gaunt_terms) {
                        Real augmentation_function_value =
                            augmentation_q_table.interpolate_augmentation_function(
                                species_type_index,
                                term.projector_pair_index_for_augmentation_function,
                                term.angular_momentum,
                                g_magnitude);
                        Real spherical_harmonic_value =
                            spherical_harmonics_table.value(g_index, term.product_lm_index);
                        Real integrand_contribution =
                            augmentation_function_value * spherical_harmonic_value;
                        switch (term.angular_momentum % 4) {
                            case 0:
                                q_g_real += term.gaunt_coefficient_value * integrand_contribution;
                                break;
                            case 1:
                                q_g_imag -= term.gaunt_coefficient_value * integrand_contribution;
                                break;
                            case 2:
                                q_g_real -= term.gaunt_coefficient_value * integrand_contribution;
                                break;
                            case 3:
                                q_g_imag += term.gaunt_coefficient_value * integrand_contribution;
                                break;
                        }
                    }
                    sum_integral += potential_g_structure_factor_product[g_index] *
                                    Complex(q_g_real, -q_g_imag);
                }

                effective_d_matrix[atom_index]
                                  [projector_index_i * number_of_projectors + projector_index_j] +=
                    std::real(sum_integral) * cell_volume;
                if (projector_index_i != projector_index_j) {
                    effective_d_matrix[atom_index][projector_index_j * number_of_projectors +
                                                   projector_index_i] +=
                        std::real(sum_integral) * cell_volume;
                }
            }
        }
    }

    return effective_d_matrix;
}

} // namespace q3
