#pragma once

#include <cmath>
#include <complex>

#include "io/logger.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

/**
 * Compute the Ewald ion-ion electrostatic energy using both
 * G-space and real-space sums with optimal convergence parameter.
 *
 * Implements the Ewald summation of P. P. Ewald, Ann. Phys. 369, 253 (1921).
 */
inline Real compute_ewald(const Crystal& crystal_structure, const PlanewaveBasis& planewave_basis) {
    using namespace constants;
    int number_of_atoms = crystal_structure.number_of_atoms();
    int number_of_g_vectors = planewave_basis.number_of_g_vectors();
    Real two_pi_over_lattice_parameter_squared =
        crystal_structure.two_pi_over_lattice_parameter_squared();
    Real cell_volume = crystal_structure.cell_volume();
    Real lattice_parameter = crystal_structure.lattice_parameter();
    const Matrix3x3& lattice_vectors = crystal_structure.lattice_vectors();

    Real charge = 0.0;
    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        charge +=
            crystal_structure.valence_charge(crystal_structure.species_type_index(atom_index));
    }

    // Find optimal alpha for convergence
    Real g_cutoff_magnitude =
        planewave_basis.charge_density_energy_cutoff() / two_pi_over_lattice_parameter_squared;
    Real alpha = 2.9;
    while (alpha > 0.1) {
        alpha -= 0.1;
        Real upperbound = 2.0 * charge * charge * std::sqrt(2.0 * alpha / pi) *
                          std::erfc(std::sqrt(two_pi_over_lattice_parameter_squared *
                                              g_cutoff_magnitude / (4.0 * alpha)));
        if (upperbound < math::tolerances::ewald_convergence_threshold)
            break;
    }
    if (alpha < 0.1)
        alpha = 0.1;

    // G-space sum
    Real ewald_g_space_sum = 0.0;
    int first_nonzero_g_index = planewave_basis.first_nonzero_g_index();

    if (first_nonzero_g_index == 1) {
        ewald_g_space_sum = -charge * charge / alpha / 4.0;
    }

    for (int g_index = first_nonzero_g_index; g_index < number_of_g_vectors; g_index++) {
        Complex charge_density_coefficient(0.0, 0.0);
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            Vector3D atomic_position =
                crystal_structure.convert_from_crystal_to_cartesian(atom_index);
            const auto& g_vector = planewave_basis.g_vectors()[g_index];
            Real phase_angle = two_pi * g_vector.g_cart.dot(atomic_position);
            charge_density_coefficient +=
                crystal_structure.valence_charge(crystal_structure.species_type_index(atom_index)) *
                compute_phase_factor(phase_angle);
        }
        Real g_squared = planewave_basis.g_vectors()[g_index].magnitude_squared *
                         two_pi_over_lattice_parameter_squared;
        Real term =
            std::norm(charge_density_coefficient) * std::exp(-g_squared / alpha / 4.0) / g_squared;
        /**
         * For gamma-only half-sphere: each G≠0 represents both G and -G,
         * and |ρ(-G)|² = |ρ(G)|² by Hermitian symmetry, so double the contribution.
         */
        if (planewave_basis.is_gamma_only()) {
            term *= 2.0;
        }
        ewald_g_space_sum += term;
    }
    ewald_g_space_sum = 2.0 * two_pi / cell_volume * ewald_g_space_sum;

    // Self-energy correction
    if (first_nonzero_g_index == 1) {
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            ewald_g_space_sum -=
                crystal_structure.valence_charge(crystal_structure.species_type_index(atom_index)) *
                crystal_structure.valence_charge(crystal_structure.species_type_index(atom_index)) *
                std::sqrt(8.0 / two_pi * alpha);
        }
    }

    // R-space sum
    Real ewald_r_space_sum = 0.0;
    Real rmax_bohr = 4.0 / std::sqrt(alpha);
    Real rmax_alat = rmax_bohr / lattice_parameter;
    Real volume_alat =
        std::abs(lattice_vectors[0].dot(lattice_vectors[1].cross(lattice_vectors[2])));

    int limit_x = std::max(
        1,
        static_cast<int>(std::ceil(rmax_alat * lattice_vectors[1].cross(lattice_vectors[2]).norm() /
                                   volume_alat)));
    int limit_y = std::max(
        1,
        static_cast<int>(std::ceil(rmax_alat * lattice_vectors[2].cross(lattice_vectors[0]).norm() /
                                   volume_alat)));
    int limit_z = std::max(
        1,
        static_cast<int>(std::ceil(rmax_alat * lattice_vectors[0].cross(lattice_vectors[1]).norm() /
                                   volume_alat)));

    for (int atom_index_a = 0; atom_index_a < number_of_atoms; atom_index_a++) {
        for (int atom_index_b = 0; atom_index_b < number_of_atoms; atom_index_b++) {
            /**
             * The image loop is sized for a separation in the home cell, and positions
             * outside [0, 1) are legal input; see Crystal::home_cell_separation().
             */
            Vector3D position_difference =
                crystal_structure.home_cell_separation(atom_index_a, atom_index_b);

            for (int lattice_index_x = -limit_x; lattice_index_x <= limit_x; lattice_index_x++) {
                for (int lattice_index_y = -limit_y; lattice_index_y <= limit_y;
                     lattice_index_y++) {
                    for (int lattice_index_z = -limit_z; lattice_index_z <= limit_z;
                         lattice_index_z++) {
                        if (atom_index_a == atom_index_b && lattice_index_x == 0 &&
                            lattice_index_y == 0 && lattice_index_z == 0)
                            continue;
                        Vector3D real_space_vector;
                        for (int dimension_index = 0; dimension_index < 3; dimension_index++) {
                            real_space_vector[dimension_index] =
                                position_difference[dimension_index] +
                                lattice_index_x * lattice_vectors[0][dimension_index] +
                                lattice_index_y * lattice_vectors[1][dimension_index] +
                                lattice_index_z * lattice_vectors[2][dimension_index];
                        }
                        Real real_space_distance = real_space_vector.norm() * lattice_parameter;
                        if (real_space_distance < math::tolerances::zero_vector_magnitude_threshold)
                            continue;
                        if (real_space_distance > rmax_bohr)
                            continue;
                        ewald_r_space_sum +=
                            crystal_structure.valence_charge(
                                crystal_structure.species_type_index(atom_index_a)) *
                            crystal_structure.valence_charge(
                                crystal_structure.species_type_index(atom_index_b)) *
                            std::erfc(std::sqrt(alpha) * real_space_distance) / real_space_distance;
                    }
                }
            }
        }
    }

    return 0.5 * electron_charge_squared_rydberg * (ewald_g_space_sum + ewald_r_space_sum);
}

} // namespace q3
