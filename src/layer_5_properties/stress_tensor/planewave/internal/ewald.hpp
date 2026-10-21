#pragma once

/**
 * Ewald stress tensor contribution.
 *
 * Computes the Ewald (ion-ion electrostatic) contribution to the stress tensor
 * using the Ewald summation technique in both G-space and R-space.
 *
 * Differentiates the Ewald summation of P. P. Ewald, Ann. Phys. 369, 253
 * (1921), with respect to strain, in the stress formalism of O. H. Nielsen &
 * R. M. Martin, Phys. Rev. B 32, 3780 (1985).
 *
 * This is a purely planewave contribution, independent of pseudopotential type.
 */

#include <cmath>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"

namespace q3 {

class EwaldStressTensorCalculator {
  public:
    EwaldStressTensorCalculator(const Crystal& crystal_structure,
                                const PlanewaveBasis& planewave_basis,
                                StressTensorContributions& result)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , result_(result) {}

    void compute() {
        Real lattice_parameter = crystal_structure_.lattice_parameter();
        Real gcutm = planewave_basis_.charge_density_energy_cutoff() /
                     crystal_structure_.two_pi_over_lattice_parameter_squared();
        Real total_charge = 0.0;
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            total_charge += crystal_structure_.valence_charge(
                crystal_structure_.species_type_index(atom_index));
        }

        Real alpha_ewald = 2.9;
        while (alpha_ewald > 0.1) {
            alpha_ewald -= 0.1;
            Real upperbound =
                constants::electron_charge_squared_rydberg * total_charge * total_charge *
                std::sqrt(2.0 * alpha_ewald / constants::pi) *
                std::erfc(std::sqrt(crystal_structure_.two_pi_over_lattice_parameter_squared() *
                                    gcutm / (4.0 * alpha_ewald)));
            if (upperbound < math::tolerances::ewald_convergence_threshold)
                break;
        }

        int g_vector_start = planewave_basis_.first_nonzero_g_index();
        Real reciprocal_space_multiplicity = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;

        Real stress_diagonal_contribution = 0.0;
        if (g_vector_start == 1) {
            stress_diagonal_contribution =
                constants::two_pi * constants::electron_charge_squared_rydberg /
                (4.0 * alpha_ewald) * (total_charge / crystal_structure_.cell_volume()) *
                (total_charge / crystal_structure_.cell_volume());
        }

        Real stress_xx = 0, stress_yx = 0, stress_yy = 0, stress_zx = 0, stress_zy = 0,
             stress_zz = 0;
        for (int g_vector_index = g_vector_start;
             g_vector_index < planewave_basis_.number_of_g_vectors();
             g_vector_index++) {
            const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
            Real g_vector_squared = g_vector.magnitude_squared *
                                    crystal_structure_.two_pi_over_lattice_parameter_squared();
            if (g_vector_squared < math::tolerances::numerical_zero_threshold)
                continue;
            Real g_squared_over_four_alpha = g_vector_squared / (4.0 * alpha_ewald);

            Complex structure_factor(0.0, 0.0);
            for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                 atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                Real valence_charge = crystal_structure_.valence_charge(species_index);
                Vector3D atomic_position =
                    crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
                Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                structure_factor += valence_charge * compute_phase_factor_conjugate(phase);
            }
            structure_factor /= Complex(crystal_structure_.cell_volume(), 0.0);

            Real ewald_stress_term = reciprocal_space_multiplicity * constants::two_pi *
                                     constants::electron_charge_squared_rydberg *
                                     std::exp(-g_squared_over_four_alpha) / g_vector_squared *
                                     std::norm(structure_factor);
            stress_diagonal_contribution -= ewald_stress_term;

            Real factor = ewald_stress_term *
                          crystal_structure_.two_pi_over_lattice_parameter_squared() * 2.0 *
                          (g_squared_over_four_alpha + 1.0) / g_vector_squared;
            stress_xx += factor * g_vector.g_cart.x * g_vector.g_cart.x;
            stress_yx += factor * g_vector.g_cart.y * g_vector.g_cart.x;
            stress_yy += factor * g_vector.g_cart.y * g_vector.g_cart.y;
            stress_zx += factor * g_vector.g_cart.z * g_vector.g_cart.x;
            stress_zy += factor * g_vector.g_cart.z * g_vector.g_cart.y;
            stress_zz += factor * g_vector.g_cart.z * g_vector.g_cart.z;
        }

        result_.ewald[0][0] = stress_diagonal_contribution + stress_xx;
        result_.ewald[0][1] = stress_yx;
        result_.ewald[1][1] = stress_diagonal_contribution + stress_yy;
        result_.ewald[0][2] = stress_zx;
        result_.ewald[1][2] = stress_zy;
        result_.ewald[2][2] = stress_diagonal_contribution + stress_zz;

        const Matrix3x3& lattice_vectors = crystal_structure_.lattice_vectors();
        Real rmax_bohr = 4.0 / std::sqrt(alpha_ewald);
        Real rmax_alat = rmax_bohr / lattice_parameter;
        Real volume_alat =
            std::abs(lattice_vectors[0].dot(lattice_vectors[1].cross(lattice_vectors[2])));

        int limit_x = std::max(
            1,
            static_cast<int>(std::ceil(
                rmax_alat * lattice_vectors[1].cross(lattice_vectors[2]).norm() / volume_alat)));
        int limit_y = std::max(
            1,
            static_cast<int>(std::ceil(
                rmax_alat * lattice_vectors[2].cross(lattice_vectors[0]).norm() / volume_alat)));
        int limit_z = std::max(
            1,
            static_cast<int>(std::ceil(
                rmax_alat * lattice_vectors[0].cross(lattice_vectors[1]).norm() / volume_alat)));

        for (int atom_index_a = 0; atom_index_a < crystal_structure_.number_of_atoms();
             atom_index_a++) {
            for (int atom_index_b = 0; atom_index_b < crystal_structure_.number_of_atoms();
                 atom_index_b++) {
                /**
                 * The image loop is sized for a separation in the home cell, and positions
                 * outside [0, 1) are legal input; see Crystal::home_cell_separation().
                 */
                Vector3D atom_displacement_cartesian =
                    crystal_structure_.home_cell_separation(atom_index_a, atom_index_b);

                for (int lattice_index_x = -limit_x; lattice_index_x <= limit_x;
                     lattice_index_x++) {
                    for (int lattice_index_y = -limit_y; lattice_index_y <= limit_y;
                         lattice_index_y++) {
                        for (int lattice_index_z = -limit_z; lattice_index_z <= limit_z;
                             lattice_index_z++) {
                            if (atom_index_a == atom_index_b && lattice_index_x == 0 &&
                                lattice_index_y == 0 && lattice_index_z == 0)
                                continue;
                            Vector3D displacement_vector;
                            for (int direction = 0; direction < 3; direction++) {
                                displacement_vector[direction] =
                                    atom_displacement_cartesian[direction] +
                                    (lattice_index_x * lattice_vectors[0][direction] +
                                     lattice_index_y * lattice_vectors[1][direction] +
                                     lattice_index_z * lattice_vectors[2][direction]);
                            }
                            Real displacement_magnitude =
                                displacement_vector.norm() * lattice_parameter;
                            if (displacement_magnitude <
                                math::tolerances::zero_vector_magnitude_threshold)
                                continue;
                            if (displacement_magnitude > rmax_bohr)
                                continue;

                            Real valence_charge_a = crystal_structure_.valence_charge(
                                crystal_structure_.species_type_index(atom_index_a));
                            Real valence_charge_b = crystal_structure_.valence_charge(
                                crystal_structure_.species_type_index(atom_index_b));
                            Real ewald_factor =
                                -constants::electron_charge_squared_rydberg /
                                (2.0 * crystal_structure_.cell_volume()) * lattice_parameter *
                                lattice_parameter * valence_charge_a * valence_charge_b /
                                (displacement_magnitude * displacement_magnitude *
                                 displacement_magnitude) *
                                (std::erfc(std::sqrt(alpha_ewald) * displacement_magnitude) +
                                 displacement_magnitude *
                                     std::sqrt(8.0 * alpha_ewald / constants::two_pi) *
                                     std::exp(-alpha_ewald * displacement_magnitude *
                                              displacement_magnitude));

                            for (int direction_a = 0; direction_a < 3; direction_a++) {
                                for (int direction_b = direction_a; direction_b < 3;
                                     direction_b++) {
                                    result_.ewald[direction_a][direction_b] +=
                                        ewald_factor * displacement_vector[direction_a] *
                                        displacement_vector[direction_b];
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int direction_a = 0; direction_a < 3; direction_a++) {
            for (int direction_b = direction_a + 1; direction_b < 3; direction_b++) {
                result_.ewald[direction_b][direction_a] = result_.ewald[direction_a][direction_b];
            }
        }

        for (int direction_a = 0; direction_a < 3; direction_a++) {
            for (int direction_b = 0; direction_b < 3; direction_b++) {
                result_.ewald[direction_a][direction_b] = -result_.ewald[direction_a][direction_b];
            }
        }
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    StressTensorContributions& result_;
};

} // namespace q3
