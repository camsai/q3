#pragma once

/**
 * Ewald (ion-ion) force contribution.
 *
 * Computes the Hellmann-Feynman force contribution from the Ewald summation
 * of the ion-ion Coulomb interaction. Split into G-space and R-space sums
 * with an optimal convergence parameter α.
 *
 * This contribution is independent of pseudopotential type.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class EwaldForceCalculator {
  public:
    EwaldForceCalculator(const Crystal& crystal_structure, const PlanewaveBasis& planewave_basis)
        : crystal_structure_(crystal_structure), planewave_basis_(planewave_basis) {}

    [[nodiscard]] std::vector<Vector3D> compute() const {
        using namespace constants;
        Real lattice_parameter = crystal_structure_.lattice_parameter();
        const Matrix3x3& lattice_vectors = crystal_structure_.lattice_vectors();

        Real total_charge = 0.0;
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            total_charge += crystal_structure_.valence_charge(
                crystal_structure_.species_type_index(atom_index));
        }

        // Optimal convergence parameter
        Real cutoff_g_space = planewave_basis_.charge_density_energy_cutoff() /
                              crystal_structure_.two_pi_over_lattice_parameter_squared();
        static constexpr Real initial_ewald_convergence_parameter = 2.9;
        static constexpr Real ewald_convergence_parameter_step = 0.1;
        Real alpha = initial_ewald_convergence_parameter;
        while (alpha > ewald_convergence_parameter_step) {
            alpha -= ewald_convergence_parameter_step;
            Real upper_bound =
                2.0 * total_charge * total_charge * std::sqrt(2.0 * alpha / pi) *
                std::erfc(std::sqrt(crystal_structure_.two_pi_over_lattice_parameter_squared() *
                                    cutoff_g_space / (4.0 * alpha)));
            if (upper_bound < math::tolerances::ewald_convergence_threshold) {
                break;
            }
        }
        if (alpha < ewald_convergence_parameter_step) {
            alpha = ewald_convergence_parameter_step;
        }

        std::vector<Vector3D> forces(crystal_structure_.number_of_atoms(), {0.0, 0.0, 0.0});
        int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();

        // G-space sum
        for (int g_vector_index = first_nonzero_g_index;
             g_vector_index < planewave_basis_.number_of_g_vectors();
             g_vector_index++) {
            Real g_vector_squared = planewave_basis_.g_vectors()[g_vector_index].magnitude_squared *
                                    crystal_structure_.two_pi_over_lattice_parameter_squared();
            /**
             * factor = 2 · exp(-G²/4α) / G², doubled again on a half sphere.
             *
             * The 2 is not about the sphere, and it is not the Rydberg e²
             * either — that is applied separately below. It is the factor from
             * differentiating a modulus squared: ∂|ρ(G)|²/∂τ = 2 Re[ρ* ∂ρ/∂τ].
             * So it is correct for the full-sphere G list Q3 normally builds.
             * Gamma-only stores only the positive half sphere
             * (`g_vectors_grid.hpp`), and
             * the summand here is *even* under G → −G (the imaginary part is
             * odd and G_α is odd), so the missing half contributes exactly as
             * much again.
             *
             * Without this the Ewald force is exactly half at Gamma while the
             * Ewald *energy* — which depends only on geometry — is unchanged:
             * measured -0.02549078 against a −dE_ewald/dy of -0.05098040.
             */
            Real half_sphere_scale = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;
            Real factor = std::exp(-g_vector_squared / alpha / 4.0) / g_vector_squared * 2.0 *
                          half_sphere_scale;

            Complex valence_charge_density_g(0.0, 0.0);
            for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                 atom_index++) {
                Vector3D atomic_position =
                    crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
                Real phase = two_pi * planewave_basis_.g_vectors()[g_vector_index].g_cart.dot(
                                          atomic_position);
                valence_charge_density_g += crystal_structure_.valence_charge(
                                                crystal_structure_.species_type_index(atom_index)) *
                                            compute_phase_factor(phase);
            }

            for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                 atom_index++) {
                Vector3D atomic_position =
                    crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
                Real phase = two_pi * planewave_basis_.g_vectors()[g_vector_index].g_cart.dot(
                                          atomic_position);
                Complex structure_factor_i = compute_phase_factor(phase);
                Real valence_charge_i = crystal_structure_.valence_charge(
                    crystal_structure_.species_type_index(atom_index));

                for (int direction = 0; direction < 3; direction++) {
                    Real g_cartesian_component =
                        planewave_basis_.g_vectors()[g_vector_index].g_cart[direction] *
                        crystal_structure_.two_pi_over_lattice_parameter();
                    Real force_prefactor = valence_charge_i * factor * g_cartesian_component;
                    Complex complex_product =
                        valence_charge_density_g * std::conj(structure_factor_i);
                    /**
                     * F_{G,α} = +(e²⋅2π/Ω) factor Z_i G_α Im[S(G) e^{-iGτ_i}]
                     * complex_product = S(G) × conj(e^{iGτ_i}) = S(G) × e^{-iGτ_i}.
                     * Im[S e^{-iGτ}] = -Im[e^{+iGτ} S*] (conjugate of correct formula),
                     * combined with the F = -∂E/∂τ sign, the result is +=.
                     */
                    forces[atom_index][direction] += electron_charge_squared_rydberg * two_pi /
                                                     crystal_structure_.cell_volume() *
                                                     force_prefactor * complex_product.imag();
                }
            }
        }

        // R-space sum
        Real rmax_bohr = 4.0 / std::sqrt(alpha);
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
                Vector3D position_difference =
                    crystal_structure_.home_cell_separation(atom_index_a, atom_index_b);

                for (int lattice_vector_multiplier_1 = -limit_x;
                     lattice_vector_multiplier_1 <= limit_x;
                     lattice_vector_multiplier_1++) {
                    for (int lattice_vector_multiplier_2 = -limit_y;
                         lattice_vector_multiplier_2 <= limit_y;
                         lattice_vector_multiplier_2++) {
                        for (int lattice_vector_multiplier_3 = -limit_z;
                             lattice_vector_multiplier_3 <= limit_z;
                             lattice_vector_multiplier_3++) {
                            if (atom_index_a == atom_index_b && lattice_vector_multiplier_1 == 0 &&
                                lattice_vector_multiplier_2 == 0 &&
                                lattice_vector_multiplier_3 == 0) {
                                continue;
                            }
                            Vector3D displacement_vector;
                            for (int direction = 0; direction < 3; direction++) {
                                displacement_vector[direction] =
                                    position_difference[direction] +
                                    lattice_vector_multiplier_1 * lattice_vectors[0][direction] +
                                    lattice_vector_multiplier_2 * lattice_vectors[1][direction] +
                                    lattice_vector_multiplier_3 * lattice_vectors[2][direction];
                            }
                            Real distance_magnitude =
                                displacement_vector.norm() * lattice_parameter;
                            if (distance_magnitude <
                                math::tolerances::zero_vector_magnitude_threshold) {
                                continue;
                            }
                            if (distance_magnitude > rmax_bohr) {
                                continue;
                            }

                            Real argument = std::sqrt(alpha) * distance_magnitude;
                            Real factor =
                                crystal_structure_.valence_charge(
                                    crystal_structure_.species_type_index(atom_index_a)) *
                                crystal_structure_.valence_charge(
                                    crystal_structure_.species_type_index(atom_index_b)) *
                                (2.0 * argument / std::sqrt(pi) * std::exp(-argument * argument) +
                                 std::erfc(argument)) /
                                (distance_magnitude * distance_magnitude);

                            for (int direction = 0; direction < 3; direction++) {
                                forces[atom_index_a][direction] -=
                                    electron_charge_squared_rydberg * factor *
                                    displacement_vector[direction] * lattice_parameter /
                                    distance_magnitude;
                            }
                        }
                    }
                }
            }
        }

        return forces;
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
};

} // namespace q3
