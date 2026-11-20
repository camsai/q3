#pragma once

/**
 * Ewald (ion-ion) force contribution.
 *
 * Computes the Hellmann-Feynman force contribution from the Ewald summation
 * of the ion-ion Coulomb interaction. Split into G-space and R-space sums
 * with an optimal convergence parameter α.
 *
 * Differentiates the Ewald summation of P. P. Ewald, Ann. Phys. 369, 253
 * (1921), with respect to the atomic positions.
 *
 * This contribution is independent of pseudopotential type.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/structure_factor_calculator.hpp"

namespace q3 {

class EwaldForceCalculator {
  public:
    /// `structure_factor_calculator` forms the G-space sums over atoms; see compute().
    EwaldForceCalculator(const Crystal& crystal_structure,
                         const PlanewaveBasis& planewave_basis,
                         const StructureFactorCalculator& structure_factor_calculator)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , structure_factor_calculator_(structure_factor_calculator) {}

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

        int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();

        /**
         * G-space sum, over this rank's density G-vectors; completed across the pool below.
         *
         *   F_{G,α}(i) = (e²·2π/Ω) Z_i Σ_G factor(G) G_α Im[S(G) e^{+iGτ_i}],
         *   S(G) = Σ_j Z_j e^{-iGτ_j},  factor(G) = 2 · exp(-G²/4α) / G²,
         *
         * doubled again on a half sphere. The 2 in factor is not about the sphere, and it is
         * not the Rydberg e² either -- that is the prefactor. It is the factor from
         * differentiating a modulus squared: ∂|ρ(G)|²/∂τ = 2 Re[ρ* ∂ρ/∂τ]. So it is correct
         * for the full-sphere G list Q3 normally builds. Gamma-only stores only the positive
         * half sphere (`g_vectors_grid.hpp`), and the summand here is *even* under G → −G (the
         * imaginary part is odd and G_α is odd), so the missing half contributes exactly as
         * much again. Without this the Ewald force is exactly half at Gamma while the Ewald
         * *energy* -- which depends only on geometry -- is unchanged: measured -0.02549078
         * against a −dE_ewald/dy of -0.05098040.
         *
         * Both sums over the atoms, S(G) and the one against each atom's phases, are the
         * structure factor calculator's; what is formed here is per G-vector.
         */
        const int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<Real> valence_charges(static_cast<std::size_t>(number_of_atoms));
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            valence_charges[static_cast<std::size_t>(atom_index)] =
                crystal_structure_.valence_charge(
                    crystal_structure_.species_type_index(atom_index));
        }
        const LocalDensityGLayout& layout = planewave_basis_.local_density_g_layout();
        std::vector<int> g_vector_indices;
        g_vector_indices.reserve(static_cast<std::size_t>(layout.size()));
        for (int entry = 0; entry < layout.size(); entry++) {
            const int g_vector_index = layout.global_g_indices()[entry];
            if (g_vector_index >= first_nonzero_g_index) {
                g_vector_indices.push_back(g_vector_index);
            }
        }
        std::vector<std::vector<Complex>> phase_weights =
            structure_factor_calculator_.structure_factors(g_vector_indices, {valence_charges});
        const Real half_sphere_scale = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;
        const Real prefactor =
            electron_charge_squared_rydberg * two_pi / crystal_structure_.cell_volume();
        for (std::size_t entry = 0; entry < g_vector_indices.size(); entry++) {
            const auto& g_vector = planewave_basis_.g_vectors()[g_vector_indices[entry]];
            Real g_vector_squared = g_vector.magnitude_squared *
                                    crystal_structure_.two_pi_over_lattice_parameter_squared();
            Real factor = std::exp(-g_vector_squared / alpha / 4.0) / g_vector_squared * 2.0 *
                          half_sphere_scale;
            phase_weights[0][entry] *= prefactor * factor;
        }
        std::vector<Vector3D> forces = structure_factor_calculator_.gradient_phase_sums(
            g_vector_indices,
            phase_weights,
            std::vector<int>(static_cast<std::size_t>(number_of_atoms), 0));
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            for (int direction = 0; direction < 3; direction++) {
                forces[static_cast<std::size_t>(atom_index)][direction] *=
                    valence_charges[static_cast<std::size_t>(atom_index)] *
                    crystal_structure_.two_pi_over_lattice_parameter();
            }
        }
        planewave_basis_.complete_density_g_sums(forces);

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
    const StructureFactorCalculator& structure_factor_calculator_;
};

} // namespace q3
