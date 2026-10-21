#pragma once

/**
 * The force from the compensation charge moving with its atom.
 *
 *   F_α = -∂/∂τ_α  Ω Σ_G V_eff*(G) n̂_I(G),   n̂_I(G) = A_I(G) e^{-iG·τ}
 *
 * with A_I(G) = Σ_ij ρ_ij Q_ij(G) the same object `add_augmentation_charges`
 * builds when it puts n̂ on the grid. Differentiating only the structure factor
 * gives
 *
 *   F_α = -Ω Σ_G G_α Im[ V_eff*(G) A_I(G) e^{-iG·τ} ]
 *
 * which is the local-potential force with V_loc → V_eff and the species form
 * factor → A_I(G).
 *
 * Implements the third term of Eq. (43) of K. Laasonen, A. Pasquarello,
 * R. Car, C. Lee & D. Vanderbilt, Phys. Rev. B 47, 10142 (1993), with ρ_ij
 * their Eq. (38) and the position dependence carried by the phase factor of
 * their Eq. (61).
 *
 * **This must be built from the same Q_ij(G) the density uses.** It used to be
 * assembled from `interpolate_qfunc`, a radial-only table indexed by *beta
 * projector* pairs, while ρ_ij is indexed by lm-expanded *channels*: the
 * strides disagreed, and the angular (-i)^L Y_LM structure was missing
 * entirely. The term came out at 3.4e-7 Ry/bohr against a nonlocal term of
 * 4.3e-2 — five orders of magnitude down, for a dataset whose compensation
 * charge carries most of the on-site density. Q3's PAW forces were a uniform
 * 1.24x overestimate as a result, matching its own energy derivative only to
 * that factor.
 *
 * For norm-conserving datasets there is no augmentation charge and this is
 * identically zero.
 */

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"

namespace q3 {

class UltrasoftForceCalculator {
  public:
    UltrasoftForceCalculator(const Crystal& crystal_structure,
                             const PlanewaveBasis& planewave_basis)
        : crystal_structure_(crystal_structure), planewave_basis_(planewave_basis) {}

    /**
     * @param potential_g_by_spin  One effective potential per spin channel.
     * @param coefficients_by_spin One coefficient matrix per spin channel, same order.
     *
     * Spin is summed *inside*, channel against matching channel. Passing the
     * spin-averaged potential and the spin-summed coefficients instead — which is
     * what this took until 2026-08-09 — drops
     * ∫(V↑−V↓)(ρ↑−ρ↓) dQ/dR / 2. That vanishes for a non-magnetic cell, which
     * is why it survived every unpolarized test, and does not vanish for a
     * magnetic one, where it left this term inconsistent with the
     * spin-resolved nonlocal force it must cancel against.
     */
    std::vector<Vector3D>
    compute(const NormConservingPlanewaveKohnShamQuantumSystem& quantum_system,
            const std::vector<std::vector<Complex>>& potential_g_by_spin,
            const std::vector<Matrix<Real>>& coefficients_by_spin) {
        int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<Vector3D> forces(number_of_atoms, {0.0, 0.0, 0.0});

        if (!quantum_system.has_ultrasoft())
            return forces;
        if (potential_g_by_spin.size() != coefficients_by_spin.size())
            throw std::runtime_error(
                "augmentation force: " + std::to_string(potential_g_by_spin.size()) +
                " potential channels against " + std::to_string(coefficients_by_spin.size()) +
                " occupancy channels; they must pair up spin by spin");

        const auto& ultrasoft_basis =
            static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(quantum_system)
                .ultrasoft_basis();

        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        Real two_pi_over_lattice_parameter = crystal_structure_.two_pi_over_lattice_parameter();
        Real cell_volume = crystal_structure_.cell_volume();
        /**
         * Gamma-only keeps only the positive half sphere, and this summand is
         * even under G -> -G, so the omitted half contributes as much again.
         * The same factor appears in the local, NLCC and Ewald forces.
         */
        Real half_sphere_scale = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;

        std::vector<Complex> augmentation_g(number_of_g_vectors);

        for (size_t spin_index = 0; spin_index < potential_g_by_spin.size(); spin_index++) {
            const std::vector<Complex>& total_potential_g = potential_g_by_spin[spin_index];
            const Matrix<Real>& projector_overlap_coefficients = coefficients_by_spin[spin_index];

            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                if (!ultrasoft_basis.has_q_functions_g(species_index))
                    continue;

                int number_of_channels =
                    quantum_system.pseudopotential_basis().number_of_projector_channels(
                        species_index);

                /**
                 * A_I(G), assembled exactly as `add_augmentation_charges` does:
                 * channel-indexed ρ_ij against the precomputed Q_ij(G), with the
                 * Q_ij = Q_ji symmetry folded into a single upper-triangle pass.
                 */
                std::fill(augmentation_g.begin(), augmentation_g.end(), Complex(0.0, 0.0));
                for (int channel_i = 0; channel_i < number_of_channels; channel_i++) {
                    int diagonal_pair = channel_i * number_of_channels + channel_i;
                    Real diagonal_projector_overlap_coefficient =
                        projector_overlap_coefficients(atom_index, diagonal_pair);
                    if (std::abs(diagonal_projector_overlap_coefficient) >=
                        math::tolerances::d_matrix_element_threshold) {
                        const Complex* q_g = ultrasoft_basis.q_function_g_data(
                            species_index, diagonal_pair, number_of_g_vectors);
                        for (int g_index = 0; g_index < number_of_g_vectors; g_index++)
                            augmentation_g[g_index] +=
                                diagonal_projector_overlap_coefficient * q_g[g_index];
                    }

                    for (int channel_j = channel_i + 1; channel_j < number_of_channels;
                         channel_j++) {
                        int pair_ij = channel_i * number_of_channels + channel_j;
                        int pair_ji = channel_j * number_of_channels + channel_i;
                        Real combined_projector_overlap_coefficient =
                            projector_overlap_coefficients(atom_index, pair_ij) +
                            projector_overlap_coefficients(atom_index, pair_ji);
                        if (std::abs(combined_projector_overlap_coefficient) <
                            math::tolerances::d_matrix_element_threshold)
                            continue;
                        const Complex* q_g = ultrasoft_basis.q_function_g_data(
                            species_index, pair_ij, number_of_g_vectors);
                        for (int g_index = 0; g_index < number_of_g_vectors; g_index++)
                            augmentation_g[g_index] +=
                                combined_projector_overlap_coefficient * q_g[g_index];
                    }
                }

                Vector3D atomic_position =
                    crystal_structure_.convert_from_crystal_to_cartesian(atom_index);

                /**
                 * G = 0 carries no force (G_α is zero there), but *which* index
                 * that is depends on the rank: once G-vectors are partitioned the
                 * local list does not start at G = 0, and
                 * `first_nonzero_g_index()` returns 0 on every rank that does not
                 * own it. Hardcoding 1 silently drops one legitimate G-vector per
                 * rank under band parallelism.
                 */
                for (int g_index = planewave_basis_.first_nonzero_g_index();
                     g_index < number_of_g_vectors;
                     g_index++) {
                    const auto& g_vector = planewave_basis_.g_vectors()[g_index];
                    Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                    Complex contribution = std::conj(total_potential_g[g_index]) *
                                           augmentation_g[g_index] * compute_phase_factor(phase);
                    Real weight = half_sphere_scale * cell_volume * contribution.imag();

                    for (int direction = 0; direction < 3; direction++) {
                        Real g_cartesian_component =
                            g_vector.g_cart[direction] * two_pi_over_lattice_parameter;
                        forces[atom_index][direction] -= g_cartesian_component * weight;
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
