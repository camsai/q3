#pragma once

/**
 * Local pseudopotential force contribution.
 *
 * Computes the Hellmann-Feynman force from the local (ionic) pseudopotential
 * acting on the electron charge density in reciprocal space.
 *
 * The algorithm is the same for norm-conserving and ultrasoft pseudopotential_set.
 */

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"

namespace q3 {

class LocalPseudopotentialForceCalculator {
  public:
    LocalPseudopotentialForceCalculator(const Crystal& crystal_structure,
                                        const PlanewaveBasis& planewave_basis,
                                        const NormConservingPlanewaveBasis& pseudopotential_basis)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis) {}

    std::vector<Vector3D> compute(const std::vector<Complex>& charge_density_g) {
        std::vector<Vector3D> forces(crystal_structure_.number_of_atoms(), {0.0, 0.0, 0.0});
        int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();
        /**
         * Gamma-only stores only the positive half sphere, and this
         * summand is even under G -> -G (the imaginary part is odd and
         * G_alpha is odd), so the omitted half contributes as much again.
         * Verified against QE's own force decomposition at Gamma
         * (`verbosity='high'`), where Q3 was exactly half.
         */
        Real half_sphere_scale = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;

        /**
         * Each rank sums over the density G-vectors of its own sticks and one reduction
         * completes the forces: the same terms, split across the pool instead of repeated
         * on every rank of it. Positions are converted once, and each phase is taken once.
         */
        const int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<Vector3D> atomic_positions(static_cast<std::size_t>(number_of_atoms));
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            atomic_positions[atom_index] =
                crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
        }
        const Real two_pi_over_lattice_parameter =
            crystal_structure_.two_pi_over_lattice_parameter();
        const Real cell_volume = crystal_structure_.cell_volume();
        const LocalDensityGLayout& layout = planewave_basis_.local_density_g_layout();

        for (int entry = 0; entry < layout.size(); entry++) {
            const int g_vector_index = layout.global_g_indices()[entry];
            if (g_vector_index < first_nonzero_g_index) {
                continue;
            }
            const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_positions[atom_index]);
                /**
                 * F_{α,na} = Ω · G_α · Im[ V_loc(G) · ρ(G) · e^{+iGτ} ]  (full sphere)
                 * Full-sphere sum over both G and -G: the -G term contributes the same as G,
                 * so no extra factor of 2 needed (it's already implicit in summing all G).
                 */
                Complex vloc_rho =
                    pseudopotential_basis_.local_potential_g(species_index)[g_vector_index] *
                    charge_density_g[g_vector_index];
                Complex exp_i_phase(std::cos(phase), std::sin(phase)); // e^{+iGτ}
                Complex contribution = vloc_rho * exp_i_phase;

                for (int direction = 0; direction < 3; direction++) {
                    Real g_cartesian_component =
                        g_vector.g_cart[direction] * two_pi_over_lattice_parameter; // G_α in a.u.
                    forces[atom_index][direction] += half_sphere_scale * cell_volume *
                                                     g_cartesian_component * contribution.imag();
                }
            }
        }
        planewave_basis_.complete_density_g_sums(forces);

        return forces;
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
};

} // namespace q3
