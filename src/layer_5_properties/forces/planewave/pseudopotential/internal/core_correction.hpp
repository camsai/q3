#pragma once

/**
 * Core correction (NLCC) force contribution.
 *
 * Computes the Hellmann-Feynman force from the nonlinear core correction —
 * the interaction between the exchange-correlation potential and the core
 * charge density gradient with respect to atomic positions.
 *
 * This contribution is only nonzero for pseudopotential_set that include NLCC.
 * The algorithm is the same for norm-conserving and ultrasoft pseudopotential_set.
 */

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"

namespace q3 {

class CoreCorrectionForceCalculator {
  public:
    CoreCorrectionForceCalculator(const Crystal& crystal_structure,
                                  const PlanewaveBasis& planewave_basis,
                                  const NormConservingPlanewaveBasis& pseudopotential_basis)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis) {}

    std::vector<Vector3D> compute(const std::vector<Complex>& exchange_correlation_potential_g) {
        std::vector<Vector3D> forces(crystal_structure_.number_of_atoms(), {0.0, 0.0, 0.0});
        /**
         * Gamma-only stores only the positive half sphere, and this
         * summand is even under G -> -G (the imaginary part is odd and
         * G_alpha is odd), so the omitted half contributes as much again.
         * Verified against QE's own force decomposition at Gamma
         * (`verbosity='high'`), where Q3 was exactly half.
         */
        Real half_sphere_scale = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;

        if (!pseudopotential_basis_.has_nonlinear_core_correction()) {
            return forces;
        }

        int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();

        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            int species_index = crystal_structure_.species_type_index(atom_index);
            const auto& pseudopotential =
                pseudopotential_basis_.pseudopotential_set()[species_index];
            if (!pseudopotential.has_nonlinear_core_correction) {
                continue;
            }

            Vector3D atomic_position =
                crystal_structure_.convert_from_crystal_to_cartesian(atom_index);

            for (int g_vector_index = first_nonzero_g_index;
                 g_vector_index < planewave_basis_.number_of_g_vectors();
                 g_vector_index++) {
                const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
                Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_position);

                Real g_vector_magnitude = std::sqrt(g_vector.magnitude_squared) *
                                          crystal_structure_.two_pi_over_lattice_parameter();
                Real core_charge_density_value =
                    pseudopotential_basis_.interpolate_core_charge_density(species_index,
                                                                           g_vector_magnitude);

                for (int direction = 0; direction < 3; direction++) {
                    Real g_direction = g_vector.g_cart[direction] *
                                       crystal_structure_.two_pi_over_lattice_parameter();
                    /**
                     * F_{s,α} = Ω G_α ρ_core(|G|) Im[V_XC(G) e^{+iGτ_s}]  (full sphere, no ×2)
                     * Structure factor must be e^{+iGτ} = compute_phase_factor_conjugate.
                     */
                    Complex contribution = exchange_correlation_potential_g[g_vector_index] *
                                           compute_phase_factor_conjugate(phase);
                    forces[atom_index][direction] += half_sphere_scale * g_direction *
                                                     contribution.imag() *
                                                     core_charge_density_value;
                }
            }
        }

        return forces;
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
};

} // namespace q3
