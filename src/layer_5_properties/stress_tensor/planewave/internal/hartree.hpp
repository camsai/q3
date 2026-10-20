#pragma once

/**
 * Hartree stress tensor contribution.
 *
 * Computes the Hartree (electrostatic) contribution to the stress tensor
 * from the charge density in reciprocal space.
 *
 * This is a purely planewave contribution, independent of pseudopotential type.
 */

#include <cmath>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"

namespace q3 {

class HartreeStressTensorCalculator {
  public:
    HartreeStressTensorCalculator(const Crystal& crystal_structure,
                                  const PlanewaveBasis& planewave_basis,
                                  const NormConservingPlanewaveKohnShamSCFState& scf_state,
                                  StressTensorContributions& result)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , scf_state_(scf_state)
        , result_(result) {}

    void compute() {
        Real hartree_energy = scf_state_.energies().hartree_energy;
        const auto& charge_density_g = scf_state_.charge_density_g;
        Real reciprocal_space_multiplicity = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;
        Real stress_xx = 0, stress_yx = 0, stress_yy = 0, stress_zx = 0, stress_zy = 0,
             stress_zz = 0;

        for (int g_vector_index = planewave_basis_.first_nonzero_g_index();
             g_vector_index < planewave_basis_.number_of_g_vectors();
             g_vector_index++) {
            const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
            Real g_vector_squared = g_vector.magnitude_squared;
            if (g_vector_squared < math::tolerances::numerical_zero_threshold)
                continue;
            Real hartree_stress_term = reciprocal_space_multiplicity *
                                       std::norm(charge_density_g[g_vector_index]) /
                                       g_vector_squared;
            stress_xx += hartree_stress_term * 2.0 * g_vector.g_cart.x * g_vector.g_cart.x /
                         g_vector_squared;
            stress_yx += hartree_stress_term * 2.0 * g_vector.g_cart.y * g_vector.g_cart.x /
                         g_vector_squared;
            stress_yy += hartree_stress_term * 2.0 * g_vector.g_cart.y * g_vector.g_cart.y /
                         g_vector_squared;
            stress_zx += hartree_stress_term * 2.0 * g_vector.g_cart.z * g_vector.g_cart.x /
                         g_vector_squared;
            stress_zy += hartree_stress_term * 2.0 * g_vector.g_cart.z * g_vector.g_cart.y /
                         g_vector_squared;
            stress_zz += hartree_stress_term * 2.0 * g_vector.g_cart.z * g_vector.g_cart.z /
                         g_vector_squared;
        }
        Real prefactor = constants::four_pi * constants::electron_charge_squared_rydberg * 0.5 /
                         crystal_structure_.two_pi_over_lattice_parameter_squared();
        Real cell_volume = crystal_structure_.cell_volume();
        result_.hartree[0][0] = prefactor * stress_xx - hartree_energy / cell_volume;
        result_.hartree[1][0] = prefactor * stress_yx;
        result_.hartree[1][1] = prefactor * stress_yy - hartree_energy / cell_volume;
        result_.hartree[2][0] = prefactor * stress_zx;
        result_.hartree[2][1] = prefactor * stress_zy;
        result_.hartree[2][2] = prefactor * stress_zz - hartree_energy / cell_volume;
        for (int direction_a = 0; direction_a < 3; direction_a++) {
            for (int direction_b = 0; direction_b < 3; direction_b++) {
                result_.hartree[direction_a][direction_b] =
                    -result_.hartree[direction_a][direction_b];
            }
        }
        result_.hartree[0][1] = result_.hartree[1][0];
        result_.hartree[0][2] = result_.hartree[2][0];
        result_.hartree[1][2] = result_.hartree[2][1];
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    StressTensorContributions& result_;
};

} // namespace q3
