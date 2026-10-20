#pragma once

/**
 * Local pseudopotential stress tensor contribution.
 *
 * Computes the stress from the local (ionic) pseudopotential in reciprocal
 * space. This includes both the diagonal (energy) part and the off-diagonal
 * (derivative of V_loc with respect to strain) part.
 *
 * The algorithm is the same for norm-conserving and ultrasoft pseudopotential_set.
 */

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"

namespace q3 {

class LocalPseudopotentialStressTensorCalculator {
  public:
    LocalPseudopotentialStressTensorCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveBasis& pseudopotential_basis,
        const NormConservingPlanewaveKohnShamSCFState& scf_state,
        StressTensorContributions& result)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis)
        , scf_state_(scf_state)
        , result_(result) {}

    void compute() {
        const auto& charge_density_g = scf_state_.charge_density_g;
        Real reciprocal_space_multiplicity = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;
        Real evloc = 0.0;
        {
            int local_gstart = planewave_basis_.first_nonzero_g_index();
            if (local_gstart == 1) {
                for (int species_index = 0;
                     species_index <
                     static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
                     species_index++) {
                    Complex structure_factor_zero(0.0, 0.0);
                    for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                         atom_index++) {
                        if (crystal_structure_.species_type_index(atom_index) != species_index)
                            continue;
                        structure_factor_zero += Complex(1.0, 0.0);
                    }
                    evloc += (std::conj(charge_density_g[0]) *
                              pseudopotential_basis_.local_potential_g(species_index)[0] *
                              structure_factor_zero)
                                 .real();
                }
            }
            for (int g_vector_index = local_gstart;
                 g_vector_index < planewave_basis_.number_of_g_vectors();
                 g_vector_index++) {
                const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
                Complex local_potential_total(0.0, 0.0);
                for (int species_index = 0;
                     species_index <
                     static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
                     species_index++) {
                    Complex structure_factor(0.0, 0.0);
                    for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                         atom_index++) {
                        if (crystal_structure_.species_type_index(atom_index) != species_index)
                            continue;
                        Vector3D atomic_position =
                            crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
                        Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                        structure_factor += compute_phase_factor(phase);
                    }
                    local_potential_total +=
                        pseudopotential_basis_.local_potential_g(species_index)[g_vector_index] *
                        structure_factor;
                }
                evloc +=
                    reciprocal_space_multiplicity *
                    (std::conj(charge_density_g[g_vector_index]) * local_potential_total).real();
            }
        }
        for (int direction_a = 0; direction_a < 3; direction_a++) {
            result_.local[direction_a][direction_a] += evloc;
        }

        Real stress_xx = 0, stress_yx = 0, stress_yy = 0, stress_zx = 0, stress_zy = 0,
             stress_zz = 0;
        for (int g_vector_index = 1; g_vector_index < planewave_basis_.number_of_g_vectors();
             g_vector_index++) {
            const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
            Real g_vector_squared = g_vector.magnitude_squared;
            if (g_vector_squared < math::tolerances::numerical_zero_threshold)
                continue;
            Real g_vector_magnitude =
                std::sqrt(g_vector_squared) * crystal_structure_.two_pi_over_lattice_parameter();

            Complex derivative_local_potential_times_structure_factor(0.0, 0.0);
            for (int species_index = 0;
                 species_index <
                 static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
                 species_index++) {
                /**
                 * Use analytical derivative from pre-tabulated V_loc(q) instead of
                 * finite-difference with two expensive radial integrals per species.
                 */
                Real derivative_local_potential_d_q =
                    pseudopotential_basis_.interpolate_local_potential_derivative(
                        species_index, g_vector_magnitude);
                Real derivative_local_potential_d_g_squared =
                    derivative_local_potential_d_q / (2.0 * g_vector_magnitude);

                Complex structure_factor(0.0, 0.0);
                for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                     atom_index++) {
                    if (crystal_structure_.species_type_index(atom_index) != species_index)
                        continue;
                    Vector3D atomic_position =
                        crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
                    Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                    structure_factor += compute_phase_factor(phase);
                }
                derivative_local_potential_times_structure_factor +=
                    derivative_local_potential_d_g_squared * structure_factor;
            }

            Real stress_partial_contribution = reciprocal_space_multiplicity * 2.0 *
                                               (std::conj(charge_density_g[g_vector_index]) *
                                                derivative_local_potential_times_structure_factor)
                                                   .real();
            stress_xx += stress_partial_contribution * g_vector.g_cart.x * g_vector.g_cart.x;
            stress_yx += stress_partial_contribution * g_vector.g_cart.y * g_vector.g_cart.x;
            stress_yy += stress_partial_contribution * g_vector.g_cart.y * g_vector.g_cart.y;
            stress_zx += stress_partial_contribution * g_vector.g_cart.z * g_vector.g_cart.x;
            stress_zy += stress_partial_contribution * g_vector.g_cart.z * g_vector.g_cart.y;
            stress_zz += stress_partial_contribution * g_vector.g_cart.z * g_vector.g_cart.z;
        }
        result_.local[0][0] +=
            stress_xx * crystal_structure_.two_pi_over_lattice_parameter_squared();
        result_.local[1][0] +=
            stress_yx * crystal_structure_.two_pi_over_lattice_parameter_squared();
        result_.local[1][1] +=
            stress_yy * crystal_structure_.two_pi_over_lattice_parameter_squared();
        result_.local[2][0] +=
            stress_zx * crystal_structure_.two_pi_over_lattice_parameter_squared();
        result_.local[2][1] +=
            stress_zy * crystal_structure_.two_pi_over_lattice_parameter_squared();
        result_.local[2][2] +=
            stress_zz * crystal_structure_.two_pi_over_lattice_parameter_squared();

        result_.local[0][1] = result_.local[1][0];
        result_.local[0][2] = result_.local[2][0];
        result_.local[1][2] = result_.local[2][1];
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    StressTensorContributions& result_;
};

} // namespace q3
