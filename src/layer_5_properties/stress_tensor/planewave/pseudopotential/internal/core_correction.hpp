#pragma once

/**
 * Core correction (NLCC) stress tensor contribution.
 *
 * Computes the stress from the nonlinear core correction — the interaction
 * between the exchange-correlation potential and the core charge density.
 * This contribution is only nonzero for pseudopotential_set that include NLCC.
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

class CoreCorrectionStressTensorCalculator {
  public:
    CoreCorrectionStressTensorCalculator(const Crystal& crystal_structure,
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
        const auto& exchange_correlation_potential_g = scf_state_.exchange_correlation_potential_g;

        if (exchange_correlation_potential_g.empty())
            return;

        bool has_any_core_correction = false;
        for (int species_index = 0;
             species_index < static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
             species_index++) {
            if (pseudopotential_basis_.pseudopotential_set()[species_index]
                    .has_nonlinear_core_correction) {
                has_any_core_correction = true;
                break;
            }
        }

        if (!has_any_core_correction)
            return;

        Real stress_diagonal_contribution = 0.0;
        Real reciprocal_space_multiplicity = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;
        Real stress_xx = 0, stress_yx = 0, stress_yy = 0, stress_zx = 0, stress_zy = 0,
             stress_zz = 0;

        if (planewave_basis_.first_nonzero_g_index() == 1) {
            for (int species_index = 0;
                 species_index <
                 static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
                 species_index++) {
                if (!pseudopotential_basis_.pseudopotential_set()[species_index]
                         .has_nonlinear_core_correction)
                    continue;
                Real structure_factor_zero = 0.0;
                for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                     atom_index++) {
                    if (crystal_structure_.species_type_index(atom_index) == species_index)
                        structure_factor_zero += 1.0;
                }
                Real core_charge_density_g_zero =
                    pseudopotential_basis_.interpolate_core_charge_density(species_index, 0.0) /
                    crystal_structure_.cell_volume();
                stress_diagonal_contribution += (std::conj(exchange_correlation_potential_g[0]) *
                                                 Complex(structure_factor_zero, 0.0))
                                                    .real() *
                                                core_charge_density_g_zero;
            }
        }

        for (int g_vector_index = planewave_basis_.first_nonzero_g_index();
             g_vector_index < planewave_basis_.number_of_g_vectors();
             g_vector_index++) {
            const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
            Real g_vector_squared = g_vector.magnitude_squared;
            if (g_vector_squared < math::tolerances::numerical_zero_threshold)
                continue;
            Real g_vector_magnitude =
                std::sqrt(g_vector_squared) * crystal_structure_.two_pi_over_lattice_parameter();

            for (int species_index = 0;
                 species_index <
                 static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
                 species_index++) {
                if (!pseudopotential_basis_.pseudopotential_set()[species_index]
                         .has_nonlinear_core_correction)
                    continue;

                /**
                 * S(G) = Σ_I e^{-iG·τ_I}, the same sign the core density itself
                 * is built with (`core_charge_data.hpp`), because this term
                 * pairs against that density: Re[V*_xc(G) ρ_core(|G|) S(G)].
                 *
                 * With the conjugate sign the pairing is not translation
                 * invariant — under τ → τ + t the potential picks up e^{-iG·t}
                 * and a same-signed structure factor picks up another, leaving
                 * a residual e^{+2iG·t}. Shifting diamond silicon's basis from
                 * (0, ¼) to (⅞, ⅛), which is the same crystal, moved the
                 * pressure from 5.63 to 571.67 kbar while the total energy
                 * agreed to 1e-5 Ry. Only datasets with a partial core charge
                 * were affected, which is most PAW datasets and no
                 * norm-conserving one without NLCC.
                 */
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

                Real core_charge_density_g = pseudopotential_basis_.interpolate_core_charge_density(
                                                 species_index, g_vector_magnitude) /
                                             crystal_structure_.cell_volume();
                stress_diagonal_contribution +=
                    reciprocal_space_multiplicity *
                    (std::conj(exchange_correlation_potential_g[g_vector_index]) * structure_factor)
                        .real() *
                    core_charge_density_g;

                /**
                 * Use analytical derivative from pre-tabulated ρ_core(q) instead of
                 * finite-difference with two expensive radial integrals per species.
                 */
                Real derivative_core_charge_density_dq =
                    pseudopotential_basis_.interpolate_core_charge_density_derivative(
                        species_index, g_vector_magnitude) /
                    crystal_structure_.cell_volume();

                Real stress_radial_contribution =
                    reciprocal_space_multiplicity *
                    (std::conj(exchange_correlation_potential_g[g_vector_index]) * structure_factor)
                        .real() *
                    derivative_core_charge_density_dq *
                    crystal_structure_.two_pi_over_lattice_parameter_squared() / g_vector_magnitude;

                stress_xx += stress_radial_contribution * g_vector.g_cart.x * g_vector.g_cart.x;
                stress_yx += stress_radial_contribution * g_vector.g_cart.y * g_vector.g_cart.x;
                stress_yy += stress_radial_contribution * g_vector.g_cart.y * g_vector.g_cart.y;
                stress_zx += stress_radial_contribution * g_vector.g_cart.z * g_vector.g_cart.x;
                stress_zy += stress_radial_contribution * g_vector.g_cart.z * g_vector.g_cart.y;
                stress_zz += stress_radial_contribution * g_vector.g_cart.z * g_vector.g_cart.z;
            }
        }

        result_.core_correction[0][0] = stress_diagonal_contribution + stress_xx;
        result_.core_correction[1][0] = stress_yx;
        result_.core_correction[1][1] = stress_diagonal_contribution + stress_yy;
        result_.core_correction[2][0] = stress_zx;
        result_.core_correction[2][1] = stress_zy;
        result_.core_correction[2][2] = stress_diagonal_contribution + stress_zz;
        result_.core_correction[0][1] = result_.core_correction[1][0];
        result_.core_correction[0][2] = result_.core_correction[2][0];
        result_.core_correction[1][2] = result_.core_correction[2][1];
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    StressTensorContributions& result_;
};

} // namespace q3
