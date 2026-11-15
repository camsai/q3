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

        /**
         * Over this rank's density G-vectors, completed by one reduction across the pool.
         * The core charge depends on the species and |G| only, so it is interpolated once per
         * species and G-vector rather than once per atom; the phase is taken once per atom and
         * G-vector rather than once per direction.
         */
        const int number_of_atoms = crystal_structure_.number_of_atoms();
        const Real two_pi_over_lattice_parameter =
            crystal_structure_.two_pi_over_lattice_parameter();
        const LocalDensityGLayout& layout = planewave_basis_.local_density_g_layout();
        const int number_of_species =
            static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
        std::vector<std::vector<Real>> core_charge_by_species(
            static_cast<std::size_t>(number_of_species));
        for (int species_index = 0; species_index < number_of_species; species_index++) {
            if (!pseudopotential_basis_.pseudopotential_set()[species_index]
                     .has_nonlinear_core_correction) {
                continue;
            }
            auto& values = core_charge_by_species[species_index];
            values.assign(static_cast<std::size_t>(layout.size()), 0.0);
            for (int entry = 0; entry < layout.size(); entry++) {
                const auto& g_vector =
                    planewave_basis_.g_vectors()[layout.global_g_indices()[entry]];
                Real g_vector_magnitude =
                    std::sqrt(g_vector.magnitude_squared) * two_pi_over_lattice_parameter;
                values[entry] = pseudopotential_basis_.interpolate_core_charge_density(
                    species_index, g_vector_magnitude);
            }
        }
        std::vector<Vector3D> atomic_positions(static_cast<std::size_t>(number_of_atoms));
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            atomic_positions[atom_index] =
                crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
        }

        for (int entry = 0; entry < layout.size(); entry++) {
            const int g_vector_index = layout.global_g_indices()[entry];
            if (g_vector_index < first_nonzero_g_index) {
                continue;
            }
            const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                if (core_charge_by_species[species_index].empty()) {
                    continue;
                }
                Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_positions[atom_index]);
                /**
                 * F_{s,α} = Ω G_α ρ_core(|G|) Im[V_XC(G) e^{+iGτ_s}]  (full sphere, no ×2)
                 * Structure factor must be e^{+iGτ} = compute_phase_factor_conjugate.
                 */
                Complex contribution = exchange_correlation_potential_g[g_vector_index] *
                                       compute_phase_factor_conjugate(phase);
                Real core_charge_density_value = core_charge_by_species[species_index][entry];
                for (int direction = 0; direction < 3; direction++) {
                    Real g_direction = g_vector.g_cart[direction] * two_pi_over_lattice_parameter;
                    forces[atom_index][direction] += half_sphere_scale * g_direction *
                                                     contribution.imag() *
                                                     core_charge_density_value;
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
