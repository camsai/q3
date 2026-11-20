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
#include <cstddef>
#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/basis/planewave/structure_factor_calculator.hpp"

namespace q3 {

class CoreCorrectionForceCalculator {
  public:
    /// `structure_factor_calculator` forms the sum over atoms and G-vectors; see compute().
    CoreCorrectionForceCalculator(const Crystal& crystal_structure,
                                  const PlanewaveBasis& planewave_basis,
                                  const NormConservingPlanewaveBasis& pseudopotential_basis,
                                  const StructureFactorCalculator& structure_factor_calculator)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis)
        , structure_factor_calculator_(structure_factor_calculator) {}

    /**
     * F_{s,α} = Ω G_α ρ_core(|G|) Im[V_XC(G) e^{+iGτ_s}]  (full sphere, no ×2), for the
     * atoms of species with a core charge. The weight ρ_core(|G|) V_XC(G) is formed here,
     * once per species and G-vector; the sum against each atom's phases is the structure
     * factor calculator's.
     */
    std::vector<Vector3D> compute(const std::vector<Complex>& exchange_correlation_potential_g) {
        const int number_of_atoms = crystal_structure_.number_of_atoms();
        if (!pseudopotential_basis_.has_nonlinear_core_correction()) {
            return std::vector<Vector3D>(static_cast<std::size_t>(number_of_atoms),
                                         Vector3D{0.0, 0.0, 0.0});
        }
        /**
         * Gamma-only stores only the positive half sphere, and this
         * summand is even under G -> -G (the imaginary part is odd and
         * G_alpha is odd), so the omitted half contributes as much again.
         * Verified against QE's own force decomposition at Gamma
         * (`verbosity='high'`), where Q3 was exactly half.
         */
        const Real half_sphere_scale = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;
        const int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();
        const Real two_pi_over_lattice_parameter =
            crystal_structure_.two_pi_over_lattice_parameter();

        /**
         * Over this rank's density G-vectors, completed by one reduction across the pool.
         * The core charge depends on the species and |G| only, so it is interpolated once per
         * species and G-vector rather than once per atom.
         */
        const LocalDensityGLayout& layout = planewave_basis_.local_density_g_layout();
        std::vector<int> g_vector_indices;
        g_vector_indices.reserve(static_cast<std::size_t>(layout.size()));
        for (int entry = 0; entry < layout.size(); entry++) {
            const int g_vector_index = layout.global_g_indices()[entry];
            if (g_vector_index >= first_nonzero_g_index) {
                g_vector_indices.push_back(g_vector_index);
            }
        }
        const int number_of_species =
            static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
        // One set of weights per species with a core charge; the other species' atoms have none.
        std::vector<int> set_of_species(static_cast<std::size_t>(number_of_species), -1);
        std::vector<std::vector<Complex>> phase_weights;
        for (int species_index = 0; species_index < number_of_species; species_index++) {
            if (!pseudopotential_basis_.pseudopotential_set()[species_index]
                     .has_nonlinear_core_correction) {
                continue;
            }
            set_of_species[static_cast<std::size_t>(species_index)] =
                static_cast<int>(phase_weights.size());
            std::vector<Complex> weights(g_vector_indices.size());
            for (std::size_t entry = 0; entry < g_vector_indices.size(); entry++) {
                const int g_vector_index = g_vector_indices[entry];
                const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
                const Real g_vector_magnitude =
                    std::sqrt(g_vector.magnitude_squared) * two_pi_over_lattice_parameter;
                weights[entry] = half_sphere_scale *
                                 pseudopotential_basis_.interpolate_core_charge_density(
                                     species_index, g_vector_magnitude) *
                                 exchange_correlation_potential_g[g_vector_index];
            }
            phase_weights.push_back(std::move(weights));
        }
        std::vector<int> set_of_atom(static_cast<std::size_t>(number_of_atoms));
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            set_of_atom[static_cast<std::size_t>(atom_index)] =
                set_of_species[static_cast<std::size_t>(
                    crystal_structure_.species_type_index(atom_index))];
        }

        std::vector<Vector3D> forces = structure_factor_calculator_.gradient_phase_sums(
            g_vector_indices, phase_weights, set_of_atom);
        for (auto& force : forces) {
            for (int direction = 0; direction < 3; direction++) {
                force[direction] *= two_pi_over_lattice_parameter;
            }
        }
        planewave_basis_.complete_density_g_sums(forces);

        return forces;
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
    const StructureFactorCalculator& structure_factor_calculator_;
};

} // namespace q3
