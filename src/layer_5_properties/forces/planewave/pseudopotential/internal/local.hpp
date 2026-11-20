#pragma once

/**
 * Local pseudopotential force contribution.
 *
 * Computes the Hellmann-Feynman force from the local (ionic) pseudopotential
 * acting on the electron charge density in reciprocal space.
 *
 * The algorithm is the same for norm-conserving and ultrasoft pseudopotential_set.
 */

#include <complex>
#include <cstddef>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/basis/planewave/structure_factor_calculator.hpp"

namespace q3 {

class LocalPseudopotentialForceCalculator {
  public:
    /// `structure_factor_calculator` forms the sum over atoms and G-vectors; see compute().
    LocalPseudopotentialForceCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveBasis& pseudopotential_basis,
        const StructureFactorCalculator& structure_factor_calculator)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis)
        , structure_factor_calculator_(structure_factor_calculator) {}

    /**
     * F_{α,a} = Ω · G_α · Im[ V_loc(G) · ρ(G) · e^{+iGτ_a} ], summed over G (full sphere).
     *
     * The weight V_loc(G) ρ(G) Ω depends on the species and G only, so it is formed here,
     * once per species and G-vector; the sum against each atom's phases, which is what
     * grows with both the cell and the cutoff, is the structure factor calculator's.
     */
    std::vector<Vector3D> compute(const std::vector<Complex>& charge_density_g) {
        /**
         * Gamma-only stores only the positive half sphere, and this
         * summand is even under G -> -G (the imaginary part is odd and
         * G_alpha is odd), so the omitted half contributes as much again.
         * Verified against QE's own force decomposition at Gamma
         * (`verbosity='high'`), where Q3 was exactly half.
         */
        const Real half_sphere_scale = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;
        const Real cell_volume = crystal_structure_.cell_volume();
        const int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();
        const int number_of_species = crystal_structure_.number_of_species();

        /**
         * Each rank sums over the density G-vectors of its own sticks and one reduction
         * completes the forces: the same terms, split across the pool instead of repeated
         * on every rank of it.
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
        std::vector<std::vector<Complex>> phase_weights(
            static_cast<std::size_t>(number_of_species),
            std::vector<Complex>(g_vector_indices.size()));
        for (int species_index = 0; species_index < number_of_species; species_index++) {
            const auto& local_potential = pseudopotential_basis_.local_potential_g(species_index);
            auto& weights = phase_weights[static_cast<std::size_t>(species_index)];
            for (std::size_t entry = 0; entry < g_vector_indices.size(); entry++) {
                const int g_vector_index = g_vector_indices[entry];
                weights[entry] = half_sphere_scale * cell_volume * local_potential[g_vector_index] *
                                 charge_density_g[g_vector_index];
            }
        }
        std::vector<int> species_of_atom(
            static_cast<std::size_t>(crystal_structure_.number_of_atoms()));
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            species_of_atom[static_cast<std::size_t>(atom_index)] =
                crystal_structure_.species_type_index(atom_index);
        }

        std::vector<Vector3D> forces = structure_factor_calculator_.gradient_phase_sums(
            g_vector_indices, phase_weights, species_of_atom);
        // G_α in atomic units.
        const Real two_pi_over_lattice_parameter =
            crystal_structure_.two_pi_over_lattice_parameter();
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
