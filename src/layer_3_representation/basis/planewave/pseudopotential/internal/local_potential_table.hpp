#pragma once

/**
 * LocalPotentialTable — tabulated local pseudopotential on a uniform q-grid.
 *
 * Pre-computes V_loc(q) for each species on the same uniform q-grid used by
 * BetaFunctionTable, enabling O(1) interpolation instead of O(N_radial)
 * Simpson integration per q-point. This eliminates the dominant bottleneck
 * in stress tensor calculations.
 *
 * Only the short-range part is tabulated; the long-range Coulomb tail is added
 * analytically at each q, and q = 0 returns the α_Z constant computed directly. The
 * whole of V_loc(q) used to be tabulated, and a 4-point fit across its 1/q²
 * divergence is off by roughly (dq/q)⁴ of that term near the origin. Bulk cells,
 * whose smallest |G| is near 1/bohr, never noticed; a cell with a long vacuum
 * direction has G-vectors close to the origin, and slabs and molecules picked up a
 * spurious energy that grew roughly as c⁵ with the cell length c, -59 mRy on a
 * Ni/CO slab in a 20 Å cell.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/math/interpolation/uniform_q_grid_table.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/potential_local.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

class LocalPotentialTable {
  public:
    LocalPotentialTable() = default;

    /**
     * Initialize the table by evaluating the short-range part of V_loc(q) on a uniform
     * q-grid for each species.
     */
    void initialize(const UniformQGrid& q_grid,
                    const PseudopotentialSet& pseudopotential_set,
                    const std::vector<int>& mesh_sizes,
                    Real cell_volume,
                    int number_of_species_types) {
        q_grid_ = q_grid;
        cell_volume_ = cell_volume;
        std::vector<Real> q_point_values = q_grid_.q_point_values();

        species_.assign(number_of_species_types, SpeciesTable{});
        for (int type_index = 0; type_index < number_of_species_types; type_index++) {
            const auto& pseudopotential = pseudopotential_set[type_index];
            SpeciesTable& species = species_[type_index];
            species.short_range.resize(q_grid_);
            for (int q_index = 0; q_index < q_grid_.number_of_q_points; q_index++) {
                species.short_range[q_index] = calculate_local_potential_short_range_at_q(
                    pseudopotential, mesh_sizes[type_index], q_point_values[q_index], cell_volume);
            }
            species.zero_q_value = calculate_local_potential_at_q(
                pseudopotential, mesh_sizes[type_index], 0.0, cell_volume);
            species.valence_charge = pseudopotential.valence_charge;
            species.bare_coulomb_tail = local_potential_is_tabulated_reciprocally(pseudopotential);
        }
    }

    /**
     * V_loc(q) for a given species at arbitrary q value: α_Z at q = 0, otherwise the
     * interpolated short-range part plus the exact long-range part.
     */
    [[nodiscard]] Real interpolate(int species_type_index, Real q_value) const {
        const SpeciesTable& species = species_[species_type_index];
        if (q_value < math::tolerances::zero_vector_magnitude_threshold)
            return species.zero_q_value;
        return species.short_range.interpolate(q_value) +
               calculate_local_potential_long_range_at_q(
                   species.valence_charge, species.bare_coulomb_tail, q_value, cell_volume_);
    }

    /**
     * The derivative dV_loc/dq for a given species at arbitrary q value, assembled the
     * same way.
     */
    [[nodiscard]] Real interpolate_derivative(int species_type_index, Real q_value) const {
        const SpeciesTable& species = species_[species_type_index];
        return species.short_range.interpolate_derivative(q_value) +
               calculate_local_potential_long_range_derivative_at_q(
                   species.valence_charge, species.bare_coulomb_tail, q_value, cell_volume_);
    }

    [[nodiscard]] const UniformQGrid& q_grid() const {
        return q_grid_;
    }

  private:
    struct SpeciesTable {
        UniformQGridTable short_range;
        Real zero_q_value = 0.0;
        Real valence_charge = 0.0;
        bool bare_coulomb_tail = false;
    };

    UniformQGrid q_grid_;
    Real cell_volume_ = 0.0;
    std::vector<SpeciesTable> species_;
};

} // namespace q3
