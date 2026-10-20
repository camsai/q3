#pragma once

/**
 * CoreChargeDensityTable — tabulated core charge density on a uniform q-grid.
 *
 * Pre-computes ρ_core(q) for each species on the same uniform q-grid used by
 * BetaFunctionTable, enabling O(1) interpolation instead of O(N_radial)
 * Simpson integration per q-point. This eliminates a significant bottleneck
 * in stress tensor core-correction calculations.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/math/interpolation/uniform_q_grid_table.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/charge_core.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

class CoreChargeDensityTable {
  public:
    CoreChargeDensityTable() = default;

    /**
     * Initialize the table by evaluating ρ_core(q) on a uniform q-grid for each species.
     */
    void initialize(const UniformQGrid& q_grid,
                    const PseudopotentialSet& pseudopotential_set,
                    const std::vector<int>& mesh_sizes,
                    int number_of_species_types) {
        q_grid_ = q_grid;
        std::vector<Real> q_point_values = q_grid_.q_point_values();

        table_.resize(number_of_species_types);
        has_core_charge_.resize(number_of_species_types, false);
        for (int type_index = 0; type_index < number_of_species_types; type_index++) {
            const auto& pseudopotential = pseudopotential_set[type_index];
            if (!pseudopotential.has_nonlinear_core_correction) {
                table_[type_index].resize(q_grid_);
                continue;
            }
            has_core_charge_[type_index] = true;
            table_[type_index].resize(q_grid_);
            for (int q_index = 0; q_index < q_grid_.number_of_q_points; q_index++) {
                table_[type_index][q_index] = calculate_core_charge_density_at_q(
                    pseudopotential, mesh_sizes[type_index], q_point_values[q_index]);
            }
        }
    }

    /**
     * Interpolate ρ_core(q) for a given species at arbitrary q value.
     */
    [[nodiscard]] Real interpolate(int species_type_index, Real q_value) const {
        if (!has_core_charge_[species_type_index]) {
            return 0.0;
        }
        return table_[species_type_index].interpolate(q_value);
    }

    /**
     * Interpolate the derivative dρ_core/dq for a given species at arbitrary q value.
     */
    [[nodiscard]] Real interpolate_derivative(int species_type_index, Real q_value) const {
        if (!has_core_charge_[species_type_index]) {
            return 0.0;
        }
        return table_[species_type_index].interpolate_derivative(q_value);
    }

    [[nodiscard]] const UniformQGrid& q_grid() const {
        return q_grid_;
    }

    [[nodiscard]] bool has_core_charge_for_species(int species_type_index) const {
        return has_core_charge_[species_type_index];
    }

  private:
    UniformQGrid q_grid_;
    std::vector<UniformQGridTable> table_;
    std::vector<bool> has_core_charge_;
};

} // namespace q3
