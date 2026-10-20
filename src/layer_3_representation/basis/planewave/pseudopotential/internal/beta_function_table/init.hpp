#pragma once

/**
 * BetaFunctionTable — tabulated beta projector functions on a uniform q-grid.
 *
 * This base class knows exactly one way to obtain β(q): the spherical Bessel
 * transform of a radially tabulated r·β(r), which is what every UPF dataset
 * (norm-conserving, ultrasoft, and UPF-PAW alike) supplies. It knows nothing
 * about any other representation, deliberately: dataset flavors that carry
 * their projectors differently override tabulate_projector() in their own
 * subclass (see beta_function_table/paw.hpp), and the factory in
 * beta_function_table/factory.hpp picks the right table for a pseudopotential
 * set. Shared plane-wave code stays free of flavor-specific includes and
 * casts.
 *
 * The override point is per projector, not per table, because a single
 * calculation may mix dataset flavors across species — a subclass falls back
 * to this base implementation for species it has nothing special to say
 * about.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/math/interpolation/uniform_q_grid_table.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/math/transform/spherical_bessel.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

class BetaFunctionTable {
  public:
    BetaFunctionTable() = default;
    virtual ~BetaFunctionTable() = default;

    /**
     * Initialize the q-grid and tabulate beta functions for all species.
     */
    void initialize(const PlanewaveBasis& planewave_basis,
                    const PseudopotentialSet& pseudopotential_set,
                    int number_of_species_types) {
        static constexpr Real q_range_padding_factor = 1.1;
        Real maximum_q_value =
            std::sqrt(planewave_basis.charge_density_energy_cutoff()) * q_range_padding_factor;
        q_grid_ =
            UniformQGrid::from_maximum_q(maximum_q_value, math::tolerances::default_q_grid_spacing);

        tabulate_all(pseudopotential_set, number_of_species_types);
    }

    /**
     * Initialize using a pre-built shared q-grid.
     */
    void initialize(const UniformQGrid& q_grid,
                    const PseudopotentialSet& pseudopotential_set,
                    int number_of_species_types) {
        q_grid_ = q_grid;
        tabulate_all(pseudopotential_set, number_of_species_types);
    }

    [[nodiscard]] Real interpolate(int species_type_index, int beta_index, Real q_value) const {
        return table_[species_type_index][beta_index].interpolate(q_value);
    }

    [[nodiscard]] Real
    interpolate_derivative(int species_type_index, int beta_index, Real q_value) const {
        return table_[species_type_index][beta_index].interpolate_derivative(q_value);
    }

    [[nodiscard]] const UniformQGrid& q_grid() const {
        return q_grid_;
    }

    [[nodiscard]] int number_of_q_points() const {
        return q_grid_.number_of_q_points;
    }

    [[nodiscard]] Real q_spacing() const {
        return q_grid_.q_spacing;
    }

  protected:
    /**
     * Fill one projector's β(q) at the given q values.
     *
     * The base implementation is the spherical Bessel transform of the
     * dataset's radial r·β(r). Subclasses override this per projector when the
     * dataset supplies β in another representation, and may call this base
     * implementation as the fallback for species they do not handle.
     */
    virtual void tabulate_projector(const Pseudopotential& pseudopotential,
                                    int projector_index,
                                    const std::vector<Real>& q_point_values,
                                    std::vector<Real>& destination) const {
        const auto& projector = pseudopotential.projectors[projector_index];
        transform_spherical_bessel(pseudopotential.radial_grid,
                                   pseudopotential.radial_grid_derivative,
                                   projector.radial_values,
                                   projector.angular_momentum,
                                   projector.number_of_radial_cutoff_points,
                                   q_point_values,
                                   destination);
    }

  private:
    void tabulate_all(const PseudopotentialSet& pseudopotential_set, int number_of_species_types) {
        std::vector<Real> q_point_values = q_grid_.q_point_values();

        table_.resize(number_of_species_types);
        for (int type_index = 0; type_index < number_of_species_types; type_index++) {
            const auto& pseudopotential = pseudopotential_set[type_index];
            table_[type_index].resize(pseudopotential.number_of_projectors);
            for (int projector_index = 0; projector_index < pseudopotential.number_of_projectors;
                 projector_index++) {
                table_[type_index][projector_index].resize(q_grid_);
                tabulate_projector(pseudopotential,
                                   projector_index,
                                   q_point_values,
                                   table_[type_index][projector_index].values());
            }
        }
    }

    UniformQGrid q_grid_;
    std::vector<std::vector<UniformQGridTable>> table_;
};

} // namespace q3
