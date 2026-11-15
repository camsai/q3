#pragma once

/**
 * CoreChargeData — NLCC core charge density data and interpolation.
 *
 * Uses the pre-computed CoreChargeDensityTable for interpolation, avoiding
 * redundant Bessel transforms during initialization.
 */

#include <vector>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/math/interpolation/uniform_q_grid_table.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/core_charge_density_table.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

class CoreChargeData {
  public:
    CoreChargeData() = default;

    /**
     * Initialize using the pre-built CoreChargeDensityTable to avoid
     * redundant Bessel transforms. Computes ρ_core(G) via interpolation.
     */
    void initialize(const Crystal& crystal_structure,
                    const PlanewaveBasis& planewave_basis,
                    const PseudopotentialSet& pseudopotential_set,
                    const std::vector<Real>& g_vector_magnitudes,
                    Real cell_volume,
                    const CoreChargeDensityTable& core_charge_density_table) {
        int number_of_species_types = static_cast<int>(pseudopotential_set.size());
        int number_of_g_vectors = static_cast<int>(g_vector_magnitudes.size());

        // Check if any species has NLCC
        has_nonlinear_core_correction_ = false;
        for (int type_index = 0; type_index < number_of_species_types; type_index++) {
            if (pseudopotential_set[type_index].has_nonlinear_core_correction) {
                has_nonlinear_core_correction_ = true;
                break;
            }
        }

        if (!has_nonlinear_core_correction_)
            return;

        /**
         * Compute core charge density in reciprocal space using table interpolation
         * Each rank its own density G-vectors, completed by one reduction (see Vloc(G)).
         */
        core_charge_density_reciprocal_.assign(number_of_g_vectors, Complex(0.0, 0.0));
        std::vector<Vector3D> atomic_positions(
            static_cast<std::size_t>(crystal_structure.number_of_atoms()));
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            atomic_positions[atom_index] =
                crystal_structure.convert_from_crystal_to_cartesian(atom_index);
        }
        const LocalDensityGLayout& layout = planewave_basis.local_density_g_layout();
        for (int entry = 0; entry < layout.size(); entry++) {
            const int g_vector_index = layout.global_g_indices()[entry];
            const auto& g_vector = planewave_basis.g_vectors()[g_vector_index];
            for (int type_index = 0; type_index < number_of_species_types; type_index++) {
                if (!pseudopotential_set[type_index].has_nonlinear_core_correction)
                    continue;
                Real q_value = g_vector_magnitudes[g_vector_index];
                Real core_charge_density_value =
                    core_charge_density_table.interpolate(type_index, q_value);

                Complex structure_factor(0.0, 0.0);
                for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms();
                     atom_index++) {
                    if (crystal_structure.species_type_index(atom_index) == type_index) {
                        Real phase =
                            constants::two_pi * g_vector.g_cart.dot(atomic_positions[atom_index]);
                        structure_factor += compute_phase_factor(phase);
                    }
                }
                core_charge_density_reciprocal_[g_vector_index] +=
                    core_charge_density_value * structure_factor / cell_volume;
            }
        }
        planewave_basis.fft_rho().reduce_g_space(core_charge_density_reciprocal_.data(),
                                                 number_of_g_vectors);
    }

    [[nodiscard]] bool has_nonlinear_core_correction() const {
        return has_nonlinear_core_correction_;
    }

    [[nodiscard]] const std::vector<Complex>& core_charge_density_reciprocal() const {
        return core_charge_density_reciprocal_;
    }

  private:
    bool has_nonlinear_core_correction_{false};
    std::vector<Complex> core_charge_density_reciprocal_;
};

} // namespace q3
