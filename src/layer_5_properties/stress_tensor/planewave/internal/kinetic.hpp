#pragma once

/**
 * Kinetic stress tensor contribution.
 *
 * Computes the kinetic-energy contribution to the stress tensor:
 *   σ_αβ^kin = (1/Ω) Σ_{nk} f_{nk} Σ_G (k+G)_α (k+G)_β |c_{nk}(G)|²
 *
 * This is a purely planewave contribution, independent of pseudopotential type.
 */

#include <cmath>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"

namespace q3 {

class KineticStressTensorCalculator {
  public:
    KineticStressTensorCalculator(const Crystal& crystal_structure,
                                  const PlanewaveBasis& planewave_basis,
                                  const KPointsGrid& kpoint_grid,
                                  const NormConservingPlanewaveKohnShamSCFState& scf_state,
                                  StressTensorContributions& result)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , kpoint_grid_(kpoint_grid)
        , scf_state_(scf_state)
        , result_(result) {}

    void compute() {
        const Matrix3x3& reciprocal_lattice_vectors =
            crystal_structure_.reciprocal_lattice_vectors();
        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        Real reciprocal_space_multiplicity = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;

        /**
         * kpoint_state_index runs over kpoint_grid_.number_of_kpoints() * number_of_spin_components
         * "quantum states" (spin-major: spin_index * number_of_kpoints + kpoint_index — matching
         * PlanewaveKohnShamQuantumSystem's storage convention). Summing only kpoint_index without
         * the spin_index loop silently drops the second spin channel for LSDA (nspin=2), exactly
         * halving this contribution — kpoint_grid_[kpoint_index] itself is NOT spin-doubled (same
         * physical k-point, same already spin-halved weight, for both channels).
         */
        for (int spin_index = 0; spin_index < scf_state_.number_of_spin_components();
             spin_index++) {
            for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
                int kpoint_state_index = spin_index * number_of_kpoints + kpoint_index;
                /**
                 * number_of_plane_waves/g_vector_indices_at_kpoint are PlanewaveBasis-derived
                 * (the G-vector set for a k-point does not depend on spin) — physical
                 * kpoint_index. occupation/wavefunction_coefficient are QuantumSystem-derived
                 * (per spin channel) — spin-doubled kpoint_state_index.
                 */
                int number_of_plane_waves = scf_state_.number_of_plane_waves(kpoint_index);
                const auto& plane_wave_indices_kpoint =
                    scf_state_.g_vector_indices_at_kpoint(kpoint_index);
                Vector3D kpoint_coordinates_crystal = kpoint_grid_[kpoint_index].coordinates;
                Real kpoint_weight = kpoint_grid_[kpoint_index].weight;
                int number_of_bands = scf_state_.number_of_bands();

                Vector3D kpoint_coordinates_cartesian;
                for (int direction = 0; direction < 3; direction++) {
                    kpoint_coordinates_cartesian[direction] =
                        kpoint_coordinates_crystal.x * reciprocal_lattice_vectors[0][direction] +
                        kpoint_coordinates_crystal.y * reciprocal_lattice_vectors[1][direction] +
                        kpoint_coordinates_crystal.z * reciprocal_lattice_vectors[2][direction];
                }

                for (int band_index = 0; band_index < number_of_bands; band_index++) {
                    Real occupation = scf_state_.occupation(kpoint_state_index, band_index);
                    if (occupation < math::tolerances::occupation_threshold)
                        continue;

                    for (int g_vector_index = 0; g_vector_index < number_of_plane_waves;
                         g_vector_index++) {
                        const auto& g_vector =
                            planewave_basis_.g_vectors()[plane_wave_indices_kpoint[g_vector_index]];
                        Vector3D kpoint_plus_g_vector;
                        kpoint_plus_g_vector.x =
                            (kpoint_coordinates_cartesian.x + g_vector.g_cart.x) *
                            crystal_structure_.two_pi_over_lattice_parameter();
                        kpoint_plus_g_vector.y =
                            (kpoint_coordinates_cartesian.y + g_vector.g_cart.y) *
                            crystal_structure_.two_pi_over_lattice_parameter();
                        kpoint_plus_g_vector.z =
                            (kpoint_coordinates_cartesian.z + g_vector.g_cart.z) *
                            crystal_structure_.two_pi_over_lattice_parameter();

                        Real wavefunction_coefficient_squared =
                            std::norm(scf_state_.wavefunction_coefficient(
                                kpoint_state_index, band_index, g_vector_index));

                        for (int direction_a = 0; direction_a < 3; direction_a++) {
                            for (int direction_b = direction_a; direction_b < 3; direction_b++) {
                                result_.kinetic[direction_a][direction_b] +=
                                    reciprocal_space_multiplicity * kpoint_weight * occupation *
                                    constants::electron_charge_squared_rydberg *
                                    kpoint_plus_g_vector[direction_a] *
                                    kpoint_plus_g_vector[direction_b] *
                                    wavefunction_coefficient_squared /
                                    crystal_structure_.cell_volume();
                            }
                        }
                    }
                }
            }
        }
        for (int direction_a = 0; direction_a < 3; direction_a++) {
            for (int direction_b = direction_a + 1; direction_b < 3; direction_b++) {
                result_.kinetic[direction_b][direction_a] =
                    result_.kinetic[direction_a][direction_b];
            }
        }
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const KPointsGrid& kpoint_grid_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    StressTensorContributions& result_;
};

} // namespace q3
