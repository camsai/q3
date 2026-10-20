#pragma once

/**
 * SphericalHarmonicsTable — precomputed real spherical harmonics Y_lm(Ĝ)
 * for all G-vectors in a PlanewaveBasis.
 *
 * This table is a general planewave concept, not specific to ultrasoft
 * pseudopotential_set. Any code that needs Y_lm evaluated at the directions
 * of the G-vectors can use this cached table.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class SphericalHarmonicsTable {
  public:
    SphericalHarmonicsTable() = default;

    /**
     * Initialize the table for all G-vectors up to the given maximum angular momentum.
     *
     * Computes Y_lm(Ĝ) for each G-vector in the basis and each (l,m) pair
     * with l ∈ [0, maximum_angular_momentum] and m ∈ [-l, l].
     */
    void init(const Crystal& crystal_structure,
              const PlanewaveBasis& planewave_basis,
              int maximum_angular_momentum) {
        maximum_angular_momentum_ = maximum_angular_momentum;
        int number_of_lm_components =
            (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);
        int number_of_g_vectors = planewave_basis.number_of_g_vectors();
        Real two_pi_over_lattice_parameter = crystal_structure.two_pi_over_lattice_parameter();

        spherical_harmonics_at_g_.resize(number_of_g_vectors);
        for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
            spherical_harmonics_at_g_[g_index].resize(number_of_lm_components, 0.0);
            const auto& g_vector = planewave_basis.g_vectors()[g_index];
            Vector3D g_cartesian{g_vector.g_cart.x * two_pi_over_lattice_parameter,
                                 g_vector.g_cart.y * two_pi_over_lattice_parameter,
                                 g_vector.g_cart.z * two_pi_over_lattice_parameter};

            for (int angular_momentum = 0; angular_momentum <= maximum_angular_momentum;
                 angular_momentum++) {
                for (int magnetic_projection = -angular_momentum;
                     magnetic_projection <= angular_momentum;
                     magnetic_projection++) {
                    int combined_index =
                        harmonic_combined_degree_order_index(angular_momentum, magnetic_projection);
                    if (combined_index < number_of_lm_components) {
                        spherical_harmonics_at_g_[g_index][combined_index] =
                            compute_real_spherical_harmonic(
                                angular_momentum, magnetic_projection, g_cartesian);
                    }
                }
            }
        }
    }

    /**
     * Access Y_lm(Ĝ) for a given G-vector index and combined (l,m) index.
     */
    [[nodiscard]] Real value(int g_index, int combined_lm_index) const {
        if (combined_lm_index < static_cast<int>(spherical_harmonics_at_g_[g_index].size())) {
            return spherical_harmonics_at_g_[g_index][combined_lm_index];
        }
        return 0.0;
    }

    /**
     * Number of (l,m) components: (lmax+1)².
     */
    [[nodiscard]] int number_of_lm_components() const {
        return (maximum_angular_momentum_ + 1) * (maximum_angular_momentum_ + 1);
    }

    [[nodiscard]] int maximum_angular_momentum() const {
        return maximum_angular_momentum_;
    }

  private:
    int maximum_angular_momentum_{0};
    std::vector<std::vector<Real>> spherical_harmonics_at_g_;
};

} // namespace q3
