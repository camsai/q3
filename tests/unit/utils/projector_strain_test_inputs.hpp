#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_strain_inputs.hpp"

namespace q3::test_support {

/**
 * The atom-independent inputs of a k-point's projector strain derivatives, computed the way
 * the nonlocal stress computes them, and owned here so that a test can lend them out.
 */
class ProjectorStrainTestInputs {
  public:
    ProjectorStrainTestInputs(const Crystal& crystal_structure,
                              const PlanewaveBasis& planewave_basis,
                              const Vector3D& kpoint_crystal,
                              const std::vector<int>& plane_wave_indices,
                              int maximum_angular_momentum)
        : plane_wave_indices_(plane_wave_indices) {
        const Matrix3x3& reciprocal = crystal_structure.reciprocal_lattice_vectors();
        Vector3D kpoint_cartesian;
        for (int direction = 0; direction < 3; direction++) {
            kpoint_cartesian[direction] = kpoint_crystal.x * reciprocal[0][direction] +
                                          kpoint_crystal.y * reciprocal[1][direction] +
                                          kpoint_crystal.z * reciprocal[2][direction];
        }
        const int number_of_plane_waves = static_cast<int>(plane_wave_indices.size());
        const Real scale = crystal_structure.two_pi_over_lattice_parameter();
        std::vector<std::array<Real, 3>> direction_reciprocal(number_of_plane_waves);
        std::vector<Real> magnitude_squared_reciprocal(number_of_plane_waves);
        magnitude_bohr_.resize(number_of_plane_waves);
        for (auto& component : components_) {
            component.resize(number_of_plane_waves);
        }
        std::vector<Real> norm_inverse(number_of_plane_waves);
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            const Vector3D k_plus_g =
                kpoint_cartesian + planewave_basis.g_vectors()[plane_wave_indices[g_index]].g_cart;
            direction_reciprocal[g_index] = {k_plus_g.x, k_plus_g.y, k_plus_g.z};
            const Real norm = k_plus_g.norm();
            magnitude_squared_reciprocal[g_index] = norm * norm;
            magnitude_bohr_[g_index] = norm * scale;
            components_[0][g_index] = k_plus_g.x * scale;
            components_[1][g_index] = k_plus_g.y * scale;
            components_[2][g_index] = k_plus_g.z * scale;
            norm_inverse[g_index] =
                magnitude_bohr_[g_index] > 1.0e-12 ? 1.0 / magnitude_bohr_[g_index] : 0.0;
        }
        const int number_of_lm = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);
        compute_real_spherical_harmonics_batch(number_of_lm,
                                               number_of_plane_waves,
                                               direction_reciprocal,
                                               magnitude_squared_reciprocal,
                                               spherical_harmonics_);
        spherical_harmonic_derivatives_.assign(3, {});
        for (int direction = 0; direction < 3; direction++) {
            compute_spherical_harmonic_derivatives_batch(
                number_of_lm,
                number_of_plane_waves,
                direction_reciprocal,
                magnitude_squared_reciprocal,
                direction,
                spherical_harmonic_derivatives_[direction]);
        }
        static constexpr int direction_a[6] = {0, 0, 0, 1, 1, 2};
        static constexpr int direction_b[6] = {0, 1, 2, 1, 2, 2};
        for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
            radial_strain_weights_[strain_pair].resize(number_of_plane_waves);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                radial_strain_weights_[strain_pair][g_index] =
                    components_[direction_a[strain_pair]][g_index] *
                    components_[direction_b[strain_pair]][g_index] * norm_inverse[g_index];
            }
        }
        inputs_.kpoint_cartesian = kpoint_cartesian;
        inputs_.plane_wave_indices = &plane_wave_indices_;
        inputs_.number_of_plane_waves = number_of_plane_waves;
        inputs_.magnitude_bohr = &magnitude_bohr_;
        inputs_.spherical_harmonics = &spherical_harmonics_;
        inputs_.spherical_harmonic_derivatives = &spherical_harmonic_derivatives_;
        inputs_.kpoint_plus_g_components = {&components_[0], &components_[1], &components_[2]};
        inputs_.radial_strain_weights = &radial_strain_weights_;
    }

    ProjectorStrainTestInputs(const ProjectorStrainTestInputs&) = delete;
    ProjectorStrainTestInputs& operator=(const ProjectorStrainTestInputs&) = delete;

    [[nodiscard]] const ProjectorStrainInputs& inputs() const {
        return inputs_;
    }

  private:
    std::vector<int> plane_wave_indices_;
    std::vector<Real> magnitude_bohr_;
    std::array<std::vector<Real>, 3> components_;
    std::vector<std::vector<Real>> spherical_harmonics_;
    std::vector<std::vector<std::vector<Real>>> spherical_harmonic_derivatives_;
    std::array<std::vector<Real>, 6> radial_strain_weights_;
    ProjectorStrainInputs inputs_;
};

} // namespace q3::test_support
