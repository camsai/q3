#pragma once

#include <cmath>
#include <cstddef>

#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"

namespace q3::test_support {

/// A quantum system of one k-point state, built without a basis: only its band block is used.
class OneKpointQuantumSystem : public PlanewaveKohnShamQuantumSystem {
  public:
    OneKpointQuantumSystem(const KPointsGrid& kpoint_grid, int number_of_bands_value) {
        kpoint_grid_ = &kpoint_grid;
        number_of_bands_ = number_of_bands_value;
        this->number_of_spin_components = 1;
    }

    /// Give every band coefficient a value of its own, written on the host.
    void fill_bands_with_a_pattern() {
        const int number_of_plane_waves = number_of_plane_waves_for_kpoint_state(0);
        Complex* block = mutable_band_block(0);
        for (int band = 0; band < number_of_bands_; ++band) {
            for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
                block[static_cast<std::size_t>(band) * number_of_plane_waves + plane_wave] =
                    Complex(std::sin(0.37 * (plane_wave + 1) + 1.3 * band),
                            std::cos(0.11 * (plane_wave + 3) * (band + 1)));
            }
        }
    }
};

} // namespace q3::test_support
