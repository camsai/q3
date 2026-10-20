#pragma once

/**
 * PlaneWaveTable — cached per-k-point plane-wave indices and kinetic energies.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/kpoint/plane_wave_selector.hpp"

namespace q3 {

class PlaneWaveTable {
  public:
    void initialize(const GVectorsGrid& g_vectors_grid,
                    const Crystal& crystal_structure,
                    const KPointsGrid& kpoint_grid,
                    Real energy_cutoff_wavefunction,
                    const FFTGrid* fft_wave = nullptr,
                    int band_size = 1) {
        int number_of_kpoints = kpoint_grid.number_of_kpoints();
        kpoint_g_vector_indices_.resize(number_of_kpoints);
        kpoint_kinetic_energy_squared_.resize(number_of_kpoints);
        kpoint_number_of_plane_waves_.resize(number_of_kpoints);

        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
            PlaneWaveSelector::select(g_vectors_grid,
                                      kpoint_grid[kpoint_index].coordinates,
                                      crystal_structure,
                                      energy_cutoff_wavefunction,
                                      kpoint_g_vector_indices_[kpoint_index],
                                      kpoint_kinetic_energy_squared_[kpoint_index],
                                      fft_wave,
                                      band_size);
            kpoint_number_of_plane_waves_[kpoint_index] =
                static_cast<int>(kpoint_g_vector_indices_[kpoint_index].size());
        }
    }

    [[nodiscard]] const std::vector<int>& g_vector_indices_by_kpoint_index(int kpoint_index) const {
        return kpoint_g_vector_indices_[kpoint_index];
    }

    [[nodiscard]] int number_of_plane_waves_by_kpoint_index(int kpoint_index) const {
        return kpoint_number_of_plane_waves_[kpoint_index];
    }

    [[nodiscard]] const std::vector<Real>&
    kinetic_energy_squared_by_kpoint_index(int kpoint_index) const {
        return kpoint_kinetic_energy_squared_[kpoint_index];
    }

    [[nodiscard]] const std::vector<std::vector<int>>& kpoint_g_vector_indices() const {
        return kpoint_g_vector_indices_;
    }

    [[nodiscard]] const std::vector<int>& kpoint_number_of_plane_waves() const {
        return kpoint_number_of_plane_waves_;
    }

    [[nodiscard]] const std::vector<std::vector<Real>>& kpoint_kinetic_energy_squared() const {
        return kpoint_kinetic_energy_squared_;
    }

  private:
    std::vector<std::vector<int>> kpoint_g_vector_indices_;
    std::vector<int> kpoint_number_of_plane_waves_;
    std::vector<std::vector<Real>> kpoint_kinetic_energy_squared_;
};

} // namespace q3
