#pragma once

/**
 * PlaneWaveSelector — selects and sorts active plane waves at one k-point.
 *
 * When an FFT grid is provided and band_size > 1, only G-vectors that map to
 * the local FFT slab are selected (band/G-vector parallelism). Otherwise, all
 * qualifying G-vectors are selected (serial or k-point-only parallelism).
 *
 * Note: slab filtering may yield zero plane waves for a rank if its FFT slabs
 * do not overlap with the cutoff sphere. The caller (build_kpoint_plane_wave_data)
 * detects this via a collective check and re-initializes with band_size=1.
 */

#include <algorithm>
#include <numeric>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"

namespace q3 {

class PlaneWaveSelector {
  public:
    /**
     * Compute |k+G|² for all G-vectors at a k-point. Returns sorted indices
     * and kinetic energies.
     *
     * When fft_wave is non-null and band_size > 1, only G-vectors whose first
     * Miller index falls into the local FFT slab are selected. This gives each
     * band rank its own subset of plane waves for the wavefunction.
     *
     * Safety: if slab filtering would yield zero plane waves, the filter is
     * disabled and ALL qualifying G-vectors are returned (replicated mode).
     * This prevents zero-dimension BLAS crashes on small systems.
     */
    static void select(const GVectorsGrid& g_vectors_grid,
                       const Vector3D& k_coordinates,
                       const Crystal& crystal_structure,
                       Real energy_cutoff_wavefunction,
                       std::vector<int>& g_vector_indices,
                       std::vector<Real>& kinetic_energy_squared,
                       const FFTGrid* fft_wave = nullptr,
                       int band_size = 1) {
        Real two_pi_over_lattice_parameter_squared =
            crystal_structure.two_pi_over_lattice_parameter_squared();
        const Matrix3x3& reciprocal_lattice = crystal_structure.reciprocal_lattice_vectors();

        Vector3D k_cartesian;
        for (int dimension = 0; dimension < 3; dimension++) {
            k_cartesian[dimension] = k_coordinates.x * reciprocal_lattice[0][dimension] +
                                     k_coordinates.y * reciprocal_lattice[1][dimension] +
                                     k_coordinates.z * reciprocal_lattice[2][dimension];
        }

        Real wavefunction_cutoff_squared =
            energy_cutoff_wavefunction / two_pi_over_lattice_parameter_squared;

        g_vector_indices.clear();
        kinetic_energy_squared.clear();

        const auto& g_vectors = g_vectors_grid.g_vectors();
        int total_g = static_cast<int>(g_vectors.size());

        /**
         * Under a shared pool each rank keeps only the G-vectors on the sticks
         * it owns. The grid decides ownership; this used to encode the slab
         * rule here, which is one reason the slab layout was hard to replace.
         * Ownership is checked against the wavefunction grid, where the
         * transforms run, not the density grid the G-vectors were generated on.
         */
        bool filter_by_ownership = (fft_wave != nullptr && band_size > 1);

        for (int g_index = 0; g_index < total_g; g_index++) {
            if (filter_by_ownership &&
                !fft_wave->owns_stick(static_cast<int>(g_vectors[g_index].miller.x),
                                      static_cast<int>(g_vectors[g_index].miller.y))) {
                continue;
            }

            Vector3D k_plus_g = k_cartesian + g_vectors[g_index].g_cart;
            Real k_plus_g_squared = k_plus_g.norm2();
            if (k_plus_g_squared <=
                wavefunction_cutoff_squared + math::tolerances::energy_cutoff_rounding_tolerance) {
                g_vector_indices.push_back(g_index);
                kinetic_energy_squared.push_back(k_plus_g_squared *
                                                 two_pi_over_lattice_parameter_squared);
            }
        }

        std::vector<int> order(g_vector_indices.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&kinetic_energy_squared](int a, int b) {
            return kinetic_energy_squared[a] < kinetic_energy_squared[b];
        });

        std::vector<int> sorted_g_vector_indices(g_vector_indices.size());
        std::vector<Real> sorted_kinetic_energies(kinetic_energy_squared.size());
        for (size_t index = 0; index < order.size(); index++) {
            sorted_g_vector_indices[index] = g_vector_indices[order[index]];
            sorted_kinetic_energies[index] = kinetic_energy_squared[order[index]];
        }
        g_vector_indices = std::move(sorted_g_vector_indices);
        kinetic_energy_squared = std::move(sorted_kinetic_energies);
    }
};

} // namespace q3
