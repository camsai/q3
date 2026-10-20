#pragma once

#include <algorithm>

/**
 * WavefunctionTransformer — G ↔ real-space transforms for wavefunction coefficients.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"

namespace q3 {

class WavefunctionTransformer {
  public:
    static void g_to_r(const GVectorsGrid& g_vectors_grid,
                       const FFTGrid& fft_wave,
                       const std::vector<Complex>& coefficients_g_space,
                       const std::vector<int>& g_vector_indices,
                       int number_of_plane_waves,
                       std::vector<Complex>& coefficients_real_space) {
        int local_points = fft_wave.local_grid_points();
        // Sized for either layout: the scatter fills reciprocal space, the transform leaves real.
        coefficients_real_space.assign(
            static_cast<std::size_t>(std::max(local_points, fft_wave.local_g_space_points())),
            Complex(0.0, 0.0));

        const auto& g_vectors = g_vectors_grid.g_vectors();
        for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
             plane_wave_index++) {
            const auto& g_vector_entry = g_vectors[g_vector_indices[plane_wave_index]];
            int fft_grid_index = fft_wave.g_vector_to_fft_grid_index(
                g_vector_entry.miller.x, g_vector_entry.miller.y, g_vector_entry.miller.z);
            /**
             * With slab-aligned G-vectors, fft_grid_index MUST be >= 0 if the partitioning
             * is correct. We keep the check as a safety measure but it should never trigger.
             */
            if (fft_grid_index >= 0) {
                coefficients_real_space[fft_grid_index] = coefficients_g_space[plane_wave_index];
            }
        }

        fft_wave.inverse(coefficients_real_space.data());
    }

    static void r_to_g(const GVectorsGrid& g_vectors_grid,
                       const FFTGrid& fft_wave,
                       const std::vector<Complex>& coefficients_real_space,
                       const std::vector<int>& g_vector_indices,
                       int number_of_plane_waves,
                       std::vector<Complex>& coefficients_g_space) {
        int local_points = fft_wave.local_grid_points();
        std::vector<Complex> work(coefficients_real_space.begin(), coefficients_real_space.end());
        work.resize(
            static_cast<std::size_t>(std::max(local_points, fft_wave.local_g_space_points())),
            Complex(0.0, 0.0));

        fft_wave.forward(work.data());

        const auto& g_vectors = g_vectors_grid.g_vectors();
        coefficients_g_space.assign(number_of_plane_waves, Complex(0.0, 0.0));
        for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
             plane_wave_index++) {
            const auto& g_vector_entry = g_vectors[g_vector_indices[plane_wave_index]];
            int fft_grid_index = fft_wave.g_vector_to_fft_grid_index(
                g_vector_entry.miller.x, g_vector_entry.miller.y, g_vector_entry.miller.z);
            if (fft_grid_index >= 0) {
                coefficients_g_space[plane_wave_index] = work[fft_grid_index];
            }
        }

        /**
         * REDUNDANCY REMOVED: In band-parallel mode, G-space coefficients are now
         * truly partitioned. Collectives like dot_product or the Hamiltonian application (kinetic)
         * handle the reduction themselves. We no longer need to Allreduce the
         * entire wavefunction after every FFT.
         */
    }
};

} // namespace q3
