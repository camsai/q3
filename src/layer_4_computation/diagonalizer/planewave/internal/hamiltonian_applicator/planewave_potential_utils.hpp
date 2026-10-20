#pragma once

/**
 * Shared utility functions for planewave Hamiltonian applicators.
 *
 * These functions are used by both the serial and GPU Hamiltonian applicators.
 * Extracted to avoid code duplication between the two implementations.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

/**
 * Restrict the effective potential from the dense (charge density) FFT grid
 * to the smooth (wavefunction) FFT grid via G-space truncation.
 *
 * 1. Forward FFT V(r) on the dense grid -> V(G)
 * 2. Copy G-components that fit on the smooth grid
 * 3. Inverse FFT on the smooth grid -> V_smooth(r)
 *
 * This is the single-process version (no MPI reduce). For the MPI-aware
 * version, call this function then reduce the result across band groups.
 */
inline std::vector<Complex>
compute_smooth_grid_potential_from_dense(const std::vector<Real>& dense_real_space_potential,
                                         const PlanewaveBasis& planewave_basis) {

    int local_dense_points = planewave_basis.fft_rho().local_grid_points();
    int local_dense_g_points = planewave_basis.fft_rho().local_g_space_points();
    int total_dense_points = planewave_basis.fft_rho().total_grid_points();
    int total_smooth_points = planewave_basis.fft_wave().total_grid_points();

    /**
     * Step 1: forward-transform the dense-grid potential. The buffer holds
     * real space going in and reciprocal space coming out, and under the
     * stick layout those are different sizes, so it is sized for the larger
     * and the normalisation runs over the reciprocal-space count.
     */
    std::vector<Complex> potential_complex(
        static_cast<std::size_t>(std::max(local_dense_points, local_dense_g_points)),
        Complex(0.0, 0.0));
    for (int index = 0; index < local_dense_points; ++index) {
        potential_complex[index] = Complex(dense_real_space_potential[index], 0.0);
    }
    planewave_basis.fft_rho().forward(potential_complex.data());

    Real inverse_dense_points = 1.0 / total_dense_points;
    for (int index = 0; index < local_dense_g_points; ++index) {
        potential_complex[index] *= inverse_dense_points;
    }

    // Step 2: Copy V(G) to the smooth grid
    std::vector<Complex> full_smooth_potential_g(total_smooth_points, Complex(0.0, 0.0));

    int smooth_dimension_1 = planewave_basis.fft_wave().grid_dimension_1();
    int smooth_dimension_2 = planewave_basis.fft_wave().grid_dimension_2();
    int smooth_dimension_3 = planewave_basis.fft_wave().grid_dimension_3();

    /**
     * Visit each wrapped smooth-grid index exactly once. On an even grid,
     * -N/2 and +N/2 alias to the same FFT slot. The former inclusive signed
     * traversal visited both: serial execution happened to overwrite the slot
     * with +N/2, while MPI ranks could own the two dense-grid coefficients
     * separately and sum both during the subsequent reduction.
     *
     * Mapping wrapped indices back to signed Miller indices with +N/2 as the
     * Nyquist representative preserves the former serial result and removes
     * the topology-dependent double contribution.
     */
    for (int wrapped_1 = 0; wrapped_1 < smooth_dimension_1; ++wrapped_1) {
        int miller_1 =
            (wrapped_1 <= smooth_dimension_1 / 2) ? wrapped_1 : wrapped_1 - smooth_dimension_1;
        for (int wrapped_2 = 0; wrapped_2 < smooth_dimension_2; ++wrapped_2) {
            int miller_2 =
                (wrapped_2 <= smooth_dimension_2 / 2) ? wrapped_2 : wrapped_2 - smooth_dimension_2;
            for (int wrapped_3 = 0; wrapped_3 < smooth_dimension_3; ++wrapped_3) {
                int miller_3 = (wrapped_3 <= smooth_dimension_3 / 2)
                                   ? wrapped_3
                                   : wrapped_3 - smooth_dimension_3;
                int dense_index = planewave_basis.fft_rho().g_vector_to_fft_grid_index(
                    miller_1, miller_2, miller_3);

                if (dense_index >= 0) {
                    int global_index =
                        (wrapped_1 * smooth_dimension_2 + wrapped_2) * smooth_dimension_3 +
                        wrapped_3;
                    full_smooth_potential_g[global_index] = potential_complex[dense_index];
                }
            }
        }
    }

    return full_smooth_potential_g;
}

/**
 * Convert full smooth-grid V(G) to real-space V_smooth(r) via inverse FFT.
 *
 * Handles local slab extraction for MPI-distributed grids.
 */
inline std::vector<Real>
inverse_fft_smooth_potential(const std::vector<Complex>& full_smooth_potential_g,
                             const PlanewaveBasis& planewave_basis) {

    int local_smooth_points = planewave_basis.fft_wave().local_grid_points();
    int local_smooth_g_points = planewave_basis.fft_wave().local_g_space_points();
    /**
     * The grid knows which of the full array's points are its own -- a
     * contiguous slab, or a set of sticks -- so it does the gathering. This
     * used to be a memcpy from local_offset(), which encoded the slab layout.
     */
    std::vector<Complex> local_smooth_potential_g(
        static_cast<std::size_t>(std::max(local_smooth_points, local_smooth_g_points)),
        Complex(0.0, 0.0));
    planewave_basis.fft_wave().gather_local_g_space(full_smooth_potential_g.data(),
                                                    local_smooth_potential_g.data());

    planewave_basis.fft_wave().inverse_raw(local_smooth_potential_g.data());

    std::vector<Real> smooth_real_space_potential(local_smooth_points);
    for (int index = 0; index < local_smooth_points; ++index) {
        smooth_real_space_potential[index] = local_smooth_potential_g[index].real();
    }
    return smooth_real_space_potential;
}

/**
 * Pre-compute G-vector to smooth FFT grid index mapping.
 *
 * Maps each plane-wave index to its smooth FFT grid index, or -1 if
 * the G-vector lies outside the smooth grid bounds.
 */
inline void compute_g_to_fft_index_mapping(int number_of_plane_waves,
                                           const std::vector<int>& g_vector_indices,
                                           const PlanewaveBasis& planewave_basis,
                                           std::vector<int>& g_to_fft_mapping) {
    int fft_grid_dimension_1 = planewave_basis.fft_wave().grid_dimension_1();
    int fft_grid_dimension_2 = planewave_basis.fft_wave().grid_dimension_2();
    int fft_grid_dimension_3 = planewave_basis.fft_wave().grid_dimension_3();
    g_to_fft_mapping.resize(number_of_plane_waves);
    for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
        const auto& g_vector = planewave_basis.g_vectors()[g_vector_indices[g_vector_index]];
        int miller_index_1 = static_cast<int>(std::round(g_vector.miller.x));
        int miller_index_2 = static_cast<int>(std::round(g_vector.miller.y));
        int miller_index_3 = static_cast<int>(std::round(g_vector.miller.z));
        if (std::abs(miller_index_1) <= fft_grid_dimension_1 / 2 &&
            std::abs(miller_index_2) <= fft_grid_dimension_2 / 2 &&
            std::abs(miller_index_3) <= fft_grid_dimension_3 / 2) {
            g_to_fft_mapping[g_vector_index] =
                planewave_basis.fft_wave().g_vector_to_fft_grid_index(
                    miller_index_1, miller_index_2, miller_index_3);
        } else {
            g_to_fft_mapping[g_vector_index] = -1;
        }
    }
}

} // namespace q3
