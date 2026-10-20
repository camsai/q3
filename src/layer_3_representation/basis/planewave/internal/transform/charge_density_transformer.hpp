#pragma once

#include <algorithm>

/**
 * ChargeDensityTransformer — G ↔ real-space transforms for charge density.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"

namespace q3 {

class ChargeDensityTransformer {
  public:
    static void rho_g_to_r(const GVectorsGrid& g_vectors_grid,
                           const FFTGrid& fft_rho,
                           const std::vector<Complex>& charge_density_g_space,
                           std::vector<Real>& charge_density_real_space) {
        int local_points = fft_rho.local_grid_points();
        /**
         * Reciprocal space goes in, real space comes out, and under the stick
         * layout the two counts differ: size for the larger.
         */
        std::vector<Complex> work(
            static_cast<std::size_t>(std::max(local_points, fft_rho.local_g_space_points())),
            Complex(0.0, 0.0));

        const auto& g_vectors = g_vectors_grid.g_vectors();
        bool is_gamma_only = g_vectors_grid.is_gamma_only();

        for (int g_vector_index = 0; g_vector_index < g_vectors_grid.number_of_g_vectors();
             g_vector_index++) {
            const auto& g_vector = g_vectors[g_vector_index];
            int fft_index = fft_rho.g_vector_to_fft_grid_index(
                g_vector.miller[0], g_vector.miller[1], g_vector.miller[2]);
            if (fft_index >= 0) {
                work[fft_index] = charge_density_g_space[g_vector_index];
            }

            // For gamma-only half-sphere: also set the -G position with conjugate.
            if (is_gamma_only && g_vector_index >= g_vectors_grid.first_nonzero_g_index()) {
                int minus_g_fft_index = fft_rho.g_vector_to_fft_grid_index(
                    -g_vector.miller[0], -g_vector.miller[1], -g_vector.miller[2]);
                if (minus_g_fft_index >= 0) {
                    work[minus_g_fft_index] = std::conj(charge_density_g_space[g_vector_index]);
                }
            }
        }

        fft_rho.inverse(work.data());

        charge_density_real_space.resize(local_points);
        int total_grid_points = fft_rho.total_grid_points();
        for (int grid_index = 0; grid_index < local_points; grid_index++) {
            charge_density_real_space[grid_index] = work[grid_index].real() * total_grid_points;
        }
    }

    static void rho_r_to_g(const GVectorsGrid& g_vectors_grid,
                           const FFTGrid& fft_rho,
                           const std::vector<Real>& charge_density_real_space,
                           std::vector<Complex>& charge_density_g_space) {
        int local_points = fft_rho.local_grid_points();
        std::vector<Complex> work(
            static_cast<std::size_t>(std::max(local_points, fft_rho.local_g_space_points())),
            Complex(0.0, 0.0));
        for (int grid_index = 0; grid_index < local_points; grid_index++) {
            work[grid_index] = Complex(charge_density_real_space[grid_index], 0.0);
        }

        fft_rho.forward(work.data());

        Real inverse_grid_size = 1.0 / fft_rho.total_grid_points();
        const auto& g_vectors = g_vectors_grid.g_vectors();
        int number_of_g_vectors = g_vectors_grid.number_of_g_vectors();
        charge_density_g_space.assign(number_of_g_vectors, Complex(0.0, 0.0));
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
            const auto& g_vector = g_vectors[g_vector_index];
            int fft_index = fft_rho.g_vector_to_fft_grid_index(
                g_vector.miller[0], g_vector.miller[1], g_vector.miller[2]);
            if (fft_index >= 0) {
                charge_density_g_space[g_vector_index] = work[fft_index] * inverse_grid_size;
            }
        }

        // Make G-space redundant
        fft_rho.reduce_g_space(charge_density_g_space.data(), number_of_g_vectors);
    }
};

} // namespace q3
