#pragma once

#include <algorithm>

/**
 * ChargeDensityTransformer — G ↔ real-space transforms for charge density.
 */

#include <vector>

#include "layer_0_core/profiling/stage_timer.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/transform/local_density_g_layout.hpp"

namespace q3 {

class ChargeDensityTransformer {
  public:
    static void rho_g_to_r(const GVectorsGrid& g_vectors_grid,
                           const FFTGrid& fft_rho,
                           const std::vector<Complex>& charge_density_g_space,
                           std::vector<Real>& charge_density_real_space) {
        StageTimer stage_timer("density_transform.to_real_space");
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
        StageTimer stage_timer("density_transform.to_reciprocal_space");
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
        StageTimer reduce_timer("density_transform.reduce_g_space");
        fft_rho.reduce_g_space(charge_density_g_space.data(), number_of_g_vectors);
    }

    /**
     * The same two transforms driven by a precomputed layout: the loops visit only the
     * local G-vectors, with their buffer positions already looked up, instead of every
     * G-vector of the sphere through a virtual index query. The values written and read
     * are the ones the forms above write and read.
     */
    static void rho_g_to_r(const LocalDensityGLayout& layout,
                           const FFTGrid& fft_rho,
                           const std::vector<Complex>& charge_density_g_space,
                           std::vector<Real>& charge_density_real_space) {
        StageTimer stage_timer("density_transform.to_real_space");
        scatter_and_inverse(
            layout,
            fft_rho,
            [&](int entry) {
                return charge_density_g_space[static_cast<std::size_t>(
                    layout.global_g_indices()[entry])];
            },
            charge_density_real_space);
    }

    /// Layout-driven forward transform, completed to the full G vector by a pool reduction.
    static void rho_r_to_g(const LocalDensityGLayout& layout,
                           int number_of_g_vectors,
                           const FFTGrid& fft_rho,
                           const std::vector<Real>& charge_density_real_space,
                           std::vector<Complex>& charge_density_g_space) {
        StageTimer stage_timer("density_transform.to_reciprocal_space");
        std::vector<Complex> work = forward_from_real(fft_rho, charge_density_real_space);
        Real inverse_grid_size = 1.0 / fft_rho.total_grid_points();
        charge_density_g_space.assign(number_of_g_vectors, Complex(0.0, 0.0));
        for (int entry = 0; entry < layout.size(); entry++) {
            charge_density_g_space[static_cast<std::size_t>(layout.global_g_indices()[entry])] =
                work[static_cast<std::size_t>(layout.fft_indices()[entry])] * inverse_grid_size;
        }
        StageTimer reduce_timer("density_transform.reduce_g_space");
        fft_rho.reduce_g_space(charge_density_g_space.data(), number_of_g_vectors);
    }

    /**
     * Reciprocal to real from a compact vector: the local G-vectors only, in layout order.
     * No communication beyond the transform's own, since the inverse reads nothing else.
     */
    static void compact_rho_g_to_r(const LocalDensityGLayout& layout,
                                   const FFTGrid& fft_rho,
                                   const std::vector<Complex>& compact_density_g,
                                   std::vector<Real>& charge_density_real_space) {
        StageTimer stage_timer("density_transform.compact_to_real_space");
        scatter_and_inverse(
            layout,
            fft_rho,
            [&](int entry) { return compact_density_g[static_cast<std::size_t>(entry)]; },
            charge_density_real_space);
    }

    /**
     * Real to reciprocal into a compact vector, without the all-reduce that would make it
     * whole: each rank keeps the G-vectors of its own sticks.
     */
    static void rho_r_to_compact_g(const LocalDensityGLayout& layout,
                                   const FFTGrid& fft_rho,
                                   const std::vector<Real>& charge_density_real_space,
                                   std::vector<Complex>& compact_density_g) {
        StageTimer stage_timer("density_transform.to_compact_reciprocal_space");
        std::vector<Complex> work = forward_from_real(fft_rho, charge_density_real_space);
        Real inverse_grid_size = 1.0 / fft_rho.total_grid_points();
        compact_density_g.resize(static_cast<std::size_t>(layout.size()));
        for (int entry = 0; entry < layout.size(); entry++) {
            compact_density_g[static_cast<std::size_t>(entry)] =
                work[static_cast<std::size_t>(layout.fft_indices()[entry])] * inverse_grid_size;
        }
    }

  private:
    template <typename ValueOfEntry>
    static void scatter_and_inverse(const LocalDensityGLayout& layout,
                                    const FFTGrid& fft_rho,
                                    ValueOfEntry value_of_entry,
                                    std::vector<Real>& charge_density_real_space) {
        int local_points = fft_rho.local_grid_points();
        std::vector<Complex> work(
            static_cast<std::size_t>(std::max(local_points, fft_rho.local_g_space_points())),
            Complex(0.0, 0.0));
        const auto& fft_indices = layout.fft_indices();
        const auto& mirror_fft_indices = layout.mirror_fft_indices();
        for (int entry = 0; entry < layout.size(); entry++) {
            const Complex value = value_of_entry(entry);
            work[static_cast<std::size_t>(fft_indices[entry])] = value;
            if (mirror_fft_indices[entry] >= 0) {
                work[static_cast<std::size_t>(mirror_fft_indices[entry])] = std::conj(value);
            }
        }

        fft_rho.inverse(work.data());

        charge_density_real_space.resize(local_points);
        int total_grid_points = fft_rho.total_grid_points();
        for (int grid_index = 0; grid_index < local_points; grid_index++) {
            charge_density_real_space[grid_index] = work[grid_index].real() * total_grid_points;
        }
    }

    static std::vector<Complex>
    forward_from_real(const FFTGrid& fft_rho, const std::vector<Real>& charge_density_real_space) {
        int local_points = fft_rho.local_grid_points();
        std::vector<Complex> work(
            static_cast<std::size_t>(std::max(local_points, fft_rho.local_g_space_points())),
            Complex(0.0, 0.0));
        for (int grid_index = 0; grid_index < local_points; grid_index++) {
            work[grid_index] = Complex(charge_density_real_space[grid_index], 0.0);
        }
        fft_rho.forward(work.data());
        return work;
    }
};

} // namespace q3
