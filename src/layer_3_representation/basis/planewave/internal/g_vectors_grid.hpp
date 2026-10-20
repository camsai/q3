#pragma once

/**
 * GVectorsGrid — sorted charge-density G-vectors within the charge_density_energy_cutoff sphere.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/reciprocal_space_vectors_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vector.hpp"

namespace q3 {

class GVectorsGrid : public ReciprocalSpaceVectorsGrid {
  public:
    GVectorsGrid() = default;

    void generate(const Crystal& crystal_structure,
                  Real energy_cutoff_charge_density,
                  int grid_dimension_1,
                  int grid_dimension_2,
                  int grid_dimension_3,
                  bool is_gamma_only = false,
                  int rank = 0,
                  int size = 1,
                  int fft_local_start = -1,
                  int fft_local_dimension_1 = -1) {
        grid_dimension_1_ = grid_dimension_1;
        grid_dimension_2_ = grid_dimension_2;
        grid_dimension_3_ = grid_dimension_3;
        is_gamma_only_ = is_gamma_only;

        Real two_pi_over_lattice_parameter_squared =
            crystal_structure.two_pi_over_lattice_parameter_squared();
        const Matrix3x3& reciprocal_lattice = crystal_structure.reciprocal_lattice_vectors();

        Real g_vector_cutoff_squared =
            energy_cutoff_charge_density / two_pi_over_lattice_parameter_squared;

        int maximum_index_1 = grid_dimension_1 / 2;
        int maximum_index_2 = grid_dimension_2 / 2;
        int maximum_index_3 = grid_dimension_3 / 2;

        g_vectors_.clear();
        g_vectors_.reserve(grid_dimension_1 * grid_dimension_2 * grid_dimension_3 / 4);

        for (int index_1 = -maximum_index_1; index_1 <= maximum_index_1; index_1++) {
            for (int index_2 = -maximum_index_2; index_2 <= maximum_index_2; index_2++) {
                for (int index_3 = -maximum_index_3; index_3 <= maximum_index_3; index_3++) {
                    /**
                     * For gamma-only, keep only the positive half-sphere:
                     *   index_3 > 0, OR
                     *   index_3 == 0 && index_2 > 0, OR
                     *   index_3 == 0 && index_2 == 0 && index_1 >= 0
                     * The half itself is dictated: rho(-G) = conj(rho(G)) for a
                     * real density, so one of each +/-G pair is redundant, and
                     * G = 0 -- its own partner -- is kept once, which is why the
                     * last key above is >= 0 and not > 0. Which half is free, and
                     * this is not the one Quantum ESPRESSO keeps: its rule leads
                     * on the FIRST Miller index where this one leads on the third,
                     * so the two codes hold mirror-image halves. Read the rule
                     * above rather than assume the two lists are interchangeable.
                     */
                    if (is_gamma_only) {
                        if (index_3 < 0)
                            continue;
                        if (index_3 == 0 && index_2 < 0)
                            continue;
                        if (index_3 == 0 && index_2 == 0 && index_1 < 0)
                            continue;
                    }

                    Vector3D g_cartesian;
                    for (int dimension = 0; dimension < 3; dimension++) {
                        g_cartesian[dimension] = index_1 * reciprocal_lattice[0][dimension] +
                                                 index_2 * reciprocal_lattice[1][dimension] +
                                                 index_3 * reciprocal_lattice[2][dimension];
                    }
                    Real g_squared = g_cartesian.norm2();
                    if (g_squared <= g_vector_cutoff_squared +
                                         math::tolerances::energy_cutoff_rounding_tolerance) {
                        GVector g_vector;
                        g_vector.miller = {index_1, index_2, index_3};
                        g_vector.g_cart = g_cartesian;
                        g_vector.magnitude_squared = g_squared;
                        g_vectors_.push_back(g_vector);
                    }
                }
            }
        }

        // Sort by magnitude (standard convention, important for kinetic energy).
        std::sort(g_vectors_.begin(),
                  g_vectors_.end(),
                  [](const GVector& first_vector, const GVector& second_vector) {
                      return first_vector.magnitude_squared < second_vector.magnitude_squared;
                  });

        // Store the global G-vector count before partitioning.
        global_number_of_g_vectors_ = static_cast<int>(g_vectors_.size());

        // Partition G-vectors across band ranks.
        if (fft_local_start >= 0 && fft_local_dimension_1 >= 0 && size > 1) {
            /**
             * SLAB-BASED PARTITIONING: Each rank owns G-vectors that fall into its FFT slabs.
             * This ensures all G-vectors are local when scattering to/from the FFT grid.
             * Note: This replaces index-based partitioning when FFT slab info is provided.
             */
            std::vector<GVector> local_g_vectors;
            for (const auto& g_vector : g_vectors_) {
                int h = g_vector.miller.x;
                int i = (h >= 0) ? h : (h + grid_dimension_1_);
                if (i >= fft_local_start && i < fft_local_start + fft_local_dimension_1) {
                    local_g_vectors.push_back(g_vector);
                }
            }
            /**
             * Overwrite global g_vectors_ with local ones.
             * WARNING: This rank now only knows about its local G-vectors.
             * For dot_product and other collectives, this is fine as long as everyone
             * knows it's a partition.
             */
            g_vectors_ = std::move(local_g_vectors);
            local_count_ = static_cast<int>(g_vectors_.size());
            local_start_ = 0; // Relative to the local array, but this is a bit ambiguous.
        } else {
            // INDEX-BASED PARTITIONING (fallback or np=1)
            int total_g = static_cast<int>(g_vectors_.size());
            local_count_ = total_g / size;
            int remainder = total_g % size;
            int global_local_start;
            if (rank < remainder) {
                local_count_++;
                global_local_start = rank * local_count_;
            } else {
                global_local_start = rank * local_count_ + remainder;
            }

            std::vector<GVector> local_g_vectors;
            for (int i = 0; i < local_count_; ++i) {
                local_g_vectors.push_back(g_vectors_[global_local_start + i]);
            }
            g_vectors_ = std::move(local_g_vectors);
            local_start_ = global_local_start;
        }

        first_nonzero_g_index_ = 0;
        if (!g_vectors_.empty() &&
            g_vectors_[0].magnitude_squared < math::tolerances::zero_vector_magnitude_threshold) {
            first_nonzero_g_index_ = 1;
        }

        Real two_pi_over_lattice_parameter = crystal_structure.two_pi_over_lattice_parameter();
        g_vector_magnitudes_.resize(g_vectors_.size());
        for (size_t g_index = 0; g_index < g_vectors_.size(); g_index++) {
            g_vector_magnitudes_[g_index] =
                std::sqrt(g_vectors_[g_index].magnitude_squared) * two_pi_over_lattice_parameter;
        }
    }

    /// Compute the FFT grid index for a G-vector from its Miller indices.
    [[nodiscard]] int g_vector_to_fft_grid_index(const GVector& g_vector) const {
        int index_1 = g_vector.miller[0];
        int index_2 = g_vector.miller[1];
        int index_3 = g_vector.miller[2];
        if (index_1 < 0)
            index_1 += grid_dimension_1_;
        if (index_2 < 0)
            index_2 += grid_dimension_2_;
        if (index_3 < 0)
            index_3 += grid_dimension_3_;
        return (index_1 * grid_dimension_2_ + index_2) * grid_dimension_3_ + index_3;
    }

    [[nodiscard]] int number_of_g_vectors() const {
        return static_cast<int>(g_vectors_.size());
    }

    [[nodiscard]] int grid_dimension_1() const {
        return grid_dimension_1_;
    }

    /**
     * Total number of G-vectors across all band ranks (before partitioning).
     * Equal to number_of_g_vectors() when band_size == 1.
     */
    [[nodiscard]] int global_number_of_g_vectors() const {
        return global_number_of_g_vectors_;
    }

    [[nodiscard]] int local_number_of_g_vectors() const {
        return local_count_;
    }

    [[nodiscard]] int local_g_vector_start_index() const {
        return local_start_;
    }

    [[nodiscard]] const std::vector<GVector>& g_vectors() const {
        return g_vectors_;
    }

    [[nodiscard]] const GVector& g_vector(int index) const {
        return g_vectors_[index];
    }

    [[nodiscard]] int first_nonzero_g_index() const {
        return first_nonzero_g_index_;
    }

    [[nodiscard]] const std::vector<Real>& g_vector_magnitudes() const {
        return g_vector_magnitudes_;
    }

    [[nodiscard]] bool is_gamma_only() const {
        return is_gamma_only_;
    }

    [[nodiscard]] int
    estimate_maximum_plane_waves_per_kpoint(Real energy_cutoff_wavefunction,
                                            Real two_pi_over_lattice_parameter_squared) const {
        Real wavefunction_cutoff_squared =
            energy_cutoff_wavefunction / two_pi_over_lattice_parameter_squared;
        int count = 0;
        for (const auto& g_vector : g_vectors_) {
            if (g_vector.magnitude_squared <=
                wavefunction_cutoff_squared + math::tolerances::energy_cutoff_rounding_tolerance)
                count++;
        }
        return count;
    }

  protected:
    std::vector<GVector> g_vectors_;
    std::vector<Real> g_vector_magnitudes_;
    int first_nonzero_g_index_{};
    int grid_dimension_1_{};
    int grid_dimension_2_{};
    int grid_dimension_3_{};
    int local_start_{0};
    int local_count_{0};
    int global_number_of_g_vectors_{0};
    bool is_gamma_only_{false};
};

} // namespace q3
