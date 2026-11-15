#pragma once

/**
 * LocalDensityGLayout: the density G-vectors this rank's FFT sticks hold, in global order.
 *
 * The density G-vectors are replicated: every rank of a pool has all of them, and a
 * real-to-reciprocal transform fills its own sticks and then completes the rest with an
 * all-reduce of the whole vector. Much of what is done with a density in G-space needs no
 * such completion -- a mixer's elementwise updates and partial inner products, the
 * Hartree and gradient kernels, anything that goes straight back to real space -- because
 * the reciprocal-to-real transform reads only the local sticks anyway. This layout is what
 * those consumers work on instead: one entry per local G-vector, ascending in the global
 * index so that "the first n G-vectors" (the shells below a cutoff, G = 0 first) means
 * the same thing locally as globally.
 *
 * Built once per grid. For a grid that does not distribute (serial, GPU) every G-vector
 * is local and the layout is the identity, so a compact vector is the full one.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"

namespace q3 {

/// The rank-local density G-vectors in ascending global order (see the file comment).
class LocalDensityGLayout {
  public:
    /// List the G-vectors on this rank's sticks with their FFT buffer positions.
    void build(const GVectorsGrid& g_vectors_grid, const FFTGrid& fft_rho) {
        global_g_indices_.clear();
        fft_indices_.clear();
        mirror_fft_indices_.clear();
        const auto& g_vectors = g_vectors_grid.g_vectors();
        const bool is_gamma_only = g_vectors_grid.is_gamma_only();
        for (int g_vector_index = 0; g_vector_index < g_vectors_grid.number_of_g_vectors();
             g_vector_index++) {
            const auto& miller = g_vectors[g_vector_index].miller;
            const int fft_index =
                fft_rho.g_vector_to_fft_grid_index(miller[0], miller[1], miller[2]);
            if (fft_index < 0) {
                continue;
            }
            global_g_indices_.push_back(g_vector_index);
            fft_indices_.push_back(fft_index);
            // The half sphere stores G only; the transform writes conj(rho(G)) at -G too.
            int mirror_fft_index = -1;
            if (is_gamma_only && g_vector_index >= g_vectors_grid.first_nonzero_g_index()) {
                mirror_fft_index =
                    fft_rho.g_vector_to_fft_grid_index(-miller[0], -miller[1], -miller[2]);
            }
            mirror_fft_indices_.push_back(mirror_fft_index);
        }
        number_of_g_vectors_ = g_vectors_grid.number_of_g_vectors();
    }

    /// Number of local G-vectors, the length of a compact vector.
    [[nodiscard]] int size() const {
        return static_cast<int>(global_g_indices_.size());
    }

    /// Global G-vector index of each compact entry, ascending.
    [[nodiscard]] const std::vector<int>& global_g_indices() const {
        return global_g_indices_;
    }

    /// Position of each compact entry in the FFT grid's reciprocal-space buffer.
    [[nodiscard]] const std::vector<int>& fft_indices() const {
        return fft_indices_;
    }

    /// Position of -G for each entry on a gamma-only half sphere, else -1.
    [[nodiscard]] const std::vector<int>& mirror_fft_indices() const {
        return mirror_fft_indices_;
    }

    /// Whether every G-vector is local, so compact and full vectors coincide.
    [[nodiscard]] bool holds_every_g_vector() const {
        return size() == number_of_g_vectors_;
    }

    /// How many compact entries have a global index below `global_count`.
    [[nodiscard]] int count_below(int global_count) const {
        int count = 0;
        while (count < size() && global_g_indices_[count] < global_count) {
            count++;
        }
        return count;
    }

    /// The compact entries of a full, replicated G-space vector.
    template <typename Value>
    [[nodiscard]] std::vector<Value> compact(const std::vector<Value>& full) const {
        std::vector<Value> local(global_g_indices_.size());
        for (std::size_t entry = 0; entry < global_g_indices_.size(); entry++) {
            local[entry] = full[static_cast<std::size_t>(global_g_indices_[entry])];
        }
        return local;
    }

  private:
    std::vector<int> global_g_indices_;
    std::vector<int> fft_indices_;
    std::vector<int> mirror_fft_indices_;
    int number_of_g_vectors_{0};
};

} // namespace q3
