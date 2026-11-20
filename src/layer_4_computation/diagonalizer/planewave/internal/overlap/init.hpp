#pragma once

/**
 * OverlapOperator — the S operator of the generalized eigenproblem H c = ε S c,
 * expressed purely in the plane-wave basis.
 *
 * When the basis is orthonormal, S = I and the eigenproblem is standard (zheev).
 * When the basis carries an additional overlap, S becomes non-trivial and the
 * eigenproblem is generalized (zhegv). This strategy isolates that difference so
 * the Davidson and direct diagonalizers share a single, branch-free eigensolver
 * path. Concrete operators inject whatever physical data they require through
 * their constructors, keeping this interface free of any model-specific concepts.
 */

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/overlap_block_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/overlap_block_gpu_recomputed.hpp"

#include "layer_0_core/math/linear_algebra/lapack.hpp"

namespace q3 {

class OverlapOperator {
  public:
    virtual ~OverlapOperator() = default;

    // Apply S to a single wavefunction: overlap_wavefunction = S · wavefunction.
    virtual void apply(int number_of_plane_waves,
                       const Complex* wavefunction,
                       Complex* overlap_wavefunction) const = 0;

    /**
     * Apply S to multiple wavefunctions in batch.
     * wavefunctions[vector_index * number_of_plane_waves + g] and
     * overlap_wavefunctions[vector_index * number_of_plane_waves + g]. Default: calls per-vector
     * apply() in a loop. Subclasses may override for efficiency.
     */
    virtual void apply_batch(int number_of_plane_waves,
                             int number_of_vectors,
                             const Complex* wavefunctions,
                             Complex* overlap_wavefunctions) const {
        for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
            apply(number_of_plane_waves,
                  wavefunctions + static_cast<size_t>(vector_index) * number_of_plane_waves,
                  overlap_wavefunctions +
                      static_cast<size_t>(vector_index) * number_of_plane_waves);
        }
    }

    /**
     * Returns true if S = I (norm-conserving case).
     * When true, Davidson can skip S·ψ ZGEMM and S_sub construction.
     */
    [[nodiscard]] virtual bool is_identity() const {
        return false;
    }

    /**
     * Pair a block of vectors on a device with the block that holds S applied to it.
     *
     * The returned object keeps the second consistent with the first in the way this
     * operator's S calls for; see GPUOverlapBlock. Both blocks are column-major,
     * number_of_plane_waves by number_of_vectors, and must outlive it, as must this
     * operator and the backend. The default is the general S: apply_batch() to form the
     * block, and every update applied to it again.
     */
    [[nodiscard]] virtual std::unique_ptr<GPUOverlapBlock>
    create_gpu_overlap_block(execution::gpu::device::GPUDeviceBackend& /*device_backend*/,
                             int number_of_plane_waves,
                             int number_of_vectors,
                             const Complex* vectors,
                             Complex* overlap_vectors) const {
        return std::make_unique<RecomputedGPUOverlapBlock>(
            [this, number_of_plane_waves, number_of_vectors](const Complex* source,
                                                             Complex* destination) {
                apply_batch(number_of_plane_waves, number_of_vectors, source, destination);
            },
            vectors,
            overlap_vectors);
    }

    /**
     * Build the dense overlap matrix S (column-major, number_of_plane_waves ×
     * number_of_plane_waves) used by the direct solver. An orthonormal basis
     * leaves it empty (S = I is implicit in the standard solver).
     */
    virtual void build_dense_matrix(int number_of_plane_waves,
                                    std::vector<Complex>& overlap_matrix) const = 0;
};

} // namespace q3
