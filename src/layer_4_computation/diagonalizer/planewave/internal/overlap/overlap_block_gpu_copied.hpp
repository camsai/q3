#pragma once

/**
 * CopiedGPUOverlapBlock — the overlap block of S = 1: a copy of the vectors.
 *
 * Every update of the vectors is followed by one device copy. Where the two blocks
 * are the same memory there is nothing to do, and nothing is done.
 */

#include <cstddef>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/overlap_block_gpu.hpp"

namespace q3 {

/// The overlap block of S = 1, kept as a device copy of the vectors.
class CopiedGPUOverlapBlock : public GPUOverlapBlock {
  public:
    /**
     * @param device_backend   Backend whose device holds both blocks; must outlive this object.
     * @param number_of_elements  Complex elements in each block.
     * @param vectors          The block of vectors.
     * @param overlap_vectors  The block kept equal to it.
     */
    CopiedGPUOverlapBlock(execution::gpu::device::GPUDeviceBackend& device_backend,
                          std::size_t number_of_elements,
                          const Complex* vectors,
                          Complex* overlap_vectors)
        : device_backend_{device_backend}
        , number_of_elements_{number_of_elements}
        , vectors_{vectors}
        , overlap_vectors_{overlap_vectors} {}

    /// Copy the vectors into the overlap block.
    void fill_from_vectors() override {
        copy_vectors();
    }

    /// Copy the updated vectors; the update itself is not run a second time.
    void follow(const BlockUpdate& /*update*/) override {
        copy_vectors();
    }

  private:
    void copy_vectors() {
        if (overlap_vectors_ == vectors_) {
            return;
        }
        device_backend_.copy(overlap_vectors_, vectors_, number_of_elements_ * sizeof(Complex));
    }

    execution::gpu::device::GPUDeviceBackend& device_backend_;
    std::size_t number_of_elements_;
    const Complex* vectors_;
    Complex* overlap_vectors_;
};

} // namespace q3
