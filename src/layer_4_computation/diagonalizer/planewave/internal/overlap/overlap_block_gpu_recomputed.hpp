#pragma once

/**
 * RecomputedGPUOverlapBlock — the overlap block of a general S.
 *
 * S·vectors is formed by the operator, and every update of the vectors is applied
 * to the overlap block as well.
 */

#include <functional>
#include <utility>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/overlap_block_gpu.hpp"

namespace q3 {

/// The overlap block of a general S, formed and updated by computation.
class RecomputedGPUOverlapBlock : public GPUOverlapBlock {
  public:
    /// Writes S·vectors into the overlap block.
    using OverlapApplication =
        std::function<void(const Complex* vectors, Complex* overlap_vectors)>;

    /**
     * @param apply_overlap    The operator's S on the whole block.
     * @param vectors          The block of vectors.
     * @param overlap_vectors  The block holding S applied to it.
     */
    RecomputedGPUOverlapBlock(OverlapApplication apply_overlap,
                              const Complex* vectors,
                              Complex* overlap_vectors)
        : apply_overlap_{std::move(apply_overlap)}
        , vectors_{vectors}
        , overlap_vectors_{overlap_vectors} {}

    /// Form S·vectors with the operator.
    void fill_from_vectors() override {
        apply_overlap_(vectors_, overlap_vectors_);
    }

    /// Apply the update to the overlap block.
    void follow(const BlockUpdate& update) override {
        update(overlap_vectors_);
    }

  private:
    OverlapApplication apply_overlap_;
    const Complex* vectors_;
    Complex* overlap_vectors_;
};

} // namespace q3
