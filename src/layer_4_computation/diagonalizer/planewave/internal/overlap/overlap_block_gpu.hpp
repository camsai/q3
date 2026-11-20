#pragma once

/**
 * GPUOverlapBlock — a block of vectors on a device together with S applied to it,
 * and the one thing that depends on what S is: how the second is kept equal to S
 * times the first.
 *
 * A subspace manager updates a block of vectors by a linear operation on its columns
 * (a projection subtracted, a triangular solve, a rotation) and has to leave the
 * block of S·vectors consistent with it. For a general S the same operation is
 * applied to the S block. For S = 1 held in a buffer of its own the S block is a copy
 * of the vectors, and copying is cheaper than computing the same numbers twice. The
 * overlap operator knows which it is and hands out the matching block
 * (OverlapOperator::create_gpu_overlap_block), so the manager asks no question.
 */

#include <functional>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// A block of vectors on a device paired with S applied to it.
class GPUOverlapBlock {
  public:
    /// A linear update of a block of vectors, applied in place.
    using BlockUpdate = std::function<void(Complex* block)>;

    virtual ~GPUOverlapBlock() = default;

    /// Set the overlap block to S·vectors.
    virtual void fill_from_vectors() = 0;

    /**
     * Bring the overlap block along after `update` has been applied to the vectors.
     * `update` must be linear in the block's columns and must not read the block of
     * vectors it is not given, so that applying it to S·vectors gives S·(updated vectors).
     */
    virtual void follow(const BlockUpdate& update) = 0;
};

} // namespace q3
