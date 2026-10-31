#pragma once

/**
 * MeanFieldProjectorSource — Kleinman-Bylander projectors, fetched on demand.
 *
 * The projector counterpart of MeanFieldBandSource (see init.hpp): every
 * projector of every atom at one k-point is as large as a band set of the same
 * count, so the record carries a source that builds one k-point at a time
 * instead of holding them all. Only the BerkeleyGW `VKB` writer reads it.
 */

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"

namespace q3 {

/// Hands out one k-point's projectors at a time; see the file comment.
class MeanFieldProjectorSource {
  public:
    virtual ~MeanFieldProjectorSource() = default;

    /**
     * <k+G|β> for physical k-point `kpoint_index`: one row per projector, atoms
     * in the crystal's own order and each atom's channels in the order of
     * `MeanFieldRecord::nonlocal_coefficients`; one column per plane wave, in
     * the order of `MeanFieldRecord::kpoint_g_vector_indices[kpoint_index]`.
     * Valid until the next call for a different k-point.
     */
    [[nodiscard]] virtual const Matrix<Complex>& projectors(int kpoint_index) const = 0;
};

} // namespace q3
