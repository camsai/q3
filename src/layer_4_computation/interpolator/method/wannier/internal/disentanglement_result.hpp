#pragma once

/**
 * DisentanglementResult — the subspace SubspaceDisentangler settled on at every
 * k-point, its gauge-invariant spread, and how many iterations it took.
 */

#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

struct [[nodiscard]] DisentanglementResult {
    /// V(k): kept_bands x N, one per k-point.
    std::vector<DenseMatrix<Complex>> subspaces;
    Real gauge_invariant_spread_bohr_squared = 0.0;
    int iterations = 0;
};

} // namespace q3
