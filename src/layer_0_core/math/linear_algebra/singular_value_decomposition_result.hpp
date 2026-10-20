#pragma once

/**
 * SingularValueDecompositionResult — what SingularValueDecomposition returns:
 * the thin factors of A = U S V^dagger, with the singular values descending, U
 * rows x count and V^dagger count x columns.
 */

#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

struct [[nodiscard]] SingularValueDecompositionResult {
    std::vector<Real> singular_values;
    DenseMatrix<Complex> left_vectors;
    DenseMatrix<Complex> right_vectors_adjoint;
};

} // namespace q3
