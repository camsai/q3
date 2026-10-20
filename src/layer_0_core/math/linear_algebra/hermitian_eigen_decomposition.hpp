#pragma once

/**
 * HermitianEigenDecomposition — what HermitianEigensolver returns: the
 * eigenvalues, ascending, and the eigenvector of each in the matching column.
 */

#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

struct [[nodiscard]] HermitianEigenDecomposition {
    std::vector<Real> eigenvalues;
    DenseMatrix<Complex> eigenvectors;
};

} // namespace q3
