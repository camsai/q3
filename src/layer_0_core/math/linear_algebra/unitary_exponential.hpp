#pragma once

/**
 * exp(A) for an anti-Hermitian A, which is unitary: the rotation a gauge
 * takes when it moves along a descent direction.
 *
 * Through the Hermitian H = -iA: exp(A) = exp(iH) = V diag(exp(i lambda))
 * V^dagger, with V and lambda from the eigensolver. Exact for any step
 * length, where a truncated series would be unitary only for a small one --
 * and a line search takes steps of every size.
 */

#include <complex>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/hermitian_eigensolver.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

[[nodiscard]] inline DenseMatrix<Complex>
exponential_of_anti_hermitian(const DenseMatrix<Complex>& generator) {
    const HermitianEigenDecomposition decomposition =
        HermitianEigensolver("exponential_of_anti_hermitian")
            .solve(generator.scaled(Complex(0.0, -1.0)));

    std::vector<Complex> phases;
    phases.reserve(decomposition.eigenvalues.size());
    for (const Real eigenvalue : decomposition.eigenvalues) {
        phases.push_back(std::exp(Complex(0.0, eigenvalue)));
    }
    return multiply(MatrixOperation::None,
                    decomposition.eigenvectors.scaled_columns(phases),
                    MatrixOperation::ConjugateTranspose,
                    decomposition.eigenvectors);
}

} // namespace q3
