#pragma once

/**
 * HermitianEigensolver — every eigenvalue and eigenvector of a Hermitian
 * matrix, through zheev, in one place.
 *
 * The workspace query, the second call and the check on `info` are the same
 * wherever a small Hermitian matrix needs diagonalising. The caller's name is
 * the one thing that differs, so it is the constructor argument, and the
 * failure names it.
 *
 * SubspaceDiagonalizer (layer 4) is not this: it solves the generalized
 * problem with an overlap, reports `info` for its callers to interpret, and
 * sits two layers up, where nothing in a lower layer can reach it.
 */

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/hermitian_eigen_decomposition.hpp"
#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

class HermitianEigensolver {
  public:
    /// `context` opens every error this solver raises: the caller, and what it was diagonalising.
    explicit HermitianEigensolver(std::string context) : context_(std::move(context)) {}

    /**
     * The decomposition of `matrix`, whose upper triangle is read. Taken by
     * value because zheev overwrites its input with the eigenvectors, which is
     * where they are then read from.
     */
    [[nodiscard]] HermitianEigenDecomposition solve(DenseMatrix<Complex> matrix) const {
        if (matrix.rows() != matrix.columns()) {
            throw std::invalid_argument(context_ +
                                        ": a Hermitian eigenproblem needs a square "
                                        "matrix, got " +
                                        std::to_string(matrix.rows()) + " x " +
                                        std::to_string(matrix.columns()));
        }
        int dimension = matrix.rows();
        HermitianEigenDecomposition decomposition;
        decomposition.eigenvalues.assign(static_cast<std::size_t>(dimension), 0.0);
        if (dimension == 0) {
            decomposition.eigenvectors = std::move(matrix);
            return decomposition;
        }

        int info = 0;
        int workspace_size = -1;
        Complex workspace_query;
        std::vector<Real> real_workspace(static_cast<std::size_t>(std::max(1, 3 * dimension - 2)));
        lapack::zheev_("V",
                       "U",
                       &dimension,
                       matrix.data(),
                       &dimension,
                       decomposition.eigenvalues.data(),
                       &workspace_query,
                       &workspace_size,
                       real_workspace.data(),
                       &info);
        if (info != 0) {
            throw std::runtime_error(
                context_ + ": zheev workspace query failed with info = " + std::to_string(info));
        }
        workspace_size = std::max(1, static_cast<int>(workspace_query.real()));
        std::vector<Complex> workspace(static_cast<std::size_t>(workspace_size));
        lapack::zheev_("V",
                       "U",
                       &dimension,
                       matrix.data(),
                       &dimension,
                       decomposition.eigenvalues.data(),
                       workspace.data(),
                       &workspace_size,
                       real_workspace.data(),
                       &info);
        if (info != 0) {
            throw std::runtime_error(
                context_ + ": zheev failed with info = " + std::to_string(info) +
                (info > 0 ? "; the iteration did not converge" : "; an argument was invalid"));
        }
        decomposition.eigenvectors = std::move(matrix);
        return decomposition;
    }

  private:
    std::string context_;
};

} // namespace q3
