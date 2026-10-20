#pragma once

/**
 * SingularValueDecomposition — A = U S V^dagger through zgesvd, the economy
 * form: U has as many columns as there are singular values, and V^dagger as
 * many rows.
 *
 * The economy form is the one a projection gauge needs -- U V^dagger is the
 * unitary nearest A, and the singular values are the direct measure of how
 * well A's columns span its rows -- and the full one costs a rows x rows U
 * that nothing reads.
 */

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_0_core/math/linear_algebra/singular_value_decomposition_result.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

class SingularValueDecomposition {
  public:
    /// `context` opens every error this decomposition raises.
    explicit SingularValueDecomposition(std::string context) : context_(std::move(context)) {}

    /// The decomposition of `matrix`, taken by value because zgesvd destroys its input.
    [[nodiscard]] SingularValueDecompositionResult compute(DenseMatrix<Complex> matrix) const {
        int rows = matrix.rows();
        int columns = matrix.columns();
        if (rows == 0 || columns == 0) {
            throw std::invalid_argument(context_ +
                                        ": a singular value decomposition needs a matrix with "
                                        "rows and columns, got " +
                                        std::to_string(rows) + " x " + std::to_string(columns));
        }
        int count = std::min(rows, columns);

        SingularValueDecompositionResult result;
        result.singular_values.assign(static_cast<std::size_t>(count), 0.0);
        result.left_vectors = DenseMatrix<Complex>(rows, count, Complex(0.0, 0.0));
        result.right_vectors_adjoint = DenseMatrix<Complex>(count, columns, Complex(0.0, 0.0));
        std::vector<Real> real_workspace(5 * static_cast<std::size_t>(count));

        int info = 0;
        int workspace_size = -1;
        Complex workspace_query;
        lapack::zgesvd_("S",
                        "S",
                        &rows,
                        &columns,
                        matrix.data(),
                        &rows,
                        result.singular_values.data(),
                        result.left_vectors.data(),
                        &rows,
                        result.right_vectors_adjoint.data(),
                        &count,
                        &workspace_query,
                        &workspace_size,
                        real_workspace.data(),
                        &info);
        if (info != 0) {
            throw std::runtime_error(
                context_ + ": zgesvd workspace query failed with info = " + std::to_string(info));
        }
        workspace_size = std::max(1, static_cast<int>(workspace_query.real()));
        std::vector<Complex> workspace(static_cast<std::size_t>(workspace_size));
        lapack::zgesvd_("S",
                        "S",
                        &rows,
                        &columns,
                        matrix.data(),
                        &rows,
                        result.singular_values.data(),
                        result.left_vectors.data(),
                        &rows,
                        result.right_vectors_adjoint.data(),
                        &count,
                        workspace.data(),
                        &workspace_size,
                        real_workspace.data(),
                        &info);
        if (info != 0) {
            throw std::runtime_error(context_ +
                                     ": zgesvd failed with info = " + std::to_string(info));
        }
        return result;
    }

  private:
    std::string context_;
};

} // namespace q3
