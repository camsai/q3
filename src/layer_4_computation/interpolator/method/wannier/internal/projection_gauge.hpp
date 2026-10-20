#pragma once

/**
 * ProjectionGauge — the one-shot ("projection-only") gauge
 * U(k) = A(k) [A(k)^dagger A(k)]^(-1/2).
 *
 * This is the starting gauge for maximal localization, and on its own it
 * already produces usable Wannier functions for an isolated manifold: it is
 * the Lowdin-orthonormalization of the trial-orbital projections, i.e. the
 * unitary closest to A in the least-squares sense. Marzari & Vanderbilt, PRB
 * 56, 12847 (1997) §III.A introduces it as the initial guess; for a
 * well-chosen trial set (Si sp3 for the valence manifold, say) the spread it
 * gives is already close to the minimum, which is why interpolated bands from
 * this gauge alone are meaningful.
 *
 * Computed through the singular value decomposition A = V S W^dagger, for
 * which U = V W^dagger exactly. Going through the SVD rather than forming
 * A^dagger A and inverting its square root matters: A^dagger A squares the
 * condition number, so a trial set that merely spans the manifold poorly
 * becomes numerically singular when it need not be. The SVD also hands us the
 * singular values, which are the direct diagnostic of that spanning quality
 * -- a near-zero one means the trial orbitals do not span the manifold, and
 * this class refuses rather than returning a silently rank-deficient gauge.
 *
 * Rectangular A (more bands than Wannier functions) is the normal case once
 * bands are being disentangled; U is then bands x functions with orthonormal
 * columns, not a square unitary.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/math/linear_algebra/singular_value_decomposition.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"

namespace q3 {

class ProjectionGauge {
  public:
    /**
     * A smallest singular value below `singular_value_tolerance` means the
     * trial orbitals fail to span the target manifold there, and every
     * downstream quantity built on the resulting gauge would be quietly
     * meaningless; it is refused.
     */
    explicit ProjectionGauge(Real singular_value_tolerance =
                                 math::tolerances::wannier_projection_singular_value_threshold)
        : singular_value_tolerance_(singular_value_tolerance) {}

    /// U(k) for every k-point's projection block; a refusal names the k-point it happened at.
    [[nodiscard]] std::vector<DenseMatrix<Complex>>
    compute_all(const std::vector<DenseMatrix<Complex>>& projections) const {
        std::vector<DenseMatrix<Complex>> gauges;
        gauges.reserve(projections.size());
        for (std::size_t kpoint_index = 0; kpoint_index < projections.size(); kpoint_index++) {
            try {
                gauges.push_back(compute(projections[kpoint_index]));
            } catch (const WannierDiagnosticError& error) {
                throw WannierDiagnosticError(error.diagnostic(),
                                             std::string(error.what()) + " This is k-point " +
                                                 std::to_string(kpoint_index) + ".");
            }
        }
        return gauges;
    }

    /**
     * U from one k-point's projection block, bands x Wannier functions with
     * the band as the row; U has the same shape.
     */
    [[nodiscard]] DenseMatrix<Complex> compute(const DenseMatrix<Complex>& projections) const {
        if (projections.rows() < projections.columns()) {
            throw std::invalid_argument("ProjectionGauge::compute: number_of_bands (" +
                                        std::to_string(projections.rows()) +
                                        ") must be at least number_of_wannier_functions (" +
                                        std::to_string(projections.columns()) + ")");
        }

        const SingularValueDecompositionResult decomposition =
            SingularValueDecomposition("ProjectionGauge").compute(projections);

        const Real smallest =
            decomposition.singular_values.empty() ? 0.0 : decomposition.singular_values.back();
        if (smallest < singular_value_tolerance_) {
            /**
             * Scientific notation, not std::to_string: the values compared here
             * are near zero by construction, and six fixed decimals would report
             * both of them as "0.000000" -- a diagnostic that names no number.
             */
            std::ostringstream message;
            message << "ProjectionGauge::compute: the trial orbitals do not span the target "
                       "manifold at this k-point -- smallest singular value of A is "
                    << std::scientific << smallest << ", below the tolerance "
                    << singular_value_tolerance_
                    << ". Choose different trial orbitals or a different band selection; a gauge "
                       "built from a rank-deficient A is meaningless.";
            throw WannierDiagnosticError(WannierDiagnostic::gauge_singular, message.str());
        }

        return multiply(MatrixOperation::None,
                        decomposition.left_vectors,
                        MatrixOperation::None,
                        decomposition.right_vectors_adjoint);
    }

  private:
    Real singular_value_tolerance_;
};

} // namespace q3
