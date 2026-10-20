#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/type_aliases.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/projection_gauge.hpp"

using namespace q3;

namespace {

/// Largest deviation of U^dagger U from the identity: whether the columns came out orthonormal.
Real orthonormality_deviation(const DenseMatrix<Complex>& gauge) {
    const DenseMatrix<Complex> gram =
        multiply(MatrixOperation::ConjugateTranspose, gauge, MatrixOperation::None, gauge);
    const DenseMatrix<Complex> identity = DenseMatrix<Complex>::identity(gram.rows());
    Real worst = 0.0;
    for (int row = 0; row < gram.rows(); row++) {
        for (int column = 0; column < gram.columns(); column++) {
            worst = std::max(worst, std::abs(gram(row, column) - identity(row, column)));
        }
    }
    return worst;
}

} // namespace

TEST(ProjectionGauge, ProducesOrthonormalColumnsFromANonOrthonormalProjection) {
    // Deliberately non-orthonormal, non-square A: 3 bands, 2 Wannier functions.
    const DenseMatrix<Complex> projections(3,
                                           2,
                                           std::vector<Complex>{
                                               Complex(1.0, 0.0),
                                               Complex(0.5, 0.2),
                                               Complex(0.1, -0.3),
                                               Complex(0.2, 0.4),
                                               Complex(0.9, 0.0),
                                               Complex(0.3, 0.1),
                                           });
    const DenseMatrix<Complex> gauge = ProjectionGauge().compute(projections);
    EXPECT_EQ(gauge.rows(), 3);
    EXPECT_EQ(gauge.columns(), 2);
    EXPECT_LT(orthonormality_deviation(gauge), 1e-12);
}

TEST(ProjectionGauge, LeavesAnAlreadyUnitaryProjectionEssentiallyUnchanged) {
    /// U(A) = A when A is already unitary, since then S = I in A = V S W^dagger.
    const DenseMatrix<Complex> identity = DenseMatrix<Complex>::identity(2);
    const DenseMatrix<Complex> gauge = ProjectionGauge().compute(identity);
    for (int row = 0; row < 2; row++) {
        for (int column = 0; column < 2; column++) {
            EXPECT_NEAR(gauge(row, column).real(), identity(row, column).real(), 1e-12);
            EXPECT_NEAR(gauge(row, column).imag(), identity(row, column).imag(), 1e-12);
        }
    }
}

TEST(ProjectionGauge, RefusesARankDeficientProjection) {
    /**
     * Two identical columns: the trial orbitals do not span a 2-dimensional
     * manifold. Returning a gauge here would produce a silently meaningless
     * Hamiltonian, so it must throw.
     */
    const DenseMatrix<Complex> projections(
        2,
        2,
        std::vector<Complex>{
            Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)});
    try {
        const DenseMatrix<Complex> gauge = ProjectionGauge().compute(projections);
        FAIL() << "two identical columns span one dimension, not two";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::gauge_singular);
    }
}

TEST(ProjectionGauge, RefusesFewerBandsThanWannierFunctions) {
    const DenseMatrix<Complex> projections(1, 2, Complex(1.0, 0.0));
    EXPECT_THROW(ProjectionGauge().compute(projections), std::invalid_argument);
}

/// Over a whole mesh, a refusal says which k-point it happened at; one block alone cannot.
TEST(ProjectionGauge, OverAMeshARefusalNamesItsKPoint) {
    const DenseMatrix<Complex> spanning = DenseMatrix<Complex>::identity(2);
    DenseMatrix<Complex> deficient(2, 2, Complex(0.0, 0.0));
    deficient(0, 0) = Complex(1.0, 0.0);

    const std::vector<DenseMatrix<Complex>> gauges =
        ProjectionGauge().compute_all({spanning, spanning});
    ASSERT_EQ(gauges.size(), 2u);
    EXPECT_LT(orthonormality_deviation(gauges[1]), 1e-12);

    try {
        (void)ProjectionGauge().compute_all({spanning, spanning, deficient});
        FAIL() << "a rank-deficient projection should be refused";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::gauge_singular);
        EXPECT_NE(std::string(error.what()).find("This is k-point 2."), std::string::npos)
            << error.what();
    }
}

/// The tolerance is the gauge's own: a looser one refuses what the default accepts.
TEST(ProjectionGauge, TheSingularValueToleranceIsTheObjectsOwn) {
    DenseMatrix<Complex> nearly_deficient = DenseMatrix<Complex>::identity(2);
    nearly_deficient(1, 1) = Complex(1.0e-4, 0.0);

    EXPECT_NO_THROW((void)ProjectionGauge().compute(nearly_deficient));
    EXPECT_THROW((void)ProjectionGauge(1.0e-3).compute(nearly_deficient), WannierDiagnosticError);
}
