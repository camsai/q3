#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/math/linear_algebra/lapack.hpp"

using Complex = std::complex<double>;
using namespace q3::lapack;

// ── zheev_ tests ────────────────────────────────────────────────────

TEST(LAPACKWrappers, ZheevDiagonalizes2x2HermitianMatrix) {
    /**
     * Diagonalize the 2x2 Hermitian matrix:
     *   H = [[2,   1+i],
     *        [1-i, 3  ]]
     *
     * tr(H) = 5, det(H) = 2*3 - |1+i|^2 = 6 - 2 = 4
     * Eigenvalues: (5 ± √(25-16))/2 = (5 ± 3)/2 = 1.0 and 4.0
     * LAPACK returns eigenvalues in ascending order.
     *
     * Column-major storage:
     *   column 0: H[0][0], H[1][0] = (2,0), (1,-1)
     *   column 1: H[0][1], H[1][1] = (1,+1), (3,0)
     */
    const int matrix_dimension = 2;
    std::vector<Complex> matrix_data = {
        Complex(2.0, 0.0), Complex(1.0, -1.0), Complex(1.0, 1.0), Complex(3.0, 0.0)};

    std::vector<double> eigenvalues(matrix_dimension);

    // Workspace query
    int workspace_size = -1;
    Complex optimal_workspace_size;
    std::vector<double> real_workspace(3 * matrix_dimension - 2);
    int info = 0;

    zheev_("V",
           "U",
           &matrix_dimension,
           matrix_data.data(),
           &matrix_dimension,
           eigenvalues.data(),
           &optimal_workspace_size,
           &workspace_size,
           real_workspace.data(),
           &info);

    ASSERT_EQ(info, 0);
    workspace_size = static_cast<int>(optimal_workspace_size.real());
    std::vector<Complex> workspace(workspace_size);

    // Reset matrix (workspace query may have modified it)
    matrix_data = {Complex(2.0, 0.0), Complex(1.0, -1.0), Complex(1.0, 1.0), Complex(3.0, 0.0)};

    zheev_("V",
           "U",
           &matrix_dimension,
           matrix_data.data(),
           &matrix_dimension,
           eigenvalues.data(),
           workspace.data(),
           &workspace_size,
           real_workspace.data(),
           &info);

    ASSERT_EQ(info, 0);

    double expected_eigenvalue_small = 1.0;
    double expected_eigenvalue_large = 4.0;

    EXPECT_NEAR(eigenvalues[0], expected_eigenvalue_small, 1e-12);
    EXPECT_NEAR(eigenvalues[1], expected_eigenvalue_large, 1e-12);
}

TEST(LAPACKWrappers, ZheevDiagonalMatrixReturnsExactEigenvalues) {
    /**
     * Diagonalize a diagonal Hermitian matrix:
     *   H = [[5, 0],
     *        [0, 2]]
     * Eigenvalues should be 2 and 5 (ascending).
     */
    const int matrix_dimension = 2;
    std::vector<Complex> matrix_data = {
        Complex(5.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(2.0, 0.0)};

    std::vector<double> eigenvalues(matrix_dimension);
    std::vector<double> real_workspace(3 * matrix_dimension - 2);

    int workspace_size = 4 * matrix_dimension;
    std::vector<Complex> workspace(workspace_size);
    int info = 0;

    zheev_("N",
           "U",
           &matrix_dimension,
           matrix_data.data(),
           &matrix_dimension,
           eigenvalues.data(),
           workspace.data(),
           &workspace_size,
           real_workspace.data(),
           &info);

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(eigenvalues[0], 2.0, 1e-14);
    EXPECT_NEAR(eigenvalues[1], 5.0, 1e-14);
}

// ── dsygv_ tests ────────────────────────────────────────────────────

TEST(LAPACKWrappers, DsygvDiagonalizes2x2RealSymmetricMatrix) {
    /**
     * Diagonalize the 2x2 real symmetric matrix using the generalized
     * eigensolver with B = identity:
     *   A = [[4, 1],
     *        [1, 3]]
     *   B = [[1, 0],
     *        [0, 1]]
     *
     * This reduces to the standard eigenproblem A * v = λ * v.
     * Eigenvalues: (7 ± √5) / 2 ≈ 2.381966 and 4.618034
     * LAPACK returns eigenvalues in ascending order.
     *
     * Column-major storage:
     *   A column 0: A[0][0], A[1][0] = 4, 1
     *   A column 1: A[0][1], A[1][1] = 1, 3
     */
    const int matrix_dimension = 2;
    const int problem_type = 1; // A*x = lambda*B*x

    std::vector<double> matrix_a = {4.0, 1.0, 1.0, 3.0};
    std::vector<double> matrix_b = {1.0, 0.0, 0.0, 1.0};
    std::vector<double> eigenvalues(matrix_dimension);

    // Workspace query
    int workspace_size = -1;
    double optimal_workspace_size;
    int info = 0;

    dsygv_(&problem_type,
           "V",
           "U",
           &matrix_dimension,
           matrix_a.data(),
           &matrix_dimension,
           matrix_b.data(),
           &matrix_dimension,
           eigenvalues.data(),
           &optimal_workspace_size,
           &workspace_size,
           &info);

    ASSERT_EQ(info, 0);
    workspace_size = static_cast<int>(optimal_workspace_size);
    std::vector<double> workspace(workspace_size);

    // Reset matrices
    matrix_a = {4.0, 1.0, 1.0, 3.0};
    matrix_b = {1.0, 0.0, 0.0, 1.0};

    dsygv_(&problem_type,
           "V",
           "U",
           &matrix_dimension,
           matrix_a.data(),
           &matrix_dimension,
           matrix_b.data(),
           &matrix_dimension,
           eigenvalues.data(),
           workspace.data(),
           &workspace_size,
           &info);

    ASSERT_EQ(info, 0);

    double expected_eigenvalue_small = (7.0 - std::sqrt(5.0)) / 2.0;
    double expected_eigenvalue_large = (7.0 + std::sqrt(5.0)) / 2.0;

    EXPECT_NEAR(eigenvalues[0], expected_eigenvalue_small, 1e-12);
    EXPECT_NEAR(eigenvalues[1], expected_eigenvalue_large, 1e-12);
}

TEST(LAPACKWrappers, DsygvDiagonalMatrixReturnsExactEigenvalues) {
    /**
     * Diagonalize a diagonal matrix with B = identity:
     *   A = [[7, 0],
     *        [0, 3]]
     * Eigenvalues should be 3 and 7 (ascending).
     */
    const int matrix_dimension = 2;
    const int problem_type = 1;

    std::vector<double> matrix_a = {7.0, 0.0, 0.0, 3.0};
    std::vector<double> matrix_b = {1.0, 0.0, 0.0, 1.0};
    std::vector<double> eigenvalues(matrix_dimension);

    int workspace_size = 4 * matrix_dimension;
    std::vector<double> workspace(workspace_size);
    int info = 0;

    dsygv_(&problem_type,
           "N",
           "U",
           &matrix_dimension,
           matrix_a.data(),
           &matrix_dimension,
           matrix_b.data(),
           &matrix_dimension,
           eigenvalues.data(),
           workspace.data(),
           &workspace_size,
           &info);

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(eigenvalues[0], 3.0, 1e-14);
    EXPECT_NEAR(eigenvalues[1], 7.0, 1e-14);
}

// ── zgesvd_ tests ───────────────────────────────────────────────────

TEST(LAPACKWrappers, ZgesvdComputesSingularValuesOf2x2Matrix) {
    /**
     * Singular value decomposition of:
     *   A = [[0, 1],
     *        [2, 0]]
     *
     * A^T * A = [[4, 0],
     *            [0, 1]]
     * Eigenvalues of A^T*A are 4 and 1, so singular values are 2 and 1.
     * LAPACK returns singular values in descending order.
     *
     * Column-major storage:
     *   column 0: A[0][0], A[1][0] = 0, 2
     *   column 1: A[0][1], A[1][1] = 1, 0
     */
    const int matrix_dimension = 2;
    std::vector<Complex> matrix_data = {
        Complex(0.0, 0.0), Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)};

    std::vector<double> singular_values(matrix_dimension);
    std::vector<Complex> left_singular_vectors(matrix_dimension * matrix_dimension);
    std::vector<Complex> right_singular_vectors(matrix_dimension * matrix_dimension);
    std::vector<double> real_workspace(5 * matrix_dimension);
    int info = 0;

    // Workspace query
    int workspace_size = -1;
    Complex optimal_workspace_size;

    zgesvd_("A",
            "A",
            &matrix_dimension,
            &matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            singular_values.data(),
            left_singular_vectors.data(),
            &matrix_dimension,
            right_singular_vectors.data(),
            &matrix_dimension,
            &optimal_workspace_size,
            &workspace_size,
            real_workspace.data(),
            &info);

    ASSERT_EQ(info, 0);
    workspace_size = static_cast<int>(optimal_workspace_size.real());
    std::vector<Complex> workspace(workspace_size);

    // Reset matrix (workspace query may have modified it)
    matrix_data = {Complex(0.0, 0.0), Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)};

    zgesvd_("A",
            "A",
            &matrix_dimension,
            &matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            singular_values.data(),
            left_singular_vectors.data(),
            &matrix_dimension,
            right_singular_vectors.data(),
            &matrix_dimension,
            workspace.data(),
            &workspace_size,
            real_workspace.data(),
            &info);

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(singular_values[0], 2.0, 1e-12);
    EXPECT_NEAR(singular_values[1], 1.0, 1e-12);
}

TEST(LAPACKWrappers, ZgesvdSingularMatrixHasZeroSmallestSingularValue) {
    /**
     * A rank-1 matrix:
     *   A = [[1, 2],
     *        [2, 4]]
     * has one nonzero singular value (sqrt(1+4+4+16) = sqrt(25) = 5) and one
     * exactly zero singular value, since row 2 = 2 * row 1.
     */
    const int matrix_dimension = 2;
    std::vector<Complex> matrix_data = {
        Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(2.0, 0.0), Complex(4.0, 0.0)};

    std::vector<double> singular_values(matrix_dimension);
    std::vector<double> real_workspace(5 * matrix_dimension);
    int info = 0;

    int workspace_size = 8 * matrix_dimension;
    std::vector<Complex> workspace(workspace_size);

    zgesvd_("N",
            "N",
            &matrix_dimension,
            &matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            singular_values.data(),
            nullptr,
            &matrix_dimension,
            nullptr,
            &matrix_dimension,
            workspace.data(),
            &workspace_size,
            real_workspace.data(),
            &info);

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(singular_values[0], 5.0, 1e-12);
    EXPECT_NEAR(singular_values[1], 0.0, 1e-12);
}

// ── zgeqp3_ tests ───────────────────────────────────────────────────

TEST(LAPACKWrappers, Zgeqp3PivotsTheLargerNormColumnFirst) {
    /**
     * QR factorization with column pivoting of:
     *   A = [[1, 10],
     *        [0,  0]]
     *
     * Column 0 has norm 1, column 1 has norm 10. Column pivoting selects
     * columns in decreasing norm order, so the pivot must swap them:
     * jpvt = [2, 1] (Fortran 1-based column indices), and the leading
     * diagonal of R must have magnitude 10 (the norm of the pivoted-in
     * column), not 1.
     *
     * Column-major storage:
     *   column 0: A[0][0], A[1][0] = 1, 0
     *   column 1: A[0][1], A[1][1] = 10, 0
     */
    const int matrix_dimension = 2;
    std::vector<Complex> matrix_data = {
        Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(10.0, 0.0), Complex(0.0, 0.0)};

    std::vector<int> column_pivots(matrix_dimension, 0);
    std::vector<Complex> reflector_scalars(matrix_dimension);
    std::vector<double> real_workspace(2 * matrix_dimension);
    int info = 0;

    // Workspace query
    int workspace_size = -1;
    Complex optimal_workspace_size;

    zgeqp3_(&matrix_dimension,
            &matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            column_pivots.data(),
            reflector_scalars.data(),
            &optimal_workspace_size,
            &workspace_size,
            real_workspace.data(),
            &info);

    ASSERT_EQ(info, 0);
    workspace_size = static_cast<int>(optimal_workspace_size.real());
    std::vector<Complex> workspace(workspace_size);

    /**
     * Reset matrix and pivot vector (workspace query may have modified them;
     * an all-zero jpvt lets LAPACK choose the pivot order freely).
     */
    matrix_data = {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(10.0, 0.0), Complex(0.0, 0.0)};
    std::fill(column_pivots.begin(), column_pivots.end(), 0);

    zgeqp3_(&matrix_dimension,
            &matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            column_pivots.data(),
            reflector_scalars.data(),
            workspace.data(),
            &workspace_size,
            real_workspace.data(),
            &info);

    ASSERT_EQ(info, 0);
    EXPECT_EQ(column_pivots[0], 2);
    EXPECT_EQ(column_pivots[1], 1);
    EXPECT_NEAR(std::abs(matrix_data[0]), 10.0, 1e-12);
}

// ── zgetrf_ / zgetri_ tests ─────────────────────────────────────────

TEST(LAPACKWrappers, ZgetrfAndZgetriInvert2x2Matrix) {
    /**
     * LU-factorize then invert:
     *   A = [[4, 3],
     *        [2, 1]]
     *
     * det(A) = 4*1 - 3*2 = -2
     * A^{-1} = (1/det) * [[1, -3], [-2, 4]] = [[-0.5, 1.5], [1, -2]]
     *
     * Column-major storage:
     *   A column 0: A[0][0], A[1][0] = 4, 2
     *   A column 1: A[0][1], A[1][1] = 3, 1
     *   A^{-1} column 0: -0.5, 1
     *   A^{-1} column 1: 1.5, -2
     */
    const int matrix_dimension = 2;
    std::vector<Complex> matrix_data = {
        Complex(4.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0), Complex(1.0, 0.0)};

    std::vector<int> pivot_indices(matrix_dimension);
    int info = 0;

    zgetrf_(&matrix_dimension,
            &matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            pivot_indices.data(),
            &info);
    ASSERT_EQ(info, 0);

    // Workspace query
    int workspace_size = -1;
    Complex optimal_workspace_size;
    zgetri_(&matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            pivot_indices.data(),
            &optimal_workspace_size,
            &workspace_size,
            &info);
    ASSERT_EQ(info, 0);
    workspace_size = static_cast<int>(optimal_workspace_size.real());
    std::vector<Complex> workspace(workspace_size);

    zgetri_(&matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            pivot_indices.data(),
            workspace.data(),
            &workspace_size,
            &info);
    ASSERT_EQ(info, 0);

    EXPECT_NEAR(matrix_data[0].real(), -0.5, 1e-12);
    EXPECT_NEAR(matrix_data[1].real(), 1.0, 1e-12);
    EXPECT_NEAR(matrix_data[2].real(), 1.5, 1e-12);
    EXPECT_NEAR(matrix_data[3].real(), -2.0, 1e-12);
    EXPECT_NEAR(matrix_data[0].imag(), 0.0, 1e-12);
    EXPECT_NEAR(matrix_data[1].imag(), 0.0, 1e-12);
    EXPECT_NEAR(matrix_data[2].imag(), 0.0, 1e-12);
    EXPECT_NEAR(matrix_data[3].imag(), 0.0, 1e-12);
}

TEST(LAPACKWrappers, ZgetrfSingularMatrixReportsNonZeroInfo) {
    /**
     * A singular matrix (row 2 = 2 * row 1) must be reported as such:
     * ZGETRF sets info = k > 0 when U(k,k) is exactly zero.
     */
    const int matrix_dimension = 2;
    std::vector<Complex> matrix_data = {
        Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(2.0, 0.0), Complex(4.0, 0.0)};

    std::vector<int> pivot_indices(matrix_dimension);
    int info = 0;

    zgetrf_(&matrix_dimension,
            &matrix_dimension,
            matrix_data.data(),
            &matrix_dimension,
            pivot_indices.data(),
            &info);

    EXPECT_GT(info, 0);
}
