#include <cmath>
#include <gtest/gtest.h>

#include <vector>

#include "layer_4_computation/effective_potential/internal/hubbard/seeding.hpp"

using namespace q3;

namespace {

/// ‖A − A^H‖_max, on a size x size matrix in either consistent layout.
double hermiticity_defect(const std::vector<Complex>& matrix, int size) {
    double defect = 0.0;
    for (int row = 0; row < size; row++)
        for (int column = 0; column < size; column++) {
            Complex difference = matrix[static_cast<size_t>(row) * size + column] -
                                 std::conj(matrix[static_cast<size_t>(column) * size + row]);
            defect = std::max(defect, std::abs(difference));
        }
    return defect;
}

std::vector<double> eigenvalues_of(std::vector<Complex> matrix, int size) {
    /**
     * Reuse the function under test's own decomposition path indirectly: seed
     * a no-op (replace an eigenvalue with itself is not available without
     * knowing the spectrum), so diagonalize here with the same LAPACK call.
     */
    std::vector<double> eigenvalues(static_cast<size_t>(size));
    int work_size = std::max(1, 2 * size - 1);
    std::vector<Complex> work(static_cast<size_t>(work_size));
    std::vector<double> rwork(static_cast<size_t>(std::max(1, 3 * size - 2)));
    int info = 0;
    char jobz = 'N';
    char uplo = 'U';
    int dimension = size;
    lapack::zheev_(&jobz,
                   &uplo,
                   &dimension,
                   reinterpret_cast<std::complex<double>*>(matrix.data()),
                   &dimension,
                   eigenvalues.data(),
                   reinterpret_cast<std::complex<double>*>(work.data()),
                   &work_size,
                   rwork.data(),
                   &info);
    EXPECT_EQ(info, 0);
    return eigenvalues;
}

} // namespace

TEST(HubbardSeeding, ReplacesExactlyTheNamedEigenvalueOfADiagonalMatrix) {
    // Diagonal matrix: spectrum is the diagonal, ascending order is {0.1, 0.4, 0.9}.
    int size = 3;
    std::vector<Complex> matrix(9, Complex(0.0, 0.0));
    matrix[0] = Complex(0.9, 0.0);
    matrix[4] = Complex(0.1, 0.0);
    matrix[8] = Complex(0.4, 0.0);

    apply_hubbard_starting_eigenvalue(matrix, size, /*eigenvalue_index=*/0, /*value=*/0.25);

    auto spectrum = eigenvalues_of(matrix, size);
    EXPECT_NEAR(spectrum[0], 0.25, 1e-12); // the smallest, replaced
    EXPECT_NEAR(spectrum[1], 0.4, 1e-12);  // untouched
    EXPECT_NEAR(spectrum[2], 0.9, 1e-12);  // untouched
    EXPECT_LT(hermiticity_defect(matrix, size), 1e-12);
}

TEST(HubbardSeeding, PreservesEigenvectorsOfANonDiagonalMatrix) {
    /**
     * 2x2 with known decomposition: eigenvalues 0 and 1, eigenvectors
     * (1, ∓1)/sqrt(2). Replacing the lower eigenvalue with 1 must produce
     * the identity -- same eigenvectors, edited spectrum.
     */
    int size = 2;
    std::vector<Complex> matrix = {
        Complex(0.5, 0.0), Complex(0.5, 0.0), Complex(0.5, 0.0), Complex(0.5, 0.0)};

    apply_hubbard_starting_eigenvalue(matrix, size, /*eigenvalue_index=*/0, /*value=*/1.0);

    EXPECT_NEAR(matrix[0].real(), 1.0, 1e-12);
    EXPECT_NEAR(matrix[1].real(), 0.0, 1e-12);
    EXPECT_NEAR(matrix[2].real(), 0.0, 1e-12);
    EXPECT_NEAR(matrix[3].real(), 1.0, 1e-12);
}

TEST(HubbardSeeding, SeedingTheSpectrumItAlreadyHasIsANoOp) {
    // The invariance that makes seeding safe: aiming at the state you are in changes nothing.
    int size = 2;
    std::vector<Complex> matrix = {
        Complex(0.7, 0.0), Complex(0.1, 0.0), Complex(0.1, 0.0), Complex(0.2, 0.0)};
    auto original = matrix;
    auto spectrum = eigenvalues_of(matrix, size);

    apply_hubbard_starting_eigenvalue(matrix, size, 0, spectrum[0]);
    apply_hubbard_starting_eigenvalue(matrix, size, 1, spectrum[1]);

    for (size_t index = 0; index < original.size(); index++)
        EXPECT_NEAR(std::abs(matrix[index] - original[index]), 0.0, 1e-12);
}

TEST(HubbardSeeding, OutOfRangeIndexThrows) {
    std::vector<Complex> matrix(25, Complex(0.0, 0.0));
    EXPECT_THROW(apply_hubbard_starting_eigenvalue(matrix, 5, 5, 0.5), std::runtime_error);
    EXPECT_THROW(apply_hubbard_starting_eigenvalue(matrix, 5, -1, 0.5), std::runtime_error);
}
