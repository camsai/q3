#include "layer_0_core/execution/serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/factory.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

namespace q3 {

TEST(SubspaceDiagonalizerTest, SerialIdentity) {
    execution::SerialExecutionPolicy policy;
    auto diagonalizer = SubspaceDiagonalizerFactory::create(policy);
    ASSERT_NE(diagonalizer, nullptr);

    int basis_size = 2;
    std::vector<Complex> hamiltonian = {
        Complex(2.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(3.0, 0.0)};
    std::vector<Complex> overlap = {
        Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};
    std::vector<Real> eigenvalues(basis_size);
    std::vector<Complex> eigenvectors(basis_size * basis_size);

    int info = diagonalizer->solve(
        basis_size, hamiltonian.data(), overlap.data(), eigenvalues.data(), eigenvectors.data());

    EXPECT_EQ(info, 0);
    EXPECT_NEAR(eigenvalues[0], 2.0, 1e-10);
    EXPECT_NEAR(eigenvalues[1], 3.0, 1e-10);
}

TEST(SubspaceDiagonalizerTest, SerialGeneralized) {
    execution::SerialExecutionPolicy policy;
    auto diagonalizer = SubspaceDiagonalizerFactory::create(policy);

    int basis_size = 2;
    /**
     * H = [2 0; 0 4], S = [1 0; 0 2]
     * Hx = lambda Sx => [2 0; 0 4][x1; x2] = lambda [1 0; 0 2][x1; x2]
     * 2x1 = lambda x1 => lambda1 = 2
     * 4x2 = 2 lambda x2 => lambda2 = 2
     */
    std::vector<Complex> hamiltonian = {
        Complex(2.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(4.0, 0.0)};
    std::vector<Complex> overlap = {
        Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(2.0, 0.0)};
    std::vector<Real> eigenvalues(basis_size);
    std::vector<Complex> eigenvectors(basis_size * basis_size);

    int info = diagonalizer->solve(
        basis_size, hamiltonian.data(), overlap.data(), eigenvalues.data(), eigenvectors.data());

    EXPECT_EQ(info, 0);
    EXPECT_NEAR(eigenvalues[0], 2.0, 1e-10);
    EXPECT_NEAR(eigenvalues[1], 2.0, 1e-10);
}

TEST(SubspaceDiagonalizerTest, SerialGammaOnly) {
    execution::SerialExecutionPolicy policy;
    auto diagonalizer = SubspaceDiagonalizerFactory::create(policy);

    int basis_size = 2;
    // Real symmetric matrix
    std::vector<Complex> hamiltonian = {
        Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)};
    std::vector<Complex> overlap = {
        Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};
    std::vector<Real> eigenvalues(basis_size);
    std::vector<Complex> eigenvectors(basis_size * basis_size);

    // eigenvalues: 2-1=1, 2+1=3
    int info = diagonalizer->solve(basis_size,
                                   hamiltonian.data(),
                                   overlap.data(),
                                   eigenvalues.data(),
                                   eigenvectors.data(),
                                   true);

    EXPECT_EQ(info, 0);
    EXPECT_NEAR(eigenvalues[0], 1.0, 1e-10);
    EXPECT_NEAR(eigenvalues[1], 3.0, 1e-10);
    // Eigenvectors should be real
    EXPECT_NEAR(eigenvectors[0].imag(), 0.0, 1e-10);
    EXPECT_NEAR(eigenvectors[1].imag(), 0.0, 1e-10);
    EXPECT_NEAR(eigenvectors[2].imag(), 0.0, 1e-10);
    EXPECT_NEAR(eigenvectors[3].imag(), 0.0, 1e-10);
}

/**
 * A Hermitian H and a positive-definite S, 48 on a side, built from a fixed
 * recurrence so that the problem is dense, complex and non-degenerate.
 */
static void fill_generalized_problem(int basis_size,
                                     std::vector<Complex>& hamiltonian,
                                     std::vector<Complex>& overlap) {
    hamiltonian.assign(static_cast<std::size_t>(basis_size) * basis_size, Complex(0.0, 0.0));
    overlap.assign(static_cast<std::size_t>(basis_size) * basis_size, Complex(0.0, 0.0));
    auto at = [basis_size](int row, int column) {
        return static_cast<std::size_t>(row) + static_cast<std::size_t>(column) * basis_size;
    };
    for (int column = 0; column < basis_size; ++column) {
        for (int row = 0; row <= column; ++row) {
            Real real_part = std::sin(0.37 * (row + 1) + 0.11 * (column + 1) * (column + 1));
            Real imaginary_part = (row == column) ? 0.0 : std::cos(0.23 * (row + 2) * (column + 1));
            hamiltonian[at(row, column)] = Complex(real_part, imaginary_part);
            hamiltonian[at(column, row)] = Complex(real_part, -imaginary_part);
            Complex coupling = (row == column) ? Complex(2.0 + 0.01 * row, 0.0)
                                               : Complex(0.02 * real_part, 0.02 * imaginary_part);
            overlap[at(row, column)] = coupling;
            overlap[at(column, row)] = std::conj(coupling);
        }
        hamiltonian[at(column, column)] += Complex(0.5 * column, 0.0);
    }
}

TEST(SubspaceDiagonalizerTest, DivideAndConquerAgreesWithTheSerialSolver) {
    const int basis_size = 48;
    std::vector<Complex> hamiltonian, overlap;
    fill_generalized_problem(basis_size, hamiltonian, overlap);

    auto solve_with = [&](SubspaceDiagonalizer& diagonalizer,
                          std::vector<Real>& eigenvalues,
                          std::vector<Complex>& eigenvectors) {
        // Both drivers overwrite their inputs.
        std::vector<Complex> hamiltonian_copy = hamiltonian;
        std::vector<Complex> overlap_copy = overlap;
        eigenvalues.assign(basis_size, 0.0);
        eigenvectors.assign(static_cast<std::size_t>(basis_size) * basis_size, Complex(0.0, 0.0));
        return diagonalizer.solve(basis_size,
                                  hamiltonian_copy.data(),
                                  overlap_copy.data(),
                                  eigenvalues.data(),
                                  eigenvectors.data());
    };

    SerialSubspaceDiagonalizer serial;
    DivideAndConquerSubspaceDiagonalizer divide_and_conquer;
    std::vector<Real> serial_eigenvalues, eigenvalues;
    std::vector<Complex> serial_eigenvectors, eigenvectors;
    ASSERT_EQ(solve_with(serial, serial_eigenvalues, serial_eigenvectors), 0);
    ASSERT_EQ(solve_with(divide_and_conquer, eigenvalues, eigenvectors), 0);

    for (int index = 0; index < basis_size; ++index) {
        EXPECT_NEAR(eigenvalues[index], serial_eigenvalues[index], 1e-11);
    }

    // Each eigenvector solves the original problem: H c = e S c.
    for (int column = 0; column < basis_size; ++column) {
        Real largest_residual = 0.0;
        for (int row = 0; row < basis_size; ++row) {
            Complex residual(0.0, 0.0);
            for (int inner = 0; inner < basis_size; ++inner) {
                std::size_t element =
                    static_cast<std::size_t>(row) + static_cast<std::size_t>(inner) * basis_size;
                Complex coefficient = eigenvectors[static_cast<std::size_t>(inner) +
                                                   static_cast<std::size_t>(column) * basis_size];
                residual +=
                    (hamiltonian[element] - eigenvalues[column] * overlap[element]) * coefficient;
            }
            largest_residual = std::max(largest_residual, std::abs(residual));
        }
        EXPECT_LT(largest_residual, 1e-10);
    }
}

TEST(SubspaceDiagonalizerTest, ADeviceSolverGetsDivideAndConquerOnOneRank) {
    execution::SerialExecutionPolicy policy;
    auto diagonalizer = SubspaceDiagonalizerFactory::create_for_device_solver(policy);
    EXPECT_NE(dynamic_cast<DivideAndConquerSubspaceDiagonalizer*>(diagonalizer.get()), nullptr);
    EXPECT_FALSE(diagonalizer->is_collective());
}

} // namespace q3
