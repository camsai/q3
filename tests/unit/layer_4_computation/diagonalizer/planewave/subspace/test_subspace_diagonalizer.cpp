#include "layer_0_core/execution/serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/factory.hpp"
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

} // namespace q3
