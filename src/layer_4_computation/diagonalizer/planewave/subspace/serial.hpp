#pragma once

/**
 * SerialSubspaceDiagonalizer — LAPACK implementation of SubspaceDiagonalizer.
 *
 * Uses zhegv/dsygv for generalized eigenvalue problems on a single process.
 */

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/init.hpp"
#include <algorithm>

namespace q3 {

class SerialSubspaceDiagonalizer : public SubspaceDiagonalizer {
  public:
    int solve(int basis_size,
              Complex* hamiltonian_matrix,
              Complex* overlap_matrix,
              Real* eigenvalues,
              Complex* eigenvector_matrix,
              bool is_gamma_only = false) override {
        if (basis_size == 0)
            return 0;

        int info;
        int diagonalization_type = 1; // H*x = lambda*S*x
        char jobz = 'V';              // Compute eigenvalues and eigenvectors
        char uplo = 'U';              // Upper triangle stored

        const int matrix_elements = basis_size * basis_size;
        if (is_gamma_only) {
            /**
             * Gamma-only: H_sub and S_sub are real symmetric matrices, so dsygv
             * solves them and its eigenvectors come out real.
             *
             * Both are packed to real in place rather than into copies of
             * their own: the real part of complex entry i is written to double
             * entry i, which lies inside complex entry i/2 -- already read --
             * so a forward pass is safe, and the eigenvectors are expanded
             * back to complex by a backward pass for the same reason. The
             * matrices are the caller's to overwrite, and the Hamiltonian is
             * usually the eigenvector buffer itself.
             */
            if (hamiltonian_matrix != eigenvector_matrix) {
                std::copy(
                    hamiltonian_matrix, hamiltonian_matrix + matrix_elements, eigenvector_matrix);
            }
            auto* h_real = reinterpret_cast<double*>(eigenvector_matrix);
            auto* s_real = reinterpret_cast<double*>(overlap_matrix);
            for (int i = 0; i < matrix_elements; ++i) {
                h_real[i] = eigenvector_matrix[i].real();
                s_real[i] = overlap_matrix[i].real();
            }

            double work_query;
            int lwork = -1;
            lapack::dsygv_(&diagonalization_type,
                           &jobz,
                           &uplo,
                           &basis_size,
                           h_real,
                           &basis_size,
                           s_real,
                           &basis_size,
                           eigenvalues,
                           &work_query,
                           &lwork,
                           &info);

            lwork = static_cast<int>(work_query);
            std::vector<double> work(lwork);
            lapack::dsygv_(&diagonalization_type,
                           &jobz,
                           &uplo,
                           &basis_size,
                           h_real,
                           &basis_size,
                           s_real,
                           &basis_size,
                           eigenvalues,
                           work.data(),
                           &lwork,
                           &info);

            // Expand the real eigenvectors back to complex, in place, backward.
            for (int i = matrix_elements - 1; i >= 0; --i) {
                eigenvector_matrix[i] = Complex(h_real[i], 0.0);
            }
        } else {
            /**
             * Non-gamma: use zhegv for complex Hermitian matrices. It leaves
             * the eigenvectors where the Hamiltonian was, so the Hamiltonian
             * goes into the eigenvector buffer -- unless the caller already
             * handed the same buffer as both.
             */
            if (hamiltonian_matrix != eigenvector_matrix) {
                std::copy(
                    hamiltonian_matrix, hamiltonian_matrix + matrix_elements, eigenvector_matrix);
            }

            // Workspace query
            Complex work_query;
            int lwork = -1;
            std::vector<Real> rwork(std::max(1, 3 * basis_size - 2));

            auto* cast_eigenvectors = reinterpret_cast<std::complex<double>*>(eigenvector_matrix);
            auto* cast_overlap = reinterpret_cast<std::complex<double>*>(overlap_matrix);
            auto* cast_work_query = reinterpret_cast<std::complex<double>*>(&work_query);

            lapack::zhegv_(&diagonalization_type,
                           &jobz,
                           &uplo,
                           &basis_size,
                           cast_eigenvectors,
                           &basis_size,
                           cast_overlap,
                           &basis_size,
                           eigenvalues,
                           cast_work_query,
                           &lwork,
                           rwork.data(),
                           &info);

            lwork = static_cast<int>(work_query.real());
            std::vector<Complex> work(lwork);
            auto* cast_work = reinterpret_cast<std::complex<double>*>(work.data());

            lapack::zhegv_(&diagonalization_type,
                           &jobz,
                           &uplo,
                           &basis_size,
                           cast_eigenvectors,
                           &basis_size,
                           cast_overlap,
                           &basis_size,
                           eigenvalues,
                           cast_work,
                           &lwork,
                           rwork.data(),
                           &info);
        }

        return info;
    }

    std::string name() const override {
        return "Serial (LAPACK)";
    }
};

} // namespace q3
