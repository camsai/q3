#pragma once

/**
 * DivideAndConquerSubspaceDiagonalizer — the serial subspace solver with
 * LAPACK's divide-and-conquer driver for the complex problem.
 *
 * The Davidson subspace problem asks for every eigenvector of a matrix a few
 * hundred on a side, hundreds of times per run. The QR driver the serial solver
 * uses (zhegv) spends most of that in the tridiagonal eigenvectors;
 * divide and conquer (zhegvd) computes the same decomposition several times
 * faster. On a device build, where the rest of the eigensolver runs on the
 * device and this one step stays on the host, it was a fifth of the
 * eigensolver's time on 64-atom silicon.
 *
 * The two drivers agree to rounding, not to the last bit, and within a
 * degenerate eigenvalue they may return a different orthonormal basis of the
 * same space. That is why this is a solver of its own, chosen where a run asks
 * for it, and not a change to the serial solver every CPU reference was
 * produced with.
 */

#include <algorithm>
#include <vector>

#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/serial.hpp"

namespace q3 {

/// Serial subspace solver that takes the complex generalized problem by divide and conquer.
class DivideAndConquerSubspaceDiagonalizer : public SerialSubspaceDiagonalizer {
  public:
    /**
     * Solve H c = e S c for all eigenpairs. The real (gamma-point) problem is
     * left to the serial solver; the complex one goes to zhegvd.
     */
    int solve(int basis_size,
              Complex* hamiltonian_matrix,
              Complex* overlap_matrix,
              Real* eigenvalues,
              Complex* eigenvector_matrix,
              bool is_gamma_only = false) override {
        if (basis_size == 0 || is_gamma_only) {
            return SerialSubspaceDiagonalizer::solve(basis_size,
                                                     hamiltonian_matrix,
                                                     overlap_matrix,
                                                     eigenvalues,
                                                     eigenvector_matrix,
                                                     is_gamma_only);
        }

        // The driver leaves the eigenvectors where the Hamiltonian was.
        if (hamiltonian_matrix != eigenvector_matrix) {
            std::copy(hamiltonian_matrix,
                      hamiltonian_matrix + static_cast<std::size_t>(basis_size) * basis_size,
                      eigenvector_matrix);
        }

        int info = 0;
        int diagonalization_type = 1; // H*x = lambda*S*x
        char jobz = 'V';
        char uplo = 'U';
        auto* cast_eigenvectors = reinterpret_cast<std::complex<double>*>(eigenvector_matrix);
        auto* cast_overlap = reinterpret_cast<std::complex<double>*>(overlap_matrix);

        // Workspace query: the driver states the three sizes it wants.
        std::complex<double> work_query;
        Real real_work_query = 0.0;
        int integer_work_query = 0;
        int query = -1;
        lapack::zhegvd_(&diagonalization_type,
                        &jobz,
                        &uplo,
                        &basis_size,
                        cast_eigenvectors,
                        &basis_size,
                        cast_overlap,
                        &basis_size,
                        eigenvalues,
                        &work_query,
                        &query,
                        &real_work_query,
                        &query,
                        &integer_work_query,
                        &query,
                        &info);
        if (info != 0) {
            return info;
        }

        int complex_work_size = std::max(1, static_cast<int>(work_query.real()));
        int real_work_size = std::max(1, static_cast<int>(real_work_query));
        int integer_work_size = std::max(1, integer_work_query);
        complex_work_.resize(static_cast<std::size_t>(complex_work_size));
        real_work_.resize(static_cast<std::size_t>(real_work_size));
        integer_work_.resize(static_cast<std::size_t>(integer_work_size));

        lapack::zhegvd_(&diagonalization_type,
                        &jobz,
                        &uplo,
                        &basis_size,
                        cast_eigenvectors,
                        &basis_size,
                        cast_overlap,
                        &basis_size,
                        eigenvalues,
                        complex_work_.data(),
                        &complex_work_size,
                        real_work_.data(),
                        &real_work_size,
                        integer_work_.data(),
                        &integer_work_size,
                        &info);
        return info;
    }

    /// Name of the solver, for the run summary.
    std::string name() const override {
        return "Serial (LAPACK, divide and conquer)";
    }

  private:
    /// Workspaces kept between calls: the solver runs hundreds of times at the same size.
    std::vector<std::complex<double>> complex_work_;
    std::vector<Real> real_work_;
    std::vector<int> integer_work_;
};

} // namespace q3
