#pragma once

/**
 * SubspaceDiagonalizer — interface for small-matrix diagonalization.
 *
 * Used inside iterative solvers (Davidson, Lanczos) to diagonalize the
 * projected Hamiltonian and overlap matrices in the basis subspace.
 * Also used by direct solvers for the final diagonalization step.
 *
 * Implementations provide serial (LAPACK) or distributed (ScaLAPACK) variants.
 */

#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

class SubspaceDiagonalizer {
  public:
    virtual ~SubspaceDiagonalizer() = default;

    /**
     * Solve the generalized eigenvalue problem: H * C = S * C * E
     *
     * @param basis_size            Dimension of the subspace (matrix size).
     * @param hamiltonian_matrix    Projected Hamiltonian (destroyed/overwritten).
     *                              May be the same buffer as eigenvector_matrix:
     *                              every implementation reads it whole before it
     *                              writes any eigenvector, which is what lets the
     *                              caller keep one copy of H rather than two.
     * @param overlap_matrix        Projected overlap (destroyed/overwritten).
     * @param eigenvalues           Output eigenvalues (size basis_size).
     * @param eigenvector_matrix    Output eigenvectors (basis_size x basis_size).
     * @return                      0 on success, non-zero on failure.
     */
    virtual int solve(int basis_size,
                      Complex* hamiltonian_matrix,
                      Complex* overlap_matrix,
                      Real* eigenvalues,
                      Complex* eigenvector_matrix,
                      bool is_gamma_only = false) = 0;

    /**
     * Whether solve() is a collective operation over the process group.
     *
     * Collective solvers (ScaLAPACK) require EVERY rank of the group to call
     * solve() — gating the call on one rank deadlocks the group — and return
     * redundant results on all ranks, so no post-solve broadcast is needed.
     * Non-collective solvers (serial LAPACK) may be called on the group root
     * only, with results broadcast afterwards.
     */
    virtual bool is_collective() const {
        return false;
    }

    virtual std::string name() const = 0;
};

} // namespace q3
