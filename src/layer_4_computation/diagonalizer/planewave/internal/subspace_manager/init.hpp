#pragma once

/**
 * PlanewaveSubspaceManager — abstract interface for Davidson subspace operations.
 *
 * Encapsulates all subspace-level linear algebra that could benefit from
 * GPU acceleration:
 *   - Gram-Schmidt orthogonalization (with S inner product)
 *   - Subspace matrix construction (V^H · HV, V^H · SV)
 *   - Subspace diagonalization (generalized eigenproblem)
 *   - Ritz vector extraction (full-space eigenvectors from subspace coefficients)
 *   - Subspace collapse (rotation back to number_of_bands vectors)
 *
 * The Davidson iteration loop itself remains in PlanewaveDavidsonDiagonalizer;
 * this interface handles only the per-iteration linear algebra calls.
 *
 * GPU implementations override to use cuBLAS/cuSolver for the ZGEMM and
 * eigensolver calls, keeping data on device between operations.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"

namespace q3 {

class PlanewaveSubspaceManager {
  public:
    virtual ~PlanewaveSubspaceManager() = default;

    /**
     * Whether this rank's plane-wave list begins with G = 0.
     *
     * Every gamma-only inner product below is 2 Re over the stored half sphere
     * with the origin -- held once, counted twice -- removed, and then reduced
     * across the ranks sharing the coefficients. Under a reciprocal-space
     * distribution exactly one rank holds the origin, and only that rank may
     * remove it. Replicated layouts always hold
     * it, which is the default. Set once per k-point, before the solve.
     */
    virtual void set_gamma_origin_is_local(bool origin_is_local) = 0;

    /**
     * S-orthogonalize and normalize a batch of candidate vectors against
     * the existing basis using modified Gram-Schmidt with S inner product.
     *
     * Returns the number of accepted (linearly independent) vectors.
     */
    virtual int orthogonalize_batch(int starting_vector_index,
                                    int number_of_candidates,
                                    int number_of_plane_waves,
                                    bool is_gamma_only,
                                    const OverlapOperator* overlap_operator,
                                    Complex* basis_vectors,
                                    Complex* overlap_basis_vectors) = 0;

    /**
     * Incrementally update the subspace matrices H_sub and S_sub with
     * new_columns vectors starting at start_column.
     */
    virtual void update_subspace_matrices(int start_column,
                                          int new_columns,
                                          int total_basis_size,
                                          int maximum_basis_size,
                                          int number_of_plane_waves,
                                          bool is_gamma_only,
                                          const Complex* basis_vectors,
                                          const Complex* hamiltonian_basis_vectors,
                                          const Complex* overlap_basis_vectors,
                                          Complex* subspace_hamiltonian,
                                          Complex* subspace_overlap) = 0;

    /**
     * Diagonalize the subspace by solving the generalized eigenvalue problem
     * H_sub · c = ε · S_sub · c.
     *
     * Returns the LAPACK info code (0 = success).
     */
    virtual int diagonalize_subspace(int basis_size,
                                     int maximum_basis_size,
                                     bool is_gamma_only,
                                     const Complex* subspace_hamiltonian,
                                     const Complex* subspace_overlap,
                                     Real* eigenvalues,
                                     Complex* eigenvector_matrix) = 0;

    /**
     * Collapse the subspace back to number_of_bands vectors by rotating
     * the basis with the eigenvector matrix.
     *
     * The rotation needs somewhere to build its result before overwriting the
     * array it is reading, and the caller supplies one buffer of at least
     * number_of_bands * number_of_plane_waves. One is enough because each
     * rotation reads a different array than it writes. These used to be three
     * members sized by maximum_basis_size, which made the collapse as
     * expensive in memory as the basis it collapses.
     *
     * The rotation matrix is the eigenvector matrix of the current subspace,
     * and the caller already holds that subspace: subspace_hamiltonian and
     * subspace_overlap are the incrementally updated matrices with leading
     * dimension maximum_basis_size, complete through basis_size. Diagonalising
     * them costs basis_size cubed; rebuilding them from the vectors first --
     * which is what this used to do -- costs two products of basis_size
     * squared by number_of_plane_waves on top, and was the largest single
     * compute term of a restart. Pass nullptr for either to force that rebuild,
     * which is also the fallback when the passed matrices cannot be
     * diagonalised: the Davidson loop asks for a collapse precisely when its
     * own diagonalisation of them has just failed.
     *
     * Returns the new basis size (= number_of_bands).
     */
    virtual int collapse_subspace(int basis_size,
                                  int number_of_plane_waves,
                                  int number_of_bands,
                                  int maximum_basis_size,
                                  bool is_gamma_only,
                                  bool is_identity_overlap,
                                  const OverlapOperator* overlap_operator,
                                  const Complex* subspace_hamiltonian,
                                  const Complex* subspace_overlap,
                                  Complex* basis_vectors,
                                  Complex* hamiltonian_basis_vectors,
                                  Complex* overlap_basis_vectors,
                                  Complex* rotation_scratch) = 0;

    /**
     * Extract converged eigenvectors from the subspace representation and
     * store them in the quantum system.
     */
    virtual void extract_eigenvectors(int basis_size,
                                      int number_of_plane_waves,
                                      int number_of_bands,
                                      int kpoint_index_state,
                                      bool is_gamma_only,
                                      const Complex* eigenvector_matrix,
                                      const Complex* basis_vectors,
                                      const Real* eigenvalues,
                                      Complex* rotated_scratch,
                                      PlanewaveKohnShamQuantumSystem& quantum_system) = 0;

    /**
     * Compute residual vectors: R = Hψ - εSψ.
     */
    /**
     * Build preconditioned correction vectors straight into the free columns
     * of the basis, and return how many were written.
     *
     * R_n = (HV)c_n - eps_n (SV)c_n, formed by two GEMMs whose destination is
     * the basis itself. Nothing else is materialised: no Ritz vectors, no
     * H|Ritz>, no separate residual array. That is the layout Quantum
     * ESPRESSO's Davidson uses, and it is three arrays of number_of_bands x
     * number_of_plane_waves that Q3 no longer has to hold.
     *
     * Only unconverged bands contribute, and they are compacted into
     * consecutive columns as they are written, so the caller can hand the
     * block straight to orthogonalize_batch.
     */
    virtual int build_correction_vectors(int basis_size,
                                         int number_of_plane_waves,
                                         int number_of_bands,
                                         int maximum_candidates,
                                         bool is_gamma_only,
                                         const Complex* eigenvector_matrix,
                                         int eigenvector_leading_dimension,
                                         const Real* eigenvalues,
                                         const std::vector<bool>& band_converged,
                                         const Real* kinetic_energy_squared,
                                         const Complex* hamiltonian_basis_vectors,
                                         const Complex* overlap_basis_vectors,
                                         Complex* correction_columns) = 0;

    /**
     * Apply Davidson preconditioner to residual wavefunctions.
     */
    virtual void apply_preconditioner(int number_of_bands,
                                      int number_of_plane_waves,
                                      const Real* kinetic_energy_squared,
                                      const Real* eigenvalues,
                                      const std::vector<bool>& band_converged,
                                      Complex* residual_wavefunctions) = 0;
};

} // namespace q3
