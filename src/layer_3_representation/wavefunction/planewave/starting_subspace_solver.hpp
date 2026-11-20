#pragma once

/**
 * StartingSubspaceSolver — the three large operations of the starting rotation,
 * as the wavefunction initializer needs them and nothing more.
 *
 * The rotation takes a set of starting vectors χ, forms H and S in their
 * subspace, solves the small eigenproblem, and rotates χ by its eigenvectors.
 * Everything that scales with the number of plane waves is in three steps:
 * H applied to every χ, the two projections χ†Hχ and χ†χ, and the rotation
 * χ·c. The initializer has a host implementation of each. A solver that
 * already applies H somewhere faster -- on a device -- implements this
 * interface, one layer up, and keeps χ and Hχ where it computed them. The
 * rotated vectors are the k-point state's bands, and the solver stores them in
 * the quantum system itself, where it computed them: the three steps cost one
 * transfer of χ in and none of the bands out.
 *
 * The overlap is the identity here: a solver offers this only for the standard
 * eigenproblem (see the diagonalizer that hands it out).
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

class PlanewaveKohnShamQuantumSystem;

/// H, the subspace projections and the rotation for one k-point's starting vectors.
class StartingSubspaceSolver {
  public:
    virtual ~StartingSubspaceSolver() = default;

    /**
     * Take the starting vectors and apply H to them.
     *
     * `vectors` is column-major host memory, χ[g + vector * number_of_plane_waves], with the
     * plane-wave count of the k-point this solver was prepared for. Both χ and Hχ are kept
     * for the two calls below.
     */
    virtual void apply_hamiltonian(const Complex* vectors, int number_of_vectors) = 0;

    /**
     * Form the subspace matrices of the vectors last given to apply_hamiltonian():
     * H_ij = χ_i†(Hχ)_j and S_ij = χ_i†χ_j, each number_of_vectors squared, column-major.
     */
    virtual void project(Complex* hamiltonian_matrix, Complex* overlap_matrix) = 0;

    /**
     * Rotate the vectors and store the result as the bands of one k-point state:
     * band n = χ · (column n of coefficients).
     *
     * `coefficients` holds one column per band, column-major with leading dimension
     * number_of_vectors. Every band of the k-point state is written, number_of_bands of
     * them, each over the plane-wave count this solver was prepared for.
     */
    virtual void rotate_into_bands(const Complex* coefficients,
                                   int number_of_bands,
                                   PlanewaveKohnShamQuantumSystem& quantum_system,
                                   int kpoint_index_state) = 0;
};

} // namespace q3
