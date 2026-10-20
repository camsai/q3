#pragma once

/**
 * OverlapOperator — the S operator of the generalized eigenproblem H c = ε S c,
 * expressed purely in the plane-wave basis.
 *
 * When the basis is orthonormal, S = I and the eigenproblem is standard (zheev).
 * When the basis carries an additional overlap, S becomes non-trivial and the
 * eigenproblem is generalized (zhegv). This strategy isolates that difference so
 * the Davidson and direct diagonalizers share a single, branch-free eigensolver
 * path. Concrete operators inject whatever physical data they require through
 * their constructors, keeping this interface free of any model-specific concepts.
 */

#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"

#include "layer_0_core/math/linear_algebra/lapack.hpp"

namespace q3 {

class OverlapOperator {
  public:
    virtual ~OverlapOperator() = default;

    // Apply S to a single wavefunction: overlap_wavefunction = S · wavefunction.
    virtual void apply(int number_of_plane_waves,
                       const Complex* wavefunction,
                       Complex* overlap_wavefunction) const = 0;

    /**
     * Apply S to multiple wavefunctions in batch.
     * wavefunctions[vector_index * number_of_plane_waves + g] and
     * overlap_wavefunctions[vector_index * number_of_plane_waves + g]. Default: calls per-vector
     * apply() in a loop. Subclasses may override for efficiency.
     */
    virtual void apply_batch(int number_of_plane_waves,
                             int number_of_vectors,
                             const Complex* wavefunctions,
                             Complex* overlap_wavefunctions) const {
        for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
            apply(number_of_plane_waves,
                  wavefunctions + static_cast<size_t>(vector_index) * number_of_plane_waves,
                  overlap_wavefunctions +
                      static_cast<size_t>(vector_index) * number_of_plane_waves);
        }
    }

    /**
     * Returns true if S = I (norm-conserving case).
     * When true, Davidson can skip S·ψ ZGEMM and S_sub construction.
     */
    [[nodiscard]] virtual bool is_identity() const {
        return false;
    }

    /**
     * Build the dense overlap matrix S (column-major, number_of_plane_waves ×
     * number_of_plane_waves) used by the direct solver. An orthonormal basis
     * leaves it empty (S = I is implicit in the standard solver).
     */
    virtual void build_dense_matrix(int number_of_plane_waves,
                                    std::vector<Complex>& overlap_matrix) const = 0;
};

} // namespace q3
