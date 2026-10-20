#pragma once

/**
 * Seeding the DFT+U occupation matrix: the user names an eigenvalue and where
 * it should start.
 *
 * Why this exists: a Hubbard functional at practical U has *multiple*
 * self-consistent solutions -- FeO with U = 4.6 eV on Fe 3d has an
 * orbitally-ordered minimum a full 123 mRy below the orbitally-uniform one,
 * and which of them an unseeded SCF finds is a property of the starting
 * guess and the mixing trajectory, not of the physics. Reproducible minimum
 * selection therefore has to be an input, and cross-code energy comparisons
 * are only defined once both codes are known to sit in the same minimum.
 * This is also the precondition of the NiO acceptance gate, which compares
 * energies only after the converged occupation eigenvalues match.
 *
 * Semantics follow the established drop-in convention
 * (`starting_ns_eigenvalue(m,ispin,ityp) = f`): on the *first* SCF
 * iteration, after the occupation matrix n has been accumulated, reduced and
 * symmetrized, diagonalize it, replace eigenvalue m (ascending order) of the
 * named spin channel with f for every Hubbard atom of the named species, and
 * rebuild n = V diag(lambda') V^H from the *same* eigenvectors. Later
 * iterations recompute n from the wavefunctions as always -- the seed biases
 * the first Hubbard potential toward the wanted minimum and then lets
 * self-consistency take over, so a seed pointing at the basin the SCF was
 * already in is a no-op by construction, and a seed the crystal symmetry
 * forbids is washed back out by the per-iteration symmetrization exactly as
 * an unrepresentable density fluctuation would be.
 */

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"

namespace q3 {

/**
 * Replace one eigenvalue of a Hermitian occupation matrix in place.
 *
 * `matrix` is `size` x `size`, and the storage convention does not matter
 * here: the matrix is Hermitian, so row-major and column-major differ by a
 * conjugate-transpose, which commutes with "diagonalize, edit the spectrum,
 * reassemble from the same eigenvectors" -- the result is the same matrix
 * either way, just re-expressed in the caller's own convention.
 *
 * ZHEEV returns eigenvalues ascending, which pins the meaning of
 * `eigenvalue_index` = m-1: m = size selects the largest. Throws on a
 * diagonalization failure or an out-of-range index; the caller supplies the
 * context (atom, species) for a useful message.
 */
inline void apply_hubbard_starting_eigenvalue(std::vector<Complex>& matrix,
                                              int size,
                                              int eigenvalue_index,
                                              Real value) {
    if (size <= 0)
        return;
    if (eigenvalue_index < 0 || eigenvalue_index >= size)
        throw std::runtime_error("hubbard starting eigenvalue index " +
                                 std::to_string(eigenvalue_index + 1) +
                                 " is outside the manifold (size " + std::to_string(size) + ")");

    std::vector<Complex> eigenvectors = matrix; // ZHEEV overwrites input with eigenvectors.
    std::vector<Real> eigenvalues(static_cast<size_t>(size), 0.0);
    int work_size = std::max(1, 2 * size - 1);
    std::vector<Complex> work(static_cast<size_t>(work_size));
    std::vector<Real> rwork(static_cast<size_t>(std::max(1, 3 * size - 2)));
    int info = 0;
    char jobz = 'V';
    char uplo = 'U';
    int dimension = size;
    lapack::zheev_(&jobz,
                   &uplo,
                   &dimension,
                   reinterpret_cast<std::complex<double>*>(eigenvectors.data()),
                   &dimension,
                   eigenvalues.data(),
                   reinterpret_cast<std::complex<double>*>(work.data()),
                   &work_size,
                   rwork.data(),
                   &info);
    if (info != 0)
        throw std::runtime_error("apply_hubbard_starting_eigenvalue: ZHEEV failed, info=" +
                                 std::to_string(info));

    eigenvalues[static_cast<size_t>(eigenvalue_index)] = value;

    /**
     * n' = W diag(lambda') W^H, in the same layout `eigenvectors` came back
     * in. ZHEEV's W is column-major (eigenvector k = column k), so
     * W(a,k) = eigenvectors[a + k*size].
     */
    for (int row = 0; row < size; row++) {
        for (int column = 0; column < size; column++) {
            Complex accumulated(0.0, 0.0);
            for (int k = 0; k < size; k++) {
                accumulated +=
                    eigenvectors[static_cast<size_t>(row) + static_cast<size_t>(k) * size] *
                    eigenvalues[static_cast<size_t>(k)] *
                    std::conj(
                        eigenvectors[static_cast<size_t>(column) + static_cast<size_t>(k) * size]);
            }
            matrix[static_cast<size_t>(row) + static_cast<size_t>(column) * size] = accumulated;
        }
    }
}

} // namespace q3
