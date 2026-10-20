#pragma once

/**
 * Löwdin (symmetric) orthogonalization of a G-space orbital set.
 *
 * Given a set of (generally non-orthogonal) atomic pseudo-orbitals |φ_i⟩,
 * builds the overlap matrix S_ij = ⟨φ_i|φ_j⟩ and returns the symmetrically
 * orthogonalized basis |φ'_i⟩ = Σ_j |φ_j⟩ (S^{-1/2})_ji, where S^{-1/2} is
 * formed from the eigendecomposition S = W·diag(e)·W^H. This is the unique
 * orthogonalization that stays "closest" (in a least-squares sense) to the
 * original basis. It is the convention Quantum ESPRESSO's projwfc.x names
 * "ortho-atomic": the atomic projector orbitals are unconditionally
 * Löwdin-orthogonalized before Kohn-Sham states are projected onto them, with
 * no "plain atomic, unorthogonalized" alternative offered by that tool (the
 * distinction exists only for the DFT+U Hubbard_projectors input variable).
 * Operates on the orbital set for a single k-point, as built by
 * atomic_orbital_builder.hpp.
 */

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * S^{-1/2} for one orbital set, together with the eigendecomposition it was
 * built from.
 *
 * The eigenpairs are kept rather than discarded because the *derivative* of
 * S^{-1/2} cannot be recovered from S^{-1/2} alone: dS and S^{-1/2} do not
 * commute, so d(S^{-1/2}) is the solution of a Lyapunov equation that is
 * diagonal only in S's own eigenbasis (see
 * lowdin_inverse_square_root_derivative). DFT+U forces with ortho-atomic
 * projectors need exactly that, and re-diagonalizing S per displaced atom and
 * Cartesian direction to get it back would be both wasteful and a second
 * definition of the same transform.
 *
 * Both matrices are column-major -- element (row, column) at
 * `row + column * number_of_orbitals`. `eigenvectors` is ZHEEV's own output
 * (column k is eigenvector k); `inverse_square_root` is Hermitian, so its
 * layout only fixes which index is written first.
 */
struct LowdinTransform {
    int number_of_orbitals = 0;
    std::vector<Real> eigenvalues;
    std::vector<Complex> eigenvectors;
    std::vector<Complex> inverse_square_root;
};

/**
 * Diagonalize the overlap matrix `overlap_column_major` (Hermitian, column-major
 * number_of_orbitals x number_of_orbitals) and form S^{-1/2} from it.
 *
 * Takes the overlap by value: ZHEEV overwrites its input with the eigenvectors,
 * which is exactly what the returned transform wants, and the caller keeps its
 * own copy. Throws if the overlap is (near-)singular, i.e. the orbitals are not
 * linearly independent -- not expected for atomic orbitals in practice, but a
 * real possibility that must fail loudly rather than silently produce NaN/Inf.
 */
[[nodiscard]] inline LowdinTransform
compute_lowdin_transform(std::vector<Complex> overlap_column_major, int number_of_orbitals) {
    LowdinTransform transform;
    transform.number_of_orbitals = number_of_orbitals;
    if (number_of_orbitals == 0)
        return transform;

    // Diagonalize S = W * diag(eigenvalues) * W^H via ZHEEV.
    transform.eigenvalues.assign(static_cast<size_t>(number_of_orbitals), 0.0);
    int work_size = std::max(1, 2 * number_of_orbitals - 1);
    std::vector<Complex> work(work_size);
    std::vector<Real> rwork(static_cast<size_t>(std::max(1, 3 * number_of_orbitals - 2)));
    int info = 0;
    char jobz = 'V';
    char uplo = 'U';
    lapack::zheev_(&jobz,
                   &uplo,
                   &number_of_orbitals,
                   reinterpret_cast<std::complex<double>*>(overlap_column_major.data()),
                   &number_of_orbitals,
                   transform.eigenvalues.data(),
                   reinterpret_cast<std::complex<double>*>(work.data()),
                   &work_size,
                   rwork.data(),
                   &info);
    if (info != 0) {
        throw std::runtime_error("compute_lowdin_transform: ZHEEV failed, info=" +
                                 std::to_string(info));
    }

    for (int k = 0; k < number_of_orbitals; k++) {
        if (transform.eigenvalues[static_cast<size_t>(k)] <
            math::tolerances::lowdin_minimum_overlap_eigenvalue) {
            throw std::runtime_error(
                "compute_lowdin_transform: near-singular overlap matrix (eigenvalue " +
                std::to_string(transform.eigenvalues[static_cast<size_t>(k)]) +
                ") -- atomic orbitals are not linearly independent");
        }
    }
    transform.eigenvectors = std::move(overlap_column_major);

    // S^{-1/2}(a,b) = sum_k W(a,k) * (1/sqrt(e_k)) * conj(W(b,k)).
    transform.inverse_square_root.assign(
        static_cast<size_t>(number_of_orbitals) * number_of_orbitals, Complex(0.0, 0.0));
    for (int k = 0; k < number_of_orbitals; k++) {
        Real inverse_square_root_eigenvalue =
            1.0 / std::sqrt(transform.eigenvalues[static_cast<size_t>(k)]);
        for (int a = 0; a < number_of_orbitals; a++) {
            Complex weighted_component =
                transform.eigenvectors[static_cast<size_t>(a + k * number_of_orbitals)] *
                inverse_square_root_eigenvalue;
            for (int b = 0; b < number_of_orbitals; b++) {
                transform.inverse_square_root[static_cast<size_t>(a + b * number_of_orbitals)] +=
                    weighted_component *
                    std::conj(
                        transform.eigenvectors[static_cast<size_t>(b + k * number_of_orbitals)]);
            }
        }
    }
    return transform;
}

/**
 * Form the rank-local orbital overlap matrix in column-major layout.
 *
 * A caller whose plane-wave dimension is distributed must sum this matrix
 * across the distribution group before constructing a Lowdin transform.
 */
[[nodiscard]] inline std::vector<Complex>
compute_orbital_overlap_matrix(const std::vector<std::vector<Complex>>& orbitals) {
    int number_of_orbitals = static_cast<int>(orbitals.size());
    if (number_of_orbitals == 0)
        return {};
    int number_of_plane_waves = static_cast<int>(orbitals[0].size());

    // Overlap S_ij = <phi_i|phi_j>, packed column-major for LAPACK.
    std::vector<Complex> overlap_column_major(static_cast<size_t>(number_of_orbitals) *
                                              number_of_orbitals);
    for (int column = 0; column < number_of_orbitals; column++) {
        for (int row = 0; row < number_of_orbitals; row++) {
            Complex value(0.0, 0.0);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                value += std::conj(orbitals[row][g_index]) * orbitals[column][g_index];
            }
            overlap_column_major[static_cast<size_t>(row + column * number_of_orbitals)] = value;
        }
    }
    return overlap_column_major;
}

/**
 * Symmetrically orthogonalize `orbitals` using a complete overlap matrix.
 *
 * The explicit overlap form lets a distributed caller sum rank-local
 * contractions before every rank constructs the same Lowdin transform.
 */
inline void lowdin_orthogonalize_orbitals(std::vector<std::vector<Complex>>& orbitals,
                                          std::vector<Complex> overlap_column_major) {
    int number_of_orbitals = static_cast<int>(orbitals.size());
    if (number_of_orbitals == 0)
        return;
    int number_of_plane_waves = static_cast<int>(orbitals[0].size());
    size_t expected_overlap_size = static_cast<size_t>(number_of_orbitals) * number_of_orbitals;
    if (overlap_column_major.size() != expected_overlap_size) {
        throw std::runtime_error("lowdin_orthogonalize_orbitals: overlap matrix has " +
                                 std::to_string(overlap_column_major.size()) +
                                 " elements, expected " + std::to_string(expected_overlap_size));
    }

    auto transform = compute_lowdin_transform(std::move(overlap_column_major), number_of_orbitals);

    // new_orbitals[i] = sum_j orbitals[j] * (S^{-1/2})(j,i).
    std::vector<std::vector<Complex>> orthogonalized_orbitals(
        number_of_orbitals, std::vector<Complex>(number_of_plane_waves, Complex(0.0, 0.0)));
    for (int i = 0; i < number_of_orbitals; i++) {
        for (int j = 0; j < number_of_orbitals; j++) {
            Complex weight =
                transform.inverse_square_root[static_cast<size_t>(j + i * number_of_orbitals)];
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                orthogonalized_orbitals[i][g_index] += weight * orbitals[j][g_index];
            }
        }
    }

    orbitals = std::move(orthogonalized_orbitals);
}

/**
 * Symmetrically orthogonalize a complete, locally stored orbital set.
 */
inline void lowdin_orthogonalize_orbitals(std::vector<std::vector<Complex>>& orbitals) {
    lowdin_orthogonalize_orbitals(orbitals, compute_orbital_overlap_matrix(orbitals));
}

/**
 * d(S^{-1/2}) given dS, in the same column-major layout.
 *
 * S^{1/2} S^{1/2} = S differentiates into a Sylvester equation, which is
 * diagonal in S's eigenbasis; inverting it there and mapping back gives the
 * closed form
 *
 *     d(S^{-1/2}) = -W * [ (W^H dS W)_ij / (e_i sqrt(e_j) + e_j sqrt(e_i)) ] * W^H
 *
 * -- Eq. (32) of Phys. Rev. B 102, 235159 (2020), the same closed form
 * Quantum ESPRESSO evaluates for ortho-atomic DFT+U forces. The denominator
 * is `sqrt(e_i) sqrt(e_j) (sqrt(e_i) + sqrt(e_j))` regrouped: the outer factors
 * come from d(S^{-1/2}) = -S^{-1/2} d(S^{1/2}) S^{-1/2} and the bracket from
 * d(S^{1/2}) itself.
 *
 * dS Hermitian in, d(S^{-1/2}) Hermitian out: the eigenbasis transform and the
 * real symmetric denominator both preserve it, which is what lets a caller read
 * the result with either index order.
 */
[[nodiscard]] inline std::vector<Complex>
lowdin_inverse_square_root_derivative(const LowdinTransform& transform,
                                      const std::vector<Complex>& overlap_derivative) {
    int number_of_orbitals = transform.number_of_orbitals;
    size_t element_count = static_cast<size_t>(number_of_orbitals) * number_of_orbitals;
    if (overlap_derivative.size() != element_count) {
        throw std::runtime_error("lowdin_inverse_square_root_derivative: overlap derivative has " +
                                 std::to_string(overlap_derivative.size()) +
                                 " elements, expected " + std::to_string(element_count));
    }
    std::vector<Complex> result(element_count, Complex(0.0, 0.0));
    if (number_of_orbitals == 0)
        return result;

    const auto& eigenvectors = transform.eigenvectors;
    const auto& eigenvalues = transform.eigenvalues;
    auto index = [number_of_orbitals](int row, int column) {
        return static_cast<size_t>(row + column * number_of_orbitals);
    };

    // dS * W, then W^H * (dS * W): the derivative in S's own eigenbasis.
    std::vector<Complex> right_product(element_count, Complex(0.0, 0.0));
    for (int column = 0; column < number_of_orbitals; column++)
        for (int inner = 0; inner < number_of_orbitals; inner++) {
            Complex eigenvector_component = eigenvectors[index(inner, column)];
            if (eigenvector_component == Complex(0.0, 0.0))
                continue;
            for (int row = 0; row < number_of_orbitals; row++)
                right_product[index(row, column)] +=
                    overlap_derivative[index(row, inner)] * eigenvector_component;
        }
    std::vector<Complex> eigenbasis_derivative(element_count, Complex(0.0, 0.0));
    for (int column = 0; column < number_of_orbitals; column++)
        for (int row = 0; row < number_of_orbitals; row++) {
            Complex accumulated(0.0, 0.0);
            for (int inner = 0; inner < number_of_orbitals; inner++)
                accumulated += std::conj(eigenvectors[index(inner, row)]) *
                               right_product[index(inner, column)];
            eigenbasis_derivative[index(row, column)] = accumulated;
        }

    // Divide by the Sylvester denominator, then map back: -W * (...) * W^H.
    std::vector<Real> square_root_eigenvalues(static_cast<size_t>(number_of_orbitals));
    for (int k = 0; k < number_of_orbitals; k++)
        square_root_eigenvalues[static_cast<size_t>(k)] =
            std::sqrt(eigenvalues[static_cast<size_t>(k)]);
    for (int column = 0; column < number_of_orbitals; column++)
        for (int row = 0; row < number_of_orbitals; row++)
            eigenbasis_derivative[index(row, column)] /=
                eigenvalues[static_cast<size_t>(row)] *
                    square_root_eigenvalues[static_cast<size_t>(column)] +
                eigenvalues[static_cast<size_t>(column)] *
                    square_root_eigenvalues[static_cast<size_t>(row)];

    std::vector<Complex> left_product(element_count, Complex(0.0, 0.0));
    for (int column = 0; column < number_of_orbitals; column++)
        for (int inner = 0; inner < number_of_orbitals; inner++) {
            Complex eigenvector_component = std::conj(eigenvectors[index(column, inner)]);
            if (eigenvector_component == Complex(0.0, 0.0))
                continue;
            for (int row = 0; row < number_of_orbitals; row++)
                left_product[index(row, column)] +=
                    eigenbasis_derivative[index(row, inner)] * eigenvector_component;
        }
    for (int column = 0; column < number_of_orbitals; column++)
        for (int row = 0; row < number_of_orbitals; row++) {
            Complex accumulated(0.0, 0.0);
            for (int inner = 0; inner < number_of_orbitals; inner++)
                accumulated += eigenvectors[index(row, inner)] * left_product[index(inner, column)];
            result[index(row, column)] = -accumulated;
        }
    return result;
}

} // namespace q3
