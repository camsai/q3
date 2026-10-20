#pragma once

/**
 * SubspaceDisentangler — choose, at every k-point, the N-dimensional subspace
 * of the kept bands that varies most smoothly across the Brillouin zone.
 *
 * This is the Souza-Marzari-Vanderbilt iteration (PRB 65, 035109). It is what
 * makes an entangled manifold interpolable: when the bands worth modelling are
 * not separated from everything above them at every k -- a transition metal's
 * d and s bands, where the sp band leaves the manifold and comes back -- no
 * choice of band range works, because the state sitting at a given index is
 * not the same state everywhere.
 *
 * What is minimised is the gauge-invariant spread
 *
 *     Omega_I = (1/N_k) sum_k sum_b w_b [ N - sum_mn |M~_mn(k,b)|^2 ]
 *
 * with M~ the overlap between the CHOSEN subspaces at k and k+b. Equivalently,
 * and this is what the iteration does, one maximises sum_k Tr[P(k) Z(k)] with
 *
 *     Z(k) = sum_b w_b M(k,b) P(k+b) M(k,b)^dagger
 *
 * Omega_I comes out in BOHR^2: the record's b-vectors are in inverse bohr, and
 * the weights that satisfy sum_b w_b b_a b_b = delta_ab therefore carry bohr^2.
 * Nothing here is in Angstrom.
 *
 * Three details decide whether this is correct rather than merely plausible.
 *
 *  - Discarded bands are gone before anything starts. The working basis at a
 *    k-point is its kept bands, and every overlap block is sliced on BOTH
 *    sides, with the row k-point's keep-list and the column k-point's, because
 *    the two differ. Leave them in and the optimiser selects them back, making
 *    the classification decorative.
 *  - The frozen states are held out by diagonalising in an explicit basis of
 *    their complement, not by projecting them out in place. The projected
 *    matrix carries one exact zero eigenvalue per frozen direction, so a
 *    genuine complement direction near zero can lose to one of them and a
 *    frozen state rotates back in.
 *  - Z is damped between iterations. The undamped iteration oscillates on
 *    metals; 0.5 is the same mixing Wannier90 defaults to.
 *
 * The result is one semi-unitary V(k), kept-bands by N, per k-point: the
 * columns span the chosen subspace. The gauge is then computed from
 * V(k)^dagger A(k), which is square, and the Hamiltonian rotation receives the
 * composite V(k) U(k).
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/convergence_window.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/hermitian_eigensolver.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/disentanglement_result.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/disentanglement_settings.hpp"

namespace q3 {

class SubspaceDisentangler {
  public:
    /**
     * `classification` is one verdict per k-point of the stencil: the kept
     * bands and the frozen ones. `number_of_wannier_functions` is the
     * dimension of the subspace chosen at every one of them.
     */
    SubspaceDisentangler(const KPointStencil& stencil,
                         const std::vector<KPointBandClassification>& classification,
                         int number_of_wannier_functions,
                         DisentanglementSettings settings)
        : stencil_(stencil)
        , classification_(classification)
        , number_of_wannier_functions_(number_of_wannier_functions)
        , settings_(settings) {
        if (number_of_wannier_functions < 1) {
            throw std::invalid_argument(
                "SubspaceDisentangler: number_of_wannier_functions must be positive, got " +
                std::to_string(number_of_wannier_functions));
        }
        if (static_cast<int>(classification.size()) != stencil.number_of_kpoints()) {
            throw std::invalid_argument("SubspaceDisentangler: the classification covers " +
                                        std::to_string(classification.size()) +
                                        " k-points and the stencil " +
                                        std::to_string(stencil.number_of_kpoints()));
        }
        if (settings.maximum_iterations < 1) {
            throw std::invalid_argument(
                "SubspaceDisentangler: maximum_iterations must be at least 1, got " +
                std::to_string(settings.maximum_iterations));
        }
        if (!(settings.damping > 0.0 && settings.damping <= 1.0)) {
            throw std::invalid_argument("SubspaceDisentangler: damping must lie in (0, 1], got " +
                                        std::to_string(settings.damping));
        }
    }

    /**
     * `overlaps` are M_mn(k,b) over the CAPTURED bands, exactly as the record
     * stores them; they are sliced to the keep-lists here, once.
     * `initial_subspaces[k]` is kept(k) x N, the orthonormalised column space
     * of A(k) restricted to the kept bands.
     */
    [[nodiscard]] DisentanglementResult
    run(const BlochOverlaps& overlaps, std::vector<DenseMatrix<Complex>> initial_subspaces) const {
        const int number_of_kpoints = stencil_.number_of_kpoints();
        require_subspace_shapes(initial_subspaces);
        const BlochOverlaps sliced = overlaps.restricted_to(classification_, stencil_);

        std::vector<DenseMatrix<Complex>> subspaces = std::move(initial_subspaces);
        std::vector<DenseMatrix<Complex>> damped_z(static_cast<std::size_t>(number_of_kpoints));
        ConvergenceWindow window(settings_.convergence_tolerance, settings_.convergence_window);

        for (int iteration = 1; iteration <= settings_.maximum_iterations; iteration++) {
            Real spread = 0.0;
            std::vector<DenseMatrix<Complex>> next_subspaces(
                static_cast<std::size_t>(number_of_kpoints));

            for (int kpoint = 0; kpoint < number_of_kpoints; kpoint++) {
                const DenseMatrix<Complex> z =
                    subspace_matrix_at(kpoint, sliced, subspaces, spread);
                DenseMatrix<Complex>& damped = damped_z[static_cast<std::size_t>(kpoint)];
                damped = damped.empty() ? z : mixed(damped, z);
                next_subspaces[static_cast<std::size_t>(kpoint)] =
                    choose_subspace(damped, classification_[static_cast<std::size_t>(kpoint)]);
            }

            subspaces = std::move(next_subspaces);
            spread /= static_cast<Real>(number_of_kpoints);
            require_nonnegative(spread);
            window.record(spread);

            if (window.has_converged()) {
                /**
                 * The spread accumulated above belongs to the subspaces this
                 * iteration STARTED from, because those are what Z was built
                 * against; the subspaces about to be returned are one update
                 * further on. At convergence the two differ by less than the
                 * tolerance, but "within tolerance" is not "the same number",
                 * and a caller that prints Omega_I beside a model is entitled
                 * to the Omega_I of THAT model. One extra pass, once.
                 */
                const Real final_spread = spread_of(sliced, subspaces);
                log::info("disentanglement: converged in {} iterations, Omega_I = {:.9f} bohr^2",
                          iteration,
                          final_spread);
                return {std::move(subspaces), final_spread, iteration};
            }
        }

        throw WannierDiagnosticError(
            WannierDiagnostic::disentanglement_unconverged,
            "disentanglement did not converge in " + std::to_string(settings_.maximum_iterations) +
                " iterations. Last gauge-invariant spreads (bohr^2): " +
                window.recent_history_text() +
                ". A subspace that has not converged is not returned as if it had.",
            window.recent_history());
    }

  private:
    /**
     * Z(k) = sum_b w_b [M(k,b) V(k+b)] [M(k,b) V(k+b)]^dagger over the kept
     * bands, the matrix whose leading eigenvectors are the next subspace at
     * `kpoint`; and, added to `spread` on the way, this k-point's part of
     * Omega_I for the subspaces Z is built against.
     *
     * M(k,b) V(k+b) is what both are built from: Z accumulates its outer
     * product, and the spread's positive term is the part of it V(k) keeps.
     * Forming it once is not only the cheaper arrangement, it is the only
     * one in which the two are provably the same quantity -- which is why
     * one function returns the first and accumulates the second.
     */
    [[nodiscard]] DenseMatrix<Complex>
    subspace_matrix_at(int kpoint,
                       const BlochOverlaps& sliced,
                       const std::vector<DenseMatrix<Complex>>& subspaces,
                       Real& spread) const {
        const int kept = classification_[static_cast<std::size_t>(kpoint)].number_kept();
        DenseMatrix<Complex> z(kept, kept, Complex(0.0, 0.0));
        for (int direction = 0; direction < stencil_.number_of_directions(); direction++) {
            const Real weight = stencil_.weight(direction);
            const DenseMatrix<Complex> carried =
                multiply(MatrixOperation::None,
                         sliced.at(kpoint, direction),
                         MatrixOperation::None,
                         subspaces[static_cast<std::size_t>(stencil_.neighbor(kpoint, direction))]);
            multiply_add(MatrixOperation::None,
                         carried,
                         MatrixOperation::ConjugateTranspose,
                         carried,
                         Complex(weight, 0.0),
                         z);
            spread +=
                weight * (static_cast<Real>(number_of_wannier_functions_) -
                          retained_weight(subspaces[static_cast<std::size_t>(kpoint)], carried));
        }
        return z;
    }

    /// damping * fresh + (1 - damping) * previous: undamped, the iteration oscillates on metals.
    [[nodiscard]] DenseMatrix<Complex> mixed(const DenseMatrix<Complex>& previous,
                                             const DenseMatrix<Complex>& fresh) const {
        DenseMatrix<Complex> result = previous.scaled(1.0 - settings_.damping);
        result.add_scaled(settings_.damping, fresh);
        return result;
    }

    /**
     * Omega_I cannot be negative. Its positive term is sum_mn |M~_mn|^2 over
     * an N-dimensional subspace, and for an overlap between two ORTHONORMAL
     * sets of states that sum is at most N, every singular value of M being
     * at most one. A negative value therefore does not mean the optimisation
     * went wrong; it means the overlaps it was given are not overlaps of
     * orthonormal states -- a corrupt record, or matrices assembled by hand
     * for a test without that property. Saying so here is worth more than
     * returning a number whose sign is impossible.
     */
    void require_nonnegative(Real spread) const {
        if (spread < -math::tolerances::wannier_negative_spread_tolerance *
                         static_cast<Real>(number_of_wannier_functions_)) {
            std::ostringstream value;
            value << std::scientific << spread;
            throw WannierDiagnosticError(
                WannierDiagnostic::overlaps_not_orthonormal,
                "disentanglement: the gauge-invariant spread came out negative (" + value.str() +
                    " bohr^2), which no overlap between orthonormal states can produce. Some "
                    "M_mn(k,b) has a singular value above one, so the record's overlaps are "
                    "corrupt or were not built from orthonormal Bloch states.");
        }
    }

    void require_subspace_shapes(const std::vector<DenseMatrix<Complex>>& subspaces) const {
        if (static_cast<int>(subspaces.size()) != stencil_.number_of_kpoints()) {
            throw std::invalid_argument(
                "SubspaceDisentangler: " + std::to_string(subspaces.size()) +
                " initial subspaces were given for " +
                std::to_string(stencil_.number_of_kpoints()) + " k-points");
        }
        for (std::size_t kpoint = 0; kpoint < subspaces.size(); kpoint++) {
            const int kept = classification_[kpoint].number_kept();
            if (subspaces[kpoint].rows() != kept ||
                subspaces[kpoint].columns() != number_of_wannier_functions_) {
                throw std::invalid_argument(
                    "SubspaceDisentangler: the initial subspace at k-point " +
                    std::to_string(kpoint) + " is " + std::to_string(subspaces[kpoint].rows()) +
                    " x " + std::to_string(subspaces[kpoint].columns()) + ", not the " +
                    std::to_string(kept) + " x " + std::to_string(number_of_wannier_functions_) +
                    " its kept bands and the Wannier functions require");
            }
        }
    }

    /// sum_mn |V(k)^dagger M V(k+b)|^2, the part of the carried subspace V(k) keeps.
    [[nodiscard]] static Real retained_weight(const DenseMatrix<Complex>& subspace,
                                              const DenseMatrix<Complex>& carried) {
        return multiply(
                   MatrixOperation::ConjugateTranspose, subspace, MatrixOperation::None, carried)
            .frobenius_norm_squared();
    }

    /**
     * Omega_I of a given set of subspaces, over the whole zone. The iteration
     * gets this for free while it builds Z; this is for the one place that
     * needs it for subspaces Z was not built from.
     */
    [[nodiscard]] Real spread_of(const BlochOverlaps& sliced,
                                 const std::vector<DenseMatrix<Complex>>& subspaces) const {
        Real spread = 0.0;
        for (int kpoint = 0; kpoint < stencil_.number_of_kpoints(); kpoint++) {
            for (int direction = 0; direction < stencil_.number_of_directions(); direction++) {
                const int neighbor = stencil_.neighbor(kpoint, direction);
                const DenseMatrix<Complex> carried =
                    multiply(MatrixOperation::None,
                             sliced.at(kpoint, direction),
                             MatrixOperation::None,
                             subspaces[static_cast<std::size_t>(neighbor)]);
                spread += stencil_.weight(direction) *
                          (static_cast<Real>(number_of_wannier_functions_) -
                           retained_weight(subspaces[static_cast<std::size_t>(kpoint)], carried));
            }
        }
        return spread / static_cast<Real>(stencil_.number_of_kpoints());
    }

    /**
     * The frozen states, plus the leading eigenvectors of Z restricted to an
     * explicit orthonormal basis of their complement.
     */
    [[nodiscard]] DenseMatrix<Complex>
    choose_subspace(const DenseMatrix<Complex>& z, const KPointBandClassification& entry) const {
        const int kept = entry.number_kept();
        const int frozen = entry.number_frozen();
        DenseMatrix<Complex> subspace(kept, number_of_wannier_functions_, Complex(0.0, 0.0));

        for (int index = 0; index < frozen; index++) {
            subspace(entry.frozen_positions[static_cast<std::size_t>(index)], index) =
                Complex(1.0, 0.0);
        }
        const int wanted = number_of_wannier_functions_ - frozen;
        if (wanted == 0) {
            return subspace;
        }

        /**
         * The frozen states are unit vectors of the working basis, so their
         * complement is spanned by the remaining unit vectors. No
         * orthogonalisation is needed, and the restricted matrix is built by
         * indexing rather than by projecting -- which is the point: the
         * frozen directions never appear in the eigenproblem at all.
         */
        std::vector<int> complement;
        complement.reserve(static_cast<std::size_t>(kept - frozen));
        std::vector<bool> is_frozen(static_cast<std::size_t>(kept), false);
        for (const int position : entry.frozen_positions) {
            is_frozen[static_cast<std::size_t>(position)] = true;
        }
        for (int position = 0; position < kept; position++) {
            if (!is_frozen[static_cast<std::size_t>(position)]) {
                complement.push_back(position);
            }
        }

        const HermitianEigenDecomposition decomposition =
            HermitianEigensolver("SubspaceDisentangler: the subspace matrix Z(k)")
                .solve(z.submatrix(complement, complement));

        /// zheev returns ascending eigenvalues; the largest are the last.
        const int dimension = static_cast<int>(complement.size());
        for (int index = 0; index < wanted; index++) {
            const int source = dimension - 1 - index;
            const int target = frozen + index;
            for (int row = 0; row < dimension; row++) {
                subspace(complement[static_cast<std::size_t>(row)], target) =
                    decomposition.eigenvectors(row, source);
            }
        }
        return subspace;
    }

    const KPointStencil& stencil_;
    const std::vector<KPointBandClassification>& classification_;
    int number_of_wannier_functions_;
    DisentanglementSettings settings_;
};

} // namespace q3
