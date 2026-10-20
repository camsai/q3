#pragma once

/**
 * SpreadFunctional — the Marzari-Vanderbilt spread Omega, its three parts, the
 * Wannier centres, and the analytic gradient with respect to the gauge.
 *
 * The projection gauge fixes a subspace and an orientation inside it in one
 * shot, and that orientation is arbitrary. What decides whether the model
 * interpolates is how localised the resulting Wannier functions are, because
 * the interpolation is a Fourier sum over lattice vectors: functions that decay
 * quickly give an H(R) that decays quickly, and a truncated sum that is
 * accurate between the mesh points as well as on them. The one-shot gauge is
 * exact ON the mesh whatever it does -- which is why a model can look perfect
 * at every k-point it was built from and be hundreds of meV wrong halfway
 * between two of them.
 *
 * Omega splits into three pieces (PRB 56, 12847, section IV):
 *
 *     Omega_I  = (1/N_k) sum_kb w_b [ N - sum_mn |M_mn|^2 ]
 *     Omega_OD = (1/N_k) sum_kb w_b sum_{m != n} |M_mn|^2
 *     Omega_D  = (1/N_k) sum_kb w_b sum_n [ Im ln M_nn + b . r_n ]^2
 *
 * with M here always the overlap AFTER the current gauge rotation. Omega_I is
 * invariant under any k-dependent unitary mixing of the chosen states, which is
 * exactly why the disentangler minimises it separately and why it is the one
 * number that must not move while this class works. Only Omega_D + Omega_OD is
 * minimisable here, and minimising it is what "maximally localised" means.
 *
 * Centres come from the same diagonal phases,
 *
 *     r_n = -(1/N_k) sum_kb w_b b Im ln M_nn,
 *
 * which is a finite-difference Berry phase: it is why the b-vectors and their
 * weights have to be a complete shell set, and why the centres come out in
 * BOHR when the b-vectors are in inverse bohr. Nothing here is in Angstrom.
 *
 * The gradient is MV equation 52. Its two halves pull in different directions:
 * A[R] drives the off-diagonal overlap weight into the diagonal, S[T] lines the
 * diagonal phases up with the centres they imply. Both are needed; either alone
 * has fixed points that are not minima of Omega.
 *
 * GUIDING CENTRES. Im ln M_nn is a principal value: std::arg returns it on
 * (-pi, pi], so it JUMPS by 2pi as the gauge rotates a diagonal element across
 * the negative real axis. Omega_D squares that phase and is discontinuous
 * wherever it happens, and no descent direction exists across a jump -- the
 * line search rejects every step and the minimisation stops with a large
 * gradient. On a rattled four-atom supercell 8% of the diagonals sat within
 * 0.64 rad of the cut and the minimiser moved the spread by 0.6% before giving
 * up.
 *
 * Given a guiding centre per Wannier function, each phase is instead taken on
 * the branch nearest the one that centre implies:
 *
 *     phase = Im ln M_nn + 2 pi m,   m = round[ (-b . r_guide - Im ln M_nn) / 2pi ]
 *
 * which is continuous as the gauge rotates through the cut, because m steps by
 * one exactly when Im ln steps by 2pi the other way.
 *
 * What matters is WHERE the shift is applied. Folding the residual
 * (Im ln M_nn + b . r_n) onto (-pi, pi] is the obvious thing and it is wrong:
 * MV's gradient relies on sum_kb w_b b q_n = 0, which holds because r_n is
 * built from the very phases q_n is measured against, and folding q_n breaks
 * it. Measured, that costs the gradient a third of its value -- a central
 * difference of 246.6 against a predicted 332.4. Shifting the PHASE keeps the
 * identity, because the shifted phase is what the centre is then computed from,
 * so the cancellation survives and equation 52 stays exact.
 *
 * The functional is an object over the stencil it sums along and the branch
 * its phases are read on (PhaseBranch), because both are fixed for the whole
 * of a minimisation: a functional whose branch choice moved with the gauge
 * would be a different functional at every step.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/principal_phase_branch.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_terms.hpp"

namespace q3 {

class SpreadFunctional {
  public:
    /**
     * `stencil` must carry b-vectors: the centres are Berry phases along them.
     * `phase_branch` is the branch every diagonal phase is read on; the
     * principal one unless another is given.
     */
    SpreadFunctional(const KPointStencil& stencil,
                     int number_of_wannier_functions,
                     std::shared_ptr<const PhaseBranch> phase_branch =
                         std::make_shared<const PrincipalPhaseBranch>())
        : stencil_(stencil)
        , number_of_wannier_functions_(number_of_wannier_functions)
        , phase_branch_(std::move(phase_branch)) {
        if (!stencil.has_b_vectors()) {
            throw std::invalid_argument("SpreadFunctional: the stencil carries no b-vectors, and "
                                        "the spread is a sum over them");
        }
        if (number_of_wannier_functions <= 0) {
            throw std::invalid_argument(
                "SpreadFunctional: number_of_wannier_functions must be positive");
        }
        if (phase_branch_ == nullptr) {
            throw std::invalid_argument("SpreadFunctional: no phase branch was given");
        }
        phase_branch_->require_serves(number_of_wannier_functions);
    }

    [[nodiscard]] int number_of_wannier_functions() const {
        return number_of_wannier_functions_;
    }

    /// `rotated_overlaps`: N x N at every (k, b), between the Wannier-gauge states at k and k+b.
    [[nodiscard]] SpreadTerms evaluate(const BlochOverlaps& rotated_overlaps) const {
        require_shape(rotated_overlaps);
        const int number_of_kpoints = stencil_.number_of_kpoints();
        const int dimension = number_of_wannier_functions_;
        const Real inverse_kpoint_count = 1.0 / static_cast<Real>(number_of_kpoints);

        SpreadTerms terms;
        terms.centers_bohr.assign(static_cast<std::size_t>(dimension), Vector3D{0.0, 0.0, 0.0});
        /// <r^2>_n, accumulated with the centres and finished into Omega_n below.
        std::vector<Real> second_moments(static_cast<std::size_t>(dimension), 0.0);

        /**
         * The centres have to be complete before Omega_D can be evaluated at
         * all -- its summand measures each diagonal phase against the centre
         * every k-point agreed on -- so the diagonal phases are swept twice.
         */
        for (int kpoint = 0; kpoint < number_of_kpoints; kpoint++) {
            for (int direction = 0; direction < stencil_.number_of_directions(); direction++) {
                const DenseMatrix<Complex>& overlap = rotated_overlaps.at(kpoint, direction);
                const Vector3D& b_vector = stencil_.b_vector(direction);
                const Real weight = stencil_.weight(direction);
                for (int wannier = 0; wannier < dimension; wannier++) {
                    const Real phase = diagonal_phase(overlap, wannier, b_vector);
                    for (int axis = 0; axis < 3; axis++) {
                        terms.centers_bohr[static_cast<std::size_t>(wannier)][axis] -=
                            inverse_kpoint_count * weight * b_vector[axis] * phase;
                    }
                    second_moments[static_cast<std::size_t>(wannier)] +=
                        inverse_kpoint_count * weight *
                        ((1.0 - std::norm(overlap(wannier, wannier))) + phase * phase);
                }
            }
        }
        terms.spreads_bohr_squared.resize(static_cast<std::size_t>(dimension));
        for (int wannier = 0; wannier < dimension; wannier++) {
            const Vector3D& center = terms.centers_bohr[static_cast<std::size_t>(wannier)];
            terms.spreads_bohr_squared[static_cast<std::size_t>(wannier)] =
                second_moments[static_cast<std::size_t>(wannier)] - center.dot(center);
        }

        for (int kpoint = 0; kpoint < number_of_kpoints; kpoint++) {
            for (int direction = 0; direction < stencil_.number_of_directions(); direction++) {
                const DenseMatrix<Complex>& overlap = rotated_overlaps.at(kpoint, direction);
                const Vector3D& b_vector = stencil_.b_vector(direction);
                const Real weight = stencil_.weight(direction) * inverse_kpoint_count;

                const Real total_square_modulus = overlap.frobenius_norm_squared();
                Real diagonal_square_modulus = 0.0;
                for (int wannier = 0; wannier < dimension; wannier++) {
                    diagonal_square_modulus += std::norm(overlap(wannier, wannier));
                }
                terms.gauge_invariant_bohr_squared +=
                    weight * (static_cast<Real>(dimension) - total_square_modulus);
                terms.off_diagonal_bohr_squared +=
                    weight * (total_square_modulus - diagonal_square_modulus);

                for (int wannier = 0; wannier < dimension; wannier++) {
                    const Real residual =
                        diagonal_phase(overlap, wannier, b_vector) +
                        b_vector.dot(terms.centers_bohr[static_cast<std::size_t>(wannier)]);
                    terms.diagonal_bohr_squared += weight * residual * residual;
                }
            }
        }
        return terms;
    }

    /**
     * dOmega/dW at every k-point, N x N and anti-Hermitian, in the convention
     * where a step along -gradient lowers Omega by the gradient's square norm
     * per unit step. `terms` must be the evaluation of these same overlaps:
     * the gradient depends on the centres, and centres from a different gauge
     * silently give a gradient of a different function.
     */
    [[nodiscard]] std::vector<DenseMatrix<Complex>> gradient(const BlochOverlaps& rotated_overlaps,
                                                             const SpreadTerms& terms) const {
        require_shape(rotated_overlaps);
        const int number_of_kpoints = stencil_.number_of_kpoints();
        const int dimension = number_of_wannier_functions_;
        std::vector<DenseMatrix<Complex>> gradients(
            static_cast<std::size_t>(number_of_kpoints),
            DenseMatrix<Complex>(dimension, dimension, Complex(0.0, 0.0)));

        /**
         * Omega carries a 1/N_k and so does its derivative, and the sign is
         * fixed by the rotation this gradient is meant to be used with:
         * U(k) -> U(k) exp(t D). Both were established against a central
         * difference of Omega itself rather than read off the paper, because
         * the paper's W is not necessarily this code's W -- a convention that
         * differs by a sign gives a minimiser that climbs, and one that differs
         * by N_k gives a first step N_k times too long, which backtracking
         * hides by quietly halving it back.
         */
        const Real scale = -1.0 / static_cast<Real>(number_of_kpoints);

        for (int kpoint = 0; kpoint < number_of_kpoints; kpoint++) {
            DenseMatrix<Complex>& gradient_at_kpoint = gradients[static_cast<std::size_t>(kpoint)];
            for (int direction = 0; direction < stencil_.number_of_directions(); direction++) {
                const DenseMatrix<Complex>& overlap = rotated_overlaps.at(kpoint, direction);
                const Vector3D& b_vector = stencil_.b_vector(direction);
                const Real weight = stencil_.weight(direction);

                /**
                 * R_mn = M_mn conj(M_nn) and T_mn = (M_mn / M_nn) q_n, with
                 * q_n the diagonal phase measured against the centre. Both
                 * columns are scaled by a property of the DIAGONAL element of
                 * that column, which is what makes them column operations.
                 */
                for (int column = 0; column < dimension; column++) {
                    const Complex diagonal = overlap(column, column);
                    require_nonzero_diagonal(diagonal, column, kpoint);

                    const Real residual =
                        diagonal_phase(overlap, column, b_vector) +
                        b_vector.dot(terms.centers_bohr[static_cast<std::size_t>(column)]);
                    const Complex column_scale = Complex(0.0, residual) / diagonal;

                    for (int row = 0; row < dimension; row++) {
                        const Complex element = overlap(row, column);
                        /**
                         * A[R] - S[T], assembled directly into the gradient:
                         * A[B] = (B - B^dagger)/2 contributes B/2 here and
                         * -conj(B)/2 at the transposed index, and
                         * S[B] = (B + B^dagger)/(2i) contributes B/(2i) and
                         * +conj(B)/(2i). Writing both halves as we go avoids
                         * materialising R and T at all.
                         */
                        const Complex r_element = element * std::conj(diagonal);
                        const Complex t_element = element * column_scale;
                        const Complex contribution =
                            scale * 4.0 * weight * (0.5 * r_element + 0.5 * t_element);
                        gradient_at_kpoint(row, column) += contribution;
                        gradient_at_kpoint(column, row) -= std::conj(contribution);
                    }
                }
            }
        }
        return gradients;
    }

    /// sum_k ||G^(k)||_F^2, the square norm the descent step is measured in.
    [[nodiscard]] static Real square_norm(const std::vector<DenseMatrix<Complex>>& gradients) {
        Real total = 0.0;
        for (const DenseMatrix<Complex>& gradient_at_kpoint : gradients) {
            total += gradient_at_kpoint.frobenius_norm_squared();
        }
        return total;
    }

  private:
    /// The diagonal phase of Wannier function `wannier`, on this functional's branch.
    [[nodiscard]] Real diagonal_phase(const DenseMatrix<Complex>& overlap,
                                      int wannier,
                                      const Vector3D& b_vector) const {
        return phase_branch_->phase_of(overlap(wannier, wannier), wannier, b_vector);
    }

    /**
     * A vanishing diagonal overlap is not a small number to be guarded with an
     * epsilon: Im ln M_nn is the Berry phase this whole construction is built
     * on, and where M_nn is zero that phase does not exist. It means the gauge
     * has collapsed -- two Wannier functions have become the same state, or the
     * mesh is too coarse for the manifold -- and continuing produces centres
     * and a spread that are noise wearing the shape of an answer.
     */
    static void require_nonzero_diagonal(const Complex& diagonal, int wannier, int kpoint) {
        if (std::abs(diagonal) < math::tolerances::wannier_diagonal_overlap_threshold) {
            throw WannierDiagnosticError(
                WannierDiagnostic::gauge_collapsed,
                "SpreadFunctional: the overlap of Wannier function " + std::to_string(wannier) +
                    " with itself across a neighbouring k-point is zero at k-point " +
                    std::to_string(kpoint) +
                    ", so its Berry phase -- and with it the centre and the spread -- is "
                    "undefined. "
                    "The gauge has collapsed: either two Wannier functions have become the same "
                    "state, or the k-mesh is too coarse to follow them.");
        }
    }

    void require_shape(const BlochOverlaps& overlaps) const {
        if (overlaps.number_of_kpoints() != stencil_.number_of_kpoints() ||
            overlaps.number_of_directions() != stencil_.number_of_directions()) {
            throw std::invalid_argument(
                "SpreadFunctional: the overlaps cover " +
                std::to_string(overlaps.number_of_kpoints()) + " k-points and " +
                std::to_string(overlaps.number_of_directions()) + " directions, the stencil " +
                std::to_string(stencil_.number_of_kpoints()) + " and " +
                std::to_string(stencil_.number_of_directions()));
        }
        for (int kpoint = 0; kpoint < overlaps.number_of_kpoints(); kpoint++) {
            for (int direction = 0; direction < overlaps.number_of_directions(); direction++) {
                const DenseMatrix<Complex>& block = overlaps.at(kpoint, direction);
                if (block.rows() != number_of_wannier_functions_ ||
                    block.columns() != number_of_wannier_functions_) {
                    throw std::invalid_argument(
                        "SpreadFunctional: the overlap block at k-point " + std::to_string(kpoint) +
                        " is " + std::to_string(block.rows()) + " x " +
                        std::to_string(block.columns()) + ", not the square " +
                        std::to_string(number_of_wannier_functions_) +
                        "-state block a gauge of that many Wannier functions gives");
                }
            }
        }
    }

    const KPointStencil& stencil_;
    int number_of_wannier_functions_;
    std::shared_ptr<const PhaseBranch> phase_branch_;
};

} // namespace q3
