#pragma once

/**
 * PolakRibiereDirection — the conjugate search direction of a descent over
 * one matrix per k-point: -G conjugated against the direction before it.
 *
 * Steepest descent on the spread functional creeps: a three-function chain
 * on eight k-points takes five hundred iterations to do what conjugation
 * does in a few dozen. The memory a conjugate direction needs -- the last
 * gradient and the last direction -- is this object's state, so the
 * minimiser asks for the next direction and carries nothing itself.
 *
 * `beta` is clamped at zero, which is the standard automatic restart: a
 * negative value means the new gradient has turned far enough that the old
 * direction is no longer useful, and following it anyway is how conjugate
 * gradient goes unstable on a non-quadratic functional. A conjugated
 * direction that points uphill is worse than no conjugation at all, so the
 * steepest-descent direction is returned instead, rather than letting a line
 * search spend its whole budget halving a step that was never going to help.
 */

#include <algorithm>
#include <cstddef>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

class PolakRibiereDirection {
  public:
    /**
     * The direction to search along from a gauge whose gradient is
     * `gradients`. `restart` forgets the previous direction and returns the
     * steepest-descent one, which the caller asks for at intervals because
     * the manifold is curved and a direction carried too far along it stops
     * being conjugate to anything.
     */
    [[nodiscard]] const std::vector<DenseMatrix<Complex>>&
    next(const std::vector<DenseMatrix<Complex>>& gradients, bool restart) {
        direction_ = conjugated(gradients, restart);
        previous_gradients_ = gradients;
        return direction_;
    }

  private:
    [[nodiscard]] std::vector<DenseMatrix<Complex>>
    conjugated(const std::vector<DenseMatrix<Complex>>& gradients, bool restart) const {
        const std::vector<DenseMatrix<Complex>> steepest_descent = negated(gradients);
        if (restart || previous_gradients_.empty() || direction_.empty()) {
            return steepest_descent;
        }

        Real numerator = 0.0;
        Real denominator = 0.0;
        for (std::size_t kpoint = 0; kpoint < gradients.size(); kpoint++) {
            DenseMatrix<Complex> change = gradients[kpoint];
            change.add_scaled(-1.0, previous_gradients_[kpoint]);
            numerator += gradients[kpoint].real_inner_product(change);
            denominator += previous_gradients_[kpoint].frobenius_norm_squared();
        }
        if (denominator <= 0.0) {
            return steepest_descent;
        }
        const Real beta = std::max(0.0, numerator / denominator);
        if (beta == 0.0) {
            return steepest_descent;
        }

        std::vector<DenseMatrix<Complex>> direction = steepest_descent;
        Real slope = 0.0;
        for (std::size_t kpoint = 0; kpoint < gradients.size(); kpoint++) {
            direction[kpoint].add_scaled(beta, direction_[kpoint]);
            slope += gradients[kpoint].real_inner_product(direction[kpoint]);
        }
        return slope >= 0.0 ? steepest_descent : direction;
    }

    /// -G at every k-point: the steepest-descent direction.
    [[nodiscard]] static std::vector<DenseMatrix<Complex>>
    negated(const std::vector<DenseMatrix<Complex>>& gradients) {
        std::vector<DenseMatrix<Complex>> result;
        result.reserve(gradients.size());
        for (const DenseMatrix<Complex>& gradient : gradients) {
            result.push_back(gradient.scaled(-1.0));
        }
        return result;
    }

    std::vector<DenseMatrix<Complex>> previous_gradients_;
    std::vector<DenseMatrix<Complex>> direction_;
};

} // namespace q3
