#pragma once

/**
 * PrincipalPhaseBranch — Im ln M_nn as std::arg returns it, on (-pi, pi].
 *
 * The branch of a gauge nobody has a centre for yet: it is what gives the
 * projection gauge its centres in the first place. It jumps by 2 pi as a
 * diagonal element crosses the negative real axis, so a minimisation is
 * never run on it (SpreadFunctional says what that costs); a test that wants
 * to exhibit the jump is.
 */

#include <complex>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/init.hpp"

namespace q3 {

class PrincipalPhaseBranch : public PhaseBranch {
  public:
    [[nodiscard]] Real phase_of(const Complex& diagonal_overlap,
                                int /*wannier_function*/,
                                const Vector3D& /*b_vector*/) const override {
        return std::arg(diagonal_overlap);
    }

    /// The principal value needs nothing per function, so it serves any number of them.
    void require_serves(int /*number_of_wannier_functions*/) const override {}
};

} // namespace q3
