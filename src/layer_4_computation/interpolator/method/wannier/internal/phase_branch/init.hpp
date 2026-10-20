#pragma once

/**
 * PhaseBranch — which branch of Im ln M_nn(k,b) the spread functional reads.
 *
 * The diagonal phase of an overlap is defined up to 2 pi, and Omega_D squares
 * it, so the choice of branch is part of the definition of the functional:
 * SpreadFunctional documents why, and why the choice must be held fixed for
 * the whole of a minimisation. Two choices exist and they are two
 * functionals, so they are two classes here rather than a mode of one --
 * PrincipalPhaseBranch, the value std::arg returns, and GuidedPhaseBranch,
 * the branch a centre known in advance implies.
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

class PhaseBranch {
  public:
    virtual ~PhaseBranch() = default;

    /**
     * The phase of `diagonal_overlap`, M_nn(k,b) of Wannier function
     * `wannier_function` along `b_vector`, on this branch.
     */
    [[nodiscard]] virtual Real phase_of(const Complex& diagonal_overlap,
                                        int wannier_function,
                                        const Vector3D& b_vector) const = 0;

    /// Refuses, with std::invalid_argument, a functional of a size this branch cannot serve.
    virtual void require_serves(int number_of_wannier_functions) const = 0;
};

} // namespace q3
