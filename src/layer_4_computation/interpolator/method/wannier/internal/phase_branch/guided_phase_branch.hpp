#pragma once

/**
 * GuidedPhaseBranch — each diagonal phase on the branch nearest the one a
 * guiding centre implies:
 *
 *     phase = Im ln M_nn + 2 pi m,   m = round[ (-b . r_guide - Im ln M_nn) / 2pi ]
 *
 * which is continuous as the gauge rotates a diagonal element through the
 * negative real axis, because m steps by one exactly when Im ln steps by
 * 2 pi the other way. The centres are fixed when this is built and never
 * move: a branch that followed the gauge would make the functional a
 * different one at every step of a minimisation.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/init.hpp"

namespace q3 {

class GuidedPhaseBranch : public PhaseBranch {
  public:
    /// `centers_bohr`: one per Wannier function, Cartesian, in the frame and unit of the b-vectors.
    explicit GuidedPhaseBranch(std::vector<Vector3D> centers_bohr)
        : centers_bohr_(std::move(centers_bohr)) {
        if (centers_bohr_.empty()) {
            throw std::invalid_argument("GuidedPhaseBranch: no guiding centres were given");
        }
    }

    void require_serves(int number_of_wannier_functions) const override {
        if (static_cast<int>(centers_bohr_.size()) != number_of_wannier_functions) {
            throw std::invalid_argument(
                "GuidedPhaseBranch: " + std::to_string(centers_bohr_.size()) +
                " guiding centres were given for " + std::to_string(number_of_wannier_functions) +
                " Wannier functions");
        }
    }

    [[nodiscard]] Real phase_of(const Complex& diagonal_overlap,
                                int wannier_function,
                                const Vector3D& b_vector) const override {
        const Real phase = std::arg(diagonal_overlap);
        const Real expected =
            -b_vector.dot(centers_bohr_[static_cast<std::size_t>(wannier_function)]);
        return phase + constants::two_pi * std::round((expected - phase) / constants::two_pi);
    }

  private:
    std::vector<Vector3D> centers_bohr_;
};

} // namespace q3
