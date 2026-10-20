#pragma once

/**
 * NoPreconditioner — the identity, used when preconditioning is disabled.
 *
 * Exists so that "no preconditioning" is an object like any other rather than
 * an empty-container sentinel the mixers have to test for. Every mixer holds a
 * preconditioner and calls it unconditionally.
 */

#include <string>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/preconditioner/init.hpp"

namespace q3 {

/// Identity preconditioner: leaves the residual untouched.
class NoPreconditioner : public Preconditioner {
  public:
    [[nodiscard]] std::string name() const override {
        return "none";
    }

    void apply(std::vector<Complex>& /*residual_g*/) const override {}
};

} // namespace q3
