#pragma once

/**
 * FixedOccupation — step-function occupation for insulating systems.
 *
 * All states below the Fermi energy are fully occupied (f=1), all above
 * are empty (f=0). No smearing entropy contribution.
 */

#include "layer_4_computation/iterator/internal/smearing/init.hpp"

namespace q3 {

/// Fixed occupations (insulators): step function.
class FixedOccupation : public SmearingScheme {
  public:
    FixedOccupation() : SmearingScheme(0.0) {}

    [[nodiscard]] Real occupation(Real x) const override {
        return x < 0.0 ? 1.0 : 0.0;
    }
    [[nodiscard]] Real entropy_weight(Real /* unused */) const override {
        return 0.0;
    }
    [[nodiscard]] std::string name() const override {
        return "fixed";
    }
};

} // namespace q3
