#pragma once

/**
 * NullIsolatedSystemCorrectionCalculator — the "no correction" scheme.
 *
 * Null Object implementation of IsolatedSystemCorrectionCalculator: reports
 * an unapplied result with no terms. Selected by
 * create_isolated_system_correction_calculator("none") so the SCF flavor can
 * call the correction interface unconditionally instead of branching on the
 * scheme name.
 */

#include <string>

#include "layer_5_properties/isolated_system_correction/planewave/init.hpp"

namespace q3 {

class NullIsolatedSystemCorrectionCalculator : public IsolatedSystemCorrectionCalculator {
  public:
    [[nodiscard]] IsolatedSystemCorrectionResult
    compute(const NormConservingPlanewaveKohnShamSCFState& /*scf_state*/) const override {
        return {};
    }

    [[nodiscard]] std::string name() const override {
        return "none";
    }
};

} // namespace q3
