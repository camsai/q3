#pragma once

/**
 * GaussianSmearing — Gaussian broadening for metallic occupation functions.
 *
 * Implements f(x) = erfc(x)/2 smearing and its entropy contribution,
 * where x = (ε - ε_F) / σ.
 */

#include <cmath>

#include "layer_4_computation/iterator/internal/smearing/init.hpp"

namespace q3 {

/// Gaussian smearing.
class GaussianSmearing : public SmearingScheme {
  public:
    explicit GaussianSmearing(Real smearing_width = 0.0) : SmearingScheme(smearing_width) {}

    [[nodiscard]] Real occupation(Real x) const override {
        return 0.5 * std::erfc(x);
    }

    [[nodiscard]] Real entropy_weight(Real x) const override {
        const Real sqrtpi_inv = 1.0 / std::sqrt(constants::pi);
        return -0.5 * sqrtpi_inv * std::exp(-x * x);
    }

    [[nodiscard]] std::string name() const override {
        return "gaussian";
    }
};

} // namespace q3
