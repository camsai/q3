#pragma once

/**
 * MarzariVanderbiltSmearing — "cold smearing" occupation function.
 *
 * Implements the cold-smearing scheme of Marzari, Vanderbilt, De Vita &
 * Payne, Phys. Rev. Lett. 82, 3296 (1999). The two closed forms below follow
 * by elementary integration of that paper's smearing delta: the occupation
 * is the integral of delta, the entropy weight the integral of t*delta.
 * Cold smearing is a sibling of, not a member of, the Hermite expansion of
 * Methfessel & Paxton, Phys. Rev. B 40, 3616 (1989) — that expansion is
 * indexed by order N >= 0 and has no N = -1 term; "n = -1" is only the
 * dispatch index some codes use to select this scheme. Cross-checked
 * numerically against Quantum ESPRESSO, whose
 * internal smearing variable is x_QE = -x (x = (E-E_F)/sigma, the convention
 * used throughout this codebase). That sign relation is confirmed by the
 * existing Gaussian and Fermi-Dirac schemes, which already match QE exactly
 * under the same substitution. Letting y = x + 1/sqrt(2):
 *
 *   occupation(x)     = (1/2) erfc(y) + exp(-y^2) / sqrt(2*pi)
 *   entropy_weight(x)  = -y * exp(-y^2) / sqrt(2*pi)
 *
 * Unlike plain Methfessel-Paxton, this is not symmetric about x=0 and its
 * occupation numbers are not confined to [0, 1] either. The overshoot is a
 * single smooth lobe, and both its size and its location follow from the
 * closed form above rather than from any boundedness assumption: setting
 * d/dy [(1/2) erfc(y) + exp(-y^2)/sqrt(2*pi)] = 0 gives y = -1/sqrt(2),
 * i.e. x = -sqrt(2) = -1.41421, where occupation(x) = 1.0833155 — an 8.33%
 * overshoot, well away from the Fermi level. The QE comparison (Cu FCC,
 * matched degauss) corroborates the scheme end to end; it is not what
 * establishes these two numbers.
 */

#include <cmath>

#include "layer_4_computation/iterator/internal/smearing/init.hpp"

namespace q3 {

/// Marzari-Vanderbilt ("cold") smearing.
class MarzariVanderbiltSmearing : public SmearingScheme {
  public:
    explicit MarzariVanderbiltSmearing(Real smearing_width = 0.0)
        : SmearingScheme(smearing_width) {}

    [[nodiscard]] Real occupation(Real x) const override {
        const Real inverse_sqrt2 = 1.0 / std::sqrt(2.0);
        Real y = x + inverse_sqrt2;
        return 0.5 * std::erfc(y) + std::exp(-y * y) / std::sqrt(2.0 * constants::pi);
    }

    [[nodiscard]] Real entropy_weight(Real x) const override {
        const Real inverse_sqrt2 = 1.0 / std::sqrt(2.0);
        Real y = x + inverse_sqrt2;
        return -y * std::exp(-y * y) / std::sqrt(2.0 * constants::pi);
    }

    [[nodiscard]] std::string name() const override {
        return "marzari-vanderbilt";
    }
};

} // namespace q3
