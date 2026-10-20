#pragma once

/**
 * SmearingScheme — abstract interface for electronic occupation smearing.
 *
 * Defines the contract for Gaussian, Methfessel-Paxton, Marzari-Vanderbilt,
 * Fermi-Dirac, and fixed-occupation schemes used in metallic Kohn-Sham
 * calculations to smooth the Fermi surface.
 */

#include <string>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Abstract interface for electronic smearing schemes.
class SmearingScheme {
  public:
    explicit SmearingScheme(Real smearing_width = 0.0) : smearing_width_(smearing_width) {}
    virtual ~SmearingScheme() = default;

    /// Compute the occupation f(x) where x = (E - E_F) / σ.
    [[nodiscard]] virtual Real occupation(Real x) const = 0;

    /// Compute the entropy contribution δ(x) weight.
    [[nodiscard]] virtual Real entropy_weight(Real x) const = 0;

    [[nodiscard]] virtual std::string name() const = 0;

    /// Smearing width σ (degauss) in Ry.
    [[nodiscard]] Real smearing_width() const {
        return smearing_width_;
    }

    /// Whether this scheme uses actual smearing (width > 0).
    [[nodiscard]] bool uses_smearing() const {
        return smearing_width_ > math::tolerances::smearing_width_threshold;
    }

  protected:
    Real smearing_width_;
};

} // namespace q3
