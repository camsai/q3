#pragma once

/**
 * DispersionCalculator — abstract interface for dispersion (van der Waals)
 * correction schemes.
 *
 * A dispersion correction is geometry-only: it depends on the crystal
 * structure but not on the electronic state, contributing an additive energy
 * plus its analytical gradients (forces and stress). Concrete schemes
 * (D3(BJ) today; D3 zero-damping, D4, TS, ... in the future) implement this
 * interface and are selected by name through create_dispersion_calculator()
 * (factory.hpp), so orchestrators never branch on the scheme themselves --
 * "none" resolves to NullDispersionCalculator (null.hpp), which contributes
 * exact zeros, removing the need for any if/else at the call sites.
 */

#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Analytical gradients of a dispersion energy correction.
 *
 * `forces` are physical forces (F = -dE/dR, Ry/bohr), one entry per atom.
 * `stress` follows Q3's Nielsen-Martin convention (sigma = -(1/Omega)
 * dE/dstrain, Ry/bohr^3, not yet converted to kbar -- that happens in the
 * stress orchestrator alongside every other contribution).
 */
struct DispersionGradients {
    std::vector<Vector3D> forces;
    Matrix3x3 stress;
};

class DispersionCalculator {
  public:
    virtual ~DispersionCalculator() = default;

    /// Total dispersion energy correction, in Rydberg.
    [[nodiscard]] virtual Real compute_energy() const = 0;

    /// Analytical energy gradients (forces and stress) -- see DispersionGradients.
    [[nodiscard]] virtual DispersionGradients compute_gradients() const = 0;

    /**
     * Scheme name as accepted by the input ("none", "d3bj", ...). Reporting
     * code uses `name() != "none"` to decide whether to print dispersion rows.
     */
    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace q3
