#pragma once

/**
 * Preconditioner — abstract base class for SCF residual preconditioners.
 *
 * A preconditioner reshapes the density residual before it enters the mixing
 * update, damping the long-wavelength charge-sloshing modes that make the
 * self-consistent iteration converge slowly (or not at all) in metals and in
 * large or inhomogeneous cells.
 *
 * The interface is an operator rather than a weight array because not every
 * useful preconditioner is diagonal in G. Kerker is (one multiply per G), but
 * a preconditioner built from a spatially varying screening length is diagonal
 * in neither representation and has to be applied by solving a linear system.
 * Concrete implementations live one per file in this directory.
 */

#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Abstract base class for all SCF residual preconditioners.
class Preconditioner {
  public:
    virtual ~Preconditioner() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * Apply the preconditioner to one G-space residual channel, in place.
     *
     * Callers pass the TOTAL charge channel only: the magnetization channel has
     * no Coulomb kernel to screen, and damping its long-wavelength components
     * would freeze slow moment relaxation. The G = 0 component must be left
     * untouched — it is fixed by charge conservation, and the mixing metric
     * excludes it for the same reason.
     *
     * @param residual_g  residual in reciprocal space; overwritten with the
     *                    preconditioned residual.
     */
    virtual void apply(std::vector<Complex>& residual_g) const = 0;

    /**
     * Supply the current real-space input density, summed over spin channels,
     * before the next apply().
     *
     * Density-independent preconditioners (identity, Kerker) ignore it. One
     * built from the local density recomputes its screening from it, which is
     * why this is a separate call: the mixer knows when the density is current,
     * and apply() may run more than once per SCF step.
     */
    virtual void update(const std::vector<Real>& /*density_real*/) {}
};

} // namespace q3
