#pragma once

/**
 * AugmentationChargeTable — abstract interface for augmentation charge
 * Q_ij(q,L) interpolation tables.
 *
 * The planewave-specific implementation (PlanewaveAugmentationChargeTable)
 * provides the concrete Bessel-transform-based tables.
 */

#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

class AugmentationChargeTable {
  public:
    virtual ~AugmentationChargeTable() = default;

    /**
     * Interpolate the L-resolved Q radial function at a given q value.
     */
    [[nodiscard]] virtual Real interpolate_qrad(int species_type_index,
                                                int projector_pair_index,
                                                int angular_momentum,
                                                Real q_value) const = 0;

    /**
     * Interpolate the derivative dQ_rad/dq of the L-resolved Q radial function.
     */
    [[nodiscard]] virtual Real interpolate_qrad_derivative(int species_type_index,
                                                           int projector_pair_index,
                                                           int angular_momentum,
                                                           Real q_value) const = 0;

    /**
     * Interpolate the total (L=0) Q function at a given q value.
     */
    [[nodiscard]] virtual Real
    interpolate_qfunc(int species_type_index, int projector_pair_index, Real q_value) const = 0;

    /**
     * Maximum angular momentum channel in the augmentation tables.
     */
    [[nodiscard]] virtual int maximum_angular_momentum_q() const = 0;
};

} // namespace q3
