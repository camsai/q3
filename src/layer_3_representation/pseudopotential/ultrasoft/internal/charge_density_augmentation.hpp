#pragma once

/**
 * AugmentationChargeTable — abstract interface for augmentation charge
 * Q_ij(q,L) interpolation tables.
 *
 * The planewave-specific implementation (PlanewaveAugmentationChargeTable)
 * provides the concrete Bessel-transform-based tables.
 *
 * The tables hold the radial transform of Eq. (32) of K. Laasonen,
 * A. Pasquarello, R. Car, C. Lee & D. Vanderbilt, Phys. Rev. B 47, 10142
 * (1993), for the Q_ij^L of their Eq. (27). The augmented density they feed
 * is Eq. (18) of D. Vanderbilt, Phys. Rev. B 41, 7892 (1990).
 */

#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

class AugmentationChargeTable {
  public:
    virtual ~AugmentationChargeTable() = default;

    /**
     * Interpolate Q_ij^L(q), the radial Fourier transform of Q_ij in channel L,
     * at a given q value.
     */
    [[nodiscard]] virtual Real interpolate_augmentation_function(int species_type_index,
                                                                 int projector_pair_index,
                                                                 int angular_momentum,
                                                                 Real q_value) const = 0;

    /**
     * Interpolate the derivative dQ_ij^L/dq of the radial Fourier transform of
     * Q_ij in channel L.
     */
    [[nodiscard]] virtual Real
    interpolate_augmentation_function_derivative(int species_type_index,
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
