#pragma once

/**
 * BandClassificationThresholds — the projectability thresholds and the frozen energy
 * window BandClassifier sorts bands by; the rule itself is documented there.
 */

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

struct [[nodiscard]] BandClassificationThresholds {
    /// p >= this freezes a band regardless of its energy.
    Real projectability_maximum = 0.95;
    /// p < this discards a band, whatever its energy: the singular-gauge rule.
    Real projectability_minimum = 0.01;
    /**
     * Energy above the Fermi level, in RYDBERG, below which a band freezes
     * regardless of its projectability (short of the rule above). Read only
     * when `automatic_frozen_window` is false.
     */
    Real frozen_window_above_fermi_rydberg = 0.0;
    /// Apply the published window, AUTOMATIC_WINDOW_RYDBERG, instead of the value above.
    bool automatic_frozen_window = false;

    /// The published window for atomic projectors, 2 eV above the Fermi level, in rydberg.
    static constexpr Real AUTOMATIC_WINDOW_RYDBERG = 2.0 / constants::RY_TO_EV;
    /**
     * A band counts as crossing the Fermi level when it reaches this far past
     * it on both sides, in rydberg. The margin keeps an insulator whose Fermi
     * level IS its highest occupied eigenvalue from being read as a metal by
     * rounding.
     */
    static constexpr Real CROSSING_MARGIN_RYDBERG = 1.0e-4;
};

} // namespace q3
