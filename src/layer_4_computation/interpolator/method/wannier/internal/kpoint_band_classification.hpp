#pragma once

/**
 * KPointBandClassification — one k-point's verdict: the role of every band,
 * and the working basis the subspace iteration sees.
 */

#include <vector>

#include "layer_4_computation/interpolator/method/wannier/internal/band_role.hpp"

namespace q3 {

struct [[nodiscard]] KPointBandClassification {
    std::vector<BandRole> roles;
    /// Band indices that survive, frozen first is NOT assumed; ascending.
    std::vector<int> kept_bands;
    /// Positions within `kept_bands` that are frozen, ascending.
    std::vector<int> frozen_positions;

    [[nodiscard]] int number_kept() const {
        return static_cast<int>(kept_bands.size());
    }
    [[nodiscard]] int number_frozen() const {
        return static_cast<int>(frozen_positions.size());
    }
};

} // namespace q3
