#pragma once

/**
 * Partial wave from a PAW dataset.
 *
 * One channel of the PAW basis: either an all-electron partial wave φ_i(r) or
 * its pseudized counterpart φ̃_i(r). Stored r-weighted (r·φ(r)) on the radial
 * grid, the same convention the Kleinman-Bylander projectors use, so that
 * products of two partial waves already carry the r² of the volume element.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// A single PAW partial wave, tabulated as r·φ(r) on the radial grid.
struct PartialWave {
    int angular_momentum{};
    std::vector<Real> radial_values; // r * phi(r)
};

} // namespace q3
