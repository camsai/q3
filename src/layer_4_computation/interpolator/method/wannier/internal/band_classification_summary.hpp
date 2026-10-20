#pragma once

/**
 * BandClassificationSummary — what the classifier decided about the system
 * as a whole, for the report: metal or not, the window it applied, and the
 * bands the singular-gauge rule kept out of the freeze.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_excluded_from_freeze.hpp"

namespace q3 {

struct [[nodiscard]] BandClassificationSummary {
    /// Whether some band straddles the Fermi level across the mesh.
    bool metallic = false;
    int number_of_bands_crossing_fermi = 0;
    /// The window actually applied, in rydberg, after any automatic resolution.
    Real frozen_window_above_fermi_rydberg = 0.0;
    bool frozen_window_automatic = false;
    /// Ascending by band.
    std::vector<BandExcludedFromFreeze> bands_excluded_from_freeze;
};

} // namespace q3
