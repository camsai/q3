#pragma once

/**
 * DisentanglementReport — what the classifier decided and what the subspace iteration
 * reached; meaningful only when `performed`.
 */

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification_summary.hpp"

namespace q3 {

struct [[nodiscard]] DisentanglementReport {
    /// Whether the subspace was chosen by classification and iteration rather than taken as given.
    bool performed = false;
    /// The classifier's verdict on the system: metal or not, the window applied, the exclusions.
    BandClassificationSummary classification;
    /// Omega_I of the subspace the iteration settled on, bohr squared.
    Real gauge_invariant_spread_bohr_squared = 0.0;
    int iterations = 0;
};

} // namespace q3
