#pragma once

/**
 * ProjectabilityDisentanglementSettings — projectability disentanglement's knobs: the
 * classifier's thresholds and the subspace iteration's budget.
 */

#include "layer_4_computation/interpolator/method/wannier/settings/band_classification_thresholds.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/disentanglement_settings.hpp"

namespace q3 {

struct [[nodiscard]] ProjectabilityDisentanglementSettings {
    BandClassificationThresholds thresholds;
    DisentanglementSettings iteration;
};

} // namespace q3
