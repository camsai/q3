#pragma once

/**
 * create_gauge_choice — the policy and the record decide the strategy, here
 * and nowhere else.
 *
 *   None       the projection gauge.
 *   Preferred  localise when the record carries the neighbour overlaps, and
 *              keep the projection gauge, saying so, when it does not.
 *   Required   localise, and refuse the record if it cannot support that.
 *
 * A mesh with a two-point axis is refused under both localising policies:
 * such a record carries overlaps, so it asked for something its mesh cannot
 * give, and a silent downgrade there would hide a mesh that needs fixing.
 * The refusal is raised here, before the subspace is chosen, so an
 * impossible request costs no iteration.
 */

#include <memory>

#include "layer_0_core/logging/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/maximally_localized_gauge_choice.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/projection_gauge_choice.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"

namespace q3 {

[[nodiscard]] inline std::unique_ptr<GaugeChoice>
create_gauge_choice(const WannierConstructionSettings& settings,
                    const WannierConstructionInputs& inputs) {
    if (!settings.localizes()) {
        return std::make_unique<ProjectionGaugeChoice>();
    }
    if (!settings.demands_localization() && !inputs.has_overlaps()) {
        log::info("interpolate: the record carries no neighbour overlaps, so the projection gauge "
                  "is used as it stands; ask for control.interpolation.localization: maximal to "
                  "make that an error instead");
        return std::make_unique<ProjectionGaugeChoice>();
    }
    MaximallyLocalizedGaugeChoice::require_supported(inputs);
    return std::make_unique<MaximallyLocalizedGaugeChoice>(settings.spread_minimization);
}

} // namespace q3
