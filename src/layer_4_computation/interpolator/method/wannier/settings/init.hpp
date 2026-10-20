#pragma once

/**
 * WannierConstructionSettings — every knob of the Wannier construction, in
 * one place and typed.
 *
 * The evaluation manifest (`control.interpolation`) is mapped onto this once,
 * by from(); the builder reads nothing else. Two decisions are made here and
 * nowhere else:
 * how the subspace is chosen (an isolated manifold as it stands, or
 * projectability disentanglement with its thresholds and iteration) and how
 * the gauge inside it is fixed (the policy below). Each maps to a strategy --
 * subspace/factory.hpp and gauge/factory.hpp -- so the builder never branches
 * on them.
 *
 * The defaults are the record-alone case: an isolated manifold in the
 * projection gauge, which is what a test or an importer wants from a record
 * with no manifest.
 */

#include <optional>

#include "layer_0_core/types/constants.hpp"
#include "layer_4_computation/interpolator/input_data/specification.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/localization_policy.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/projectability_disentanglement_settings.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/spread_minimization_settings.hpp"

namespace q3 {

struct [[nodiscard]] WannierConstructionSettings {
    /**
     * Absent, the band window must already be exactly the N states the gauge
     * is built from, which is what an isolated manifold gives. Present,
     * classification and the subspace iteration choose them out of a wider
     * window, which is what an entangled manifold needs.
     */
    std::optional<ProjectabilityDisentanglementSettings> disentanglement;
    LocalizationPolicy localization = LocalizationPolicy::None;
    /// The spread minimiser's settings, read only when a gauge is localised.
    SpreadMinimizationSettings spread_minimization;

    /**
     * The evaluation manifest's settings: the one place the
     * `control.interpolation` knobs are mapped onto the builder's.
     *
     * Disentanglement and localisation are chosen when the model is
     * EVALUATED, not when the record is captured: the record stores the raw
     * window, its projectabilities and its overlaps, so the same capture
     * serves a run with them and a run without, and the Wannier90 export
     * keeps handing another code the untouched matrices. Whether the record
     * can support what the manifest asks for is the builder's question,
     * answered against the record itself (gauge/factory.hpp).
     */
    [[nodiscard]] static WannierConstructionSettings
    from(const InterpolationSpecification& specification) {
        WannierConstructionSettings settings;
        if (specification.disentangles()) {
            ProjectabilityDisentanglementSettings chosen;
            chosen.thresholds.projectability_minimum = specification.projectability_minimum;
            chosen.thresholds.projectability_maximum = specification.projectability_maximum;
            chosen.thresholds.frozen_window_above_fermi_rydberg =
                specification.frozen_window_above_fermi_ev / constants::RY_TO_EV;
            chosen.thresholds.automatic_frozen_window = specification.frozen_window_is_automatic;
            chosen.iteration.maximum_iterations = specification.disentanglement_maximum_iterations;
            chosen.iteration.convergence_tolerance =
                specification.disentanglement_convergence_tolerance;
            settings.disentanglement = chosen;
        }
        settings.localization = localization_policy_from(specification.localization);
        settings.spread_minimization.maximum_iterations =
            specification.localization_maximum_iterations;
        return settings;
    }

    [[nodiscard]] bool disentangles() const {
        return disentanglement.has_value();
    }
    /// True unless localisation was refused outright.
    [[nodiscard]] bool localizes() const {
        return localization != LocalizationPolicy::None;
    }
    /// True when localisation is demanded, so a record that cannot support it is an error.
    [[nodiscard]] bool demands_localization() const {
        return localization == LocalizationPolicy::Required;
    }
};

} // namespace q3
