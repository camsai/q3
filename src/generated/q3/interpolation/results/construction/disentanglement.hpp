#pragma once

// @generated from interpolation/results/construction/disentanglement.json — DO NOT EDIT

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

#include "generated/q3/interpolation/results/construction/disentanglement/bands_excluded_from_freeze.hpp"

namespace q3::generated {

/**
 * What the classifier decided when the subspace was chosen by projectability disentanglement: the window it resolved and which bands the singular-gauge rule kept out of the freeze. Present only when the model was disentangled.
 */
struct InterpolationResultsDisentanglement {
    // Whether a band crosses the Fermi level, which is what makes the automatic window E_F + 2 eV rather than 0.
    bool metallic{};
    // How many bands cross the Fermi level.
    int64_t bands_crossing_fermi{};
    // The frozen window that was applied, in electronvolts above the Fermi level.
    double frozen_window_above_fermi_ev{};
    // Whether the window was resolved from the record (true) or given in the manifest (false).
    bool frozen_window_automatic{};
    // Bands inside the frozen window whose projectability fell below the minimum at some k-points, and were excluded from the freeze there rather than frozen: the trial orbitals cannot represent them, and freezing them would make the gauge singular.
    std::vector<InterpolationResultsExcludedBand> bands_excluded_from_freeze;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsDisentanglement& data) {
    json_output = nlohmann::json::object();
    json_output["metallic"] = data.metallic;
    json_output["bands_crossing_fermi"] = data.bands_crossing_fermi;
    json_output["frozen_window_above_fermi_ev"] = data.frozen_window_above_fermi_ev;
    json_output["frozen_window_automatic"] = data.frozen_window_automatic;
    json_output["bands_excluded_from_freeze"] = data.bands_excluded_from_freeze;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsDisentanglement& data) {
    json_input.at("metallic").get_to(data.metallic);
    json_input.at("bands_crossing_fermi").get_to(data.bands_crossing_fermi);
    json_input.at("frozen_window_above_fermi_ev").get_to(data.frozen_window_above_fermi_ev);
    json_input.at("frozen_window_automatic").get_to(data.frozen_window_automatic);
    json_input.at("bands_excluded_from_freeze").get_to(data.bands_excluded_from_freeze);
}

} // namespace q3::generated
