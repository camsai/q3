#pragma once

// @generated from interpolation/results/quality/ladder.json — DO NOT EDIT

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 */
struct InterpolationResultsLadderAttempt {
    // The rung of the ladder that made the attempt: "default", "unconverged", "stalled", "window", "thresholds" or "mesh".
    std::string rung;
    // The k-mesh of the record the attempt evaluated.
    std::vector<int64_t> grid;
    // The control.interpolation keys the attempt changed from the defaults, as it wrote them; for the mesh rung, the spacing it recaptured at (kpoint_spacing_inverse_angstrom).
    nlohmann::json settings;
    // What became of the attempt: "accept" or "retry" for an evaluation the gate judged, "error:<route>" for one that raised (the route is "unconverged", "localization", "infeasible" or "unknown", from the engine's diagnostic code), and "recapture" for the mesh rung, which evaluates nothing.
    std::string status;
    // Why, in words: the gate's verdict or the engine's own error line.
    std::string reason;
    // The band distance of the attempt, in electronvolts; null when the evaluation raised.
    nlohmann::json eta_ev;
    // The largest pointwise band difference of the attempt, in electronvolts; null when the evaluation raised.
    nlohmann::json eta_maximum_ev;
    // Whether the attempt's gauge was localised; null when the evaluation raised.
    nlohmann::json localized;
    // Whether that localisation converged; null when the evaluation raised or did not localise.
    nlohmann::json localization_converged;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsLadderAttempt& data) {
    json_output = nlohmann::json::object();
    json_output["rung"] = data.rung;
    json_output["grid"] = data.grid;
    json_output["settings"] = data.settings;
    json_output["status"] = data.status;
    json_output["reason"] = data.reason;
    json_output["eta_ev"] = data.eta_ev;
    json_output["eta_maximum_ev"] = data.eta_maximum_ev;
    json_output["localized"] = data.localized;
    json_output["localization_converged"] = data.localization_converged;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsLadderAttempt& data) {
    json_input.at("rung").get_to(data.rung);
    json_input.at("grid").get_to(data.grid);
    json_input.at("settings").get_to(data.settings);
    json_input.at("status").get_to(data.status);
    json_input.at("reason").get_to(data.reason);
    json_input.at("eta_ev").get_to(data.eta_ev);
    json_input.at("eta_maximum_ev").get_to(data.eta_maximum_ev);
    json_input.at("localized").get_to(data.localized);
    json_input.at("localization_converged").get_to(data.localization_converged);
}

} // namespace q3::generated
