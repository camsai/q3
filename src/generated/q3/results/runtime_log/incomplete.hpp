#pragma once

// @generated from results/runtime_log/incomplete.json — DO NOT EDIT

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * Present only when a derivative was left out, each entry saying why. The Python relaxation workflows refuse to run on a result whose forces are incomplete.
 */
struct ResultsIncomplete {
    // Why the forces are incomplete: a force term Q3 cannot yet compute was left out.
    std::optional<std::string> forces;
    // Why the stress is incomplete: a stress term was left out, or the topology skipped the stress entirely, in which case there is no stress_tensor or pressure record.
    std::optional<std::string> stress;
};

inline void to_json(nlohmann::json& json_output, const ResultsIncomplete& data) {
    json_output = nlohmann::json::object();
    if (data.forces) {
        json_output["forces"] = *data.forces;
    }
    if (data.stress) {
        json_output["stress"] = *data.stress;
    }
}

inline void from_json(const nlohmann::json& json_input, ResultsIncomplete& data) {
    if (json_input.contains("forces")) {
        data.forces = json_input.at("forces").get<std::string>();
    }
    if (json_input.contains("stress")) {
        data.stress = json_input.at("stress").get<std::string>();
    }
}

} // namespace q3::generated
