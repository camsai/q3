#pragma once

// @generated from results/properties/energy_contributions.json — DO NOT EDIT

#include <optional>

#include <nlohmann/json.hpp>

#include "generated/esse/core/reusable/energy.hpp"

namespace q3::generated {

/**
 * Energy terms Q3 reports that ESSE's total_energy_contributions has no key for yet, each an ESSE energy in rydberg; the terms ESSE does have are in that record under properties. hubbard (DFT+U, Dudarev), dispersion (already inside total_energy) and one_center (PAW) are present only when nonzero.
 */
struct ResultsEnergyContributions {
    Energy double_counting;
    std::optional<Energy> hubbard;
    std::optional<Energy> dispersion;
    std::optional<Energy> one_center;
};

inline void to_json(nlohmann::json& json_output, const ResultsEnergyContributions& data) {
    json_output = nlohmann::json::object();
    json_output["double_counting"] = data.double_counting;
    if (data.hubbard) {
        json_output["hubbard"] = *data.hubbard;
    }
    if (data.dispersion) {
        json_output["dispersion"] = *data.dispersion;
    }
    if (data.one_center) {
        json_output["one_center"] = *data.one_center;
    }
}

inline void from_json(const nlohmann::json& json_input, ResultsEnergyContributions& data) {
    json_input.at("double_counting").get_to(data.double_counting);
    if (json_input.contains("hubbard")) {
        data.hubbard = json_input.at("hubbard").get<Energy>();
    }
    if (json_input.contains("dispersion")) {
        data.dispersion = json_input.at("dispersion").get<Energy>();
    }
    if (json_input.contains("one_center")) {
        data.one_center = json_input.at("one_center").get<Energy>();
    }
}

} // namespace q3::generated
