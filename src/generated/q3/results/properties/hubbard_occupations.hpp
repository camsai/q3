#pragma once

// @generated from results/properties/hubbard_occupations.json — DO NOT EDIT

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 */
struct ResultsPropertiesHubbardOccupationsItem {
    // 0-based, matching the orbital_projections labels. The log prints it 1-based.
    int64_t atom_index{};
    // One entry per spin component.
    std::vector<double> occupation_trace;
};

inline void to_json(nlohmann::json& json_output, const ResultsPropertiesHubbardOccupationsItem& data) {
    json_output = nlohmann::json::object();
    json_output["atom_index"] = data.atom_index;
    json_output["occupation_trace"] = data.occupation_trace;
}

inline void from_json(const nlohmann::json& json_input, ResultsPropertiesHubbardOccupationsItem& data) {
    json_input.at("atom_index").get_to(data.atom_index);
    json_input.at("occupation_trace").get_to(data.occupation_trace);
}

} // namespace q3::generated
