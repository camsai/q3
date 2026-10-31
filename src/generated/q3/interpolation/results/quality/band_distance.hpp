#pragma once

// @generated from interpolation/results/quality/band_distance.json — DO NOT EDIT

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * Added by q3.workflows.wannier: the band distance of this model against a direct calculation on the reference k-points.
 */
struct InterpolationResultsBandDistance {
    // The band distance eta, in electronvolts.
    double eta_ev{};
    // The largest pointwise band difference, in electronvolts.
    double eta_maximum_ev{};
    // How many (k-point, band) pairs entered the distance.
    int64_t number_of_states{};
    // What the reference k-points were: "path" or "half-shifted mesh".
    std::string reference;
    // How many reference k-points were compared.
    int64_t number_of_kpoints{};
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsBandDistance& data) {
    json_output = nlohmann::json::object();
    json_output["eta_ev"] = data.eta_ev;
    json_output["eta_maximum_ev"] = data.eta_maximum_ev;
    json_output["number_of_states"] = data.number_of_states;
    json_output["reference"] = data.reference;
    json_output["number_of_kpoints"] = data.number_of_kpoints;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsBandDistance& data) {
    json_input.at("eta_ev").get_to(data.eta_ev);
    json_input.at("eta_maximum_ev").get_to(data.eta_maximum_ev);
    json_input.at("number_of_states").get_to(data.number_of_states);
    json_input.at("reference").get_to(data.reference);
    json_input.at("number_of_kpoints").get_to(data.number_of_kpoints);
}

} // namespace q3::generated
