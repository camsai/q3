#pragma once

// @generated from results/properties/isolated_system_correction.json — DO NOT EDIT

#include <map>
#include <string>

#include <nlohmann/json.hpp>

#include "generated/esse/core/reusable/energy.hpp"

namespace q3::generated {

/**
 * The charged/isolated-cell correction, when one was applied: the scheme's own labelled terms (Makov-Payne: first_order, second_order; Martyna-Tuckerman: energy), all signed so that corrected_total_energy = total_energy + total.
 */
struct ResultsIsolatedSystemCorrection {
    std::string method;
    std::map<std::string, Energy> terms;
    Energy total;
    Energy corrected_total_energy;
};

inline void to_json(nlohmann::json& json_output, const ResultsIsolatedSystemCorrection& data) {
    json_output = nlohmann::json::object();
    json_output["method"] = data.method;
    json_output["terms"] = data.terms;
    json_output["total"] = data.total;
    json_output["corrected_total_energy"] = data.corrected_total_energy;
}

inline void from_json(const nlohmann::json& json_input, ResultsIsolatedSystemCorrection& data) {
    json_input.at("method").get_to(data.method);
    json_input.at("terms").get_to(data.terms);
    json_input.at("total").get_to(data.total);
    json_input.at("corrected_total_energy").get_to(data.corrected_total_energy);
}

} // namespace q3::generated
