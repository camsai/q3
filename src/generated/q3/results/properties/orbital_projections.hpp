#pragma once

// @generated from results/properties/orbital_projections.json — DO NOT EDIT

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 */
struct ResultsOrbitalProjectionsLabelsItem {
    int64_t atom_index{};
    int64_t species_type_index{};
    std::string shell_label;
    int64_t angular_momentum{};
    int64_t magnetic_quantum_number{};
};

inline void to_json(nlohmann::json& json_output, const ResultsOrbitalProjectionsLabelsItem& data) {
    json_output = nlohmann::json::object();
    json_output["atom_index"] = data.atom_index;
    json_output["species_type_index"] = data.species_type_index;
    json_output["shell_label"] = data.shell_label;
    json_output["angular_momentum"] = data.angular_momentum;
    json_output["magnetic_quantum_number"] = data.magnetic_quantum_number;
}

inline void from_json(const nlohmann::json& json_input, ResultsOrbitalProjectionsLabelsItem& data) {
    json_input.at("atom_index").get_to(data.atom_index);
    json_input.at("species_type_index").get_to(data.species_type_index);
    json_input.at("shell_label").get_to(data.shell_label);
    json_input.at("angular_momentum").get_to(data.angular_momentum);
    json_input.at("magnetic_quantum_number").get_to(data.magnetic_quantum_number);
}

/**
 * Present only when a non-self-consistent run was asked for projections.
 */
struct ResultsOrbitalProjections {
    // |<phi'_i|psi_nk>|^2, indexed [k-point state][band][orbital]. Present only when a non-self-consistent run was asked for projections.
    std::vector<std::vector<std::vector<double>>> values;
    // What each orbital index of orbital_projections is, in the same order.
    std::vector<ResultsOrbitalProjectionsLabelsItem> labels;
};

inline void to_json(nlohmann::json& json_output, const ResultsOrbitalProjections& data) {
    json_output = nlohmann::json::object();
    json_output["values"] = data.values;
    json_output["labels"] = data.labels;
}

inline void from_json(const nlohmann::json& json_input, ResultsOrbitalProjections& data) {
    json_input.at("values").get_to(data.values);
    json_input.at("labels").get_to(data.labels);
}

} // namespace q3::generated
