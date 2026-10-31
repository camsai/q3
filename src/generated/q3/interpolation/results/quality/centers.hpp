#pragma once

// @generated from interpolation/results/quality/centers.json — DO NOT EDIT

#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * The Wannier centres against the atoms of the manifest's material, over periodic images; present when a spread was evaluated and the manifest names atoms.
 */
struct InterpolationResultsQualityCenters {
    // Per function, the distance to the nearest atom.
    std::vector<double> nearest_atom_distance_bohr;
    // Per function, the distance to the nearest midpoint of a bond.
    std::vector<double> nearest_bond_midpoint_distance_bohr;
    // The shortest interatomic distance, which defines a bond here as anything within 20 percent of it.
    double shortest_bond_bohr{};
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsQualityCenters& data) {
    json_output = nlohmann::json::object();
    json_output["nearest_atom_distance_bohr"] = data.nearest_atom_distance_bohr;
    json_output["nearest_bond_midpoint_distance_bohr"] = data.nearest_bond_midpoint_distance_bohr;
    json_output["shortest_bond_bohr"] = data.shortest_bond_bohr;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsQualityCenters& data) {
    json_input.at("nearest_atom_distance_bohr").get_to(data.nearest_atom_distance_bohr);
    json_input.at("nearest_bond_midpoint_distance_bohr").get_to(data.nearest_bond_midpoint_distance_bohr);
    json_input.at("shortest_bond_bohr").get_to(data.shortest_bond_bohr);
}

} // namespace q3::generated
