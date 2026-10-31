#pragma once

// @generated from results/structure.json — DO NOT EDIT

#include <optional>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * The structure the run used.
 */
struct ResultsStructure {
    // The structure the run used, in the engine's own Cartesian frame: ESSE lattice (vectors in angstrom) and basis (crystal coordinates; basis.constraints true where a Cartesian component may move, as mat3ra-made writes them). q3.workflows starts a relaxation from it.
    std::optional<nlohmann::json> material;
};

inline void to_json(nlohmann::json& json_output, const ResultsStructure& data) {
    json_output = nlohmann::json::object();
    if (data.material) {
        json_output["material"] = *data.material;
    }
}

inline void from_json(const nlohmann::json& json_input, ResultsStructure& data) {
    if (json_input.contains("material")) {
        data.material = json_input.at("material").get<nlohmann::json>();
    }
}

} // namespace q3::generated
