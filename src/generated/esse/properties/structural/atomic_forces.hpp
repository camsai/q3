#pragma once

// @generated from properties_directory/structural/atomic_forces.json — DO NOT EDIT

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class AtomicForcesNameEnum {
    ATOMIC_FORCES
};

inline void to_json(nlohmann::json& json_output, const AtomicForcesNameEnum& enum_value) {
    switch (enum_value) {
        case AtomicForcesNameEnum::ATOMIC_FORCES: json_output = "atomic_forces"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, AtomicForcesNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "atomic_forces") {
        enum_value = AtomicForcesNameEnum::ATOMIC_FORCES;
    } else {
        throw std::runtime_error("Unknown AtomicForcesNameEnum: " + string_value);
    }
}

enum class AtomicForcesUnitsEnum {
    EV_BOHR,
    EV_ANGSTROM,
    RY_A_U,
    NEWTON,
    KG_M_S_2,
    EV_A_U
};

inline void to_json(nlohmann::json& json_output, const AtomicForcesUnitsEnum& enum_value) {
    switch (enum_value) {
        case AtomicForcesUnitsEnum::EV_BOHR: json_output = "eV/bohr"; break;
        case AtomicForcesUnitsEnum::EV_ANGSTROM: json_output = "eV/angstrom"; break;
        case AtomicForcesUnitsEnum::RY_A_U: json_output = "Ry/a.u."; break;
        case AtomicForcesUnitsEnum::NEWTON: json_output = "newton"; break;
        case AtomicForcesUnitsEnum::KG_M_S_2: json_output = "kg*m/s^2"; break;
        case AtomicForcesUnitsEnum::EV_A_U: json_output = "eV/a.u."; break;
    }
}

inline void from_json(const nlohmann::json& json_input, AtomicForcesUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "eV/bohr") {
        enum_value = AtomicForcesUnitsEnum::EV_BOHR;
    } else if (string_value == "eV/angstrom") {
        enum_value = AtomicForcesUnitsEnum::EV_ANGSTROM;
    } else if (string_value == "Ry/a.u.") {
        enum_value = AtomicForcesUnitsEnum::RY_A_U;
    } else if (string_value == "newton") {
        enum_value = AtomicForcesUnitsEnum::NEWTON;
    } else if (string_value == "kg*m/s^2") {
        enum_value = AtomicForcesUnitsEnum::KG_M_S_2;
    } else if (string_value == "eV/a.u.") {
        enum_value = AtomicForcesUnitsEnum::EV_A_U;
    } else {
        throw std::runtime_error("Unknown AtomicForcesUnitsEnum: " + string_value);
    }
}

/**
 * object containing integer id and value each
 */
struct AtomicForcesAtomicVector {
    // value of this entry
    std::vector<double> value;
    // integer id of this entry
    int64_t id{};
};

inline void to_json(nlohmann::json& json_output, const AtomicForcesAtomicVector& data) {
    json_output = nlohmann::json::object();
    json_output["value"] = data.value;
    json_output["id"] = data.id;
}

inline void from_json(const nlohmann::json& json_input, AtomicForcesAtomicVector& data) {
    json_input.at("value").get_to(data.value);
    json_input.at("id").get_to(data.id);
}

/**
 * coordinates of atoms by ids, vector, unitless
 *
 * Schema: properties_directory/structural/atomic_forces.json
 */
struct AtomicForces {
    AtomicForcesNameEnum name;
    std::vector<AtomicForcesAtomicVector> values;
    AtomicForcesUnitsEnum units;
};

inline void to_json(nlohmann::json& json_output, const AtomicForces& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["values"] = data.values;
    json_output["units"] = data.units;
}

inline void from_json(const nlohmann::json& json_input, AtomicForces& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("values").get_to(data.values);
    json_input.at("units").get_to(data.units);
}

} // namespace q3::generated
