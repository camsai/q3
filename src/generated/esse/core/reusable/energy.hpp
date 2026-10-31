#pragma once

// @generated from core/reusable/energy.json — DO NOT EDIT

#include <string>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class EnergyUnitsEnum {
    KCAL_MOL,
    KJ_MOL,
    EV,
    J_MOL,
    HARTREE,
    CM_1,
    RY,
    EV_ATOM,
    EV_A_2
};

inline void to_json(nlohmann::json& json_output, const EnergyUnitsEnum& enum_value) {
    switch (enum_value) {
        case EnergyUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case EnergyUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case EnergyUnitsEnum::EV: json_output = "eV"; break;
        case EnergyUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case EnergyUnitsEnum::HARTREE: json_output = "hartree"; break;
        case EnergyUnitsEnum::CM_1: json_output = "cm-1"; break;
        case EnergyUnitsEnum::RY: json_output = "Ry"; break;
        case EnergyUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
        case EnergyUnitsEnum::EV_A_2: json_output = "eV/A^2"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, EnergyUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = EnergyUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = EnergyUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = EnergyUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = EnergyUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = EnergyUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = EnergyUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = EnergyUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = EnergyUnitsEnum::EV_ATOM;
    } else if (string_value == "eV/A^2") {
        enum_value = EnergyUnitsEnum::EV_A_2;
    } else {
        throw std::runtime_error("Unknown EnergyUnitsEnum: " + string_value);
    }
}

/**
 *
 * Schema: core/reusable/energy.json
 */
struct Energy {
    std::string name;
    EnergyUnitsEnum units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const Energy& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["units"] = data.units;
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, Energy& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
    json_input.at("value").get_to(data.value);
}

} // namespace q3::generated
