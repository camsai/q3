#pragma once

// @generated from properties_directory/scalar/total_energy.json — DO NOT EDIT


#include <nlohmann/json.hpp>

namespace q3::generated {

enum class TotalEnergyNameEnum {
    TOTAL_ENERGY
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyNameEnum::TOTAL_ENERGY: json_output = "total_energy"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "total_energy") {
        enum_value = TotalEnergyNameEnum::TOTAL_ENERGY;
    } else {
        throw std::runtime_error("Unknown TotalEnergyNameEnum: " + string_value);
    }
}

enum class TotalEnergyUnitsEnum {
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

inline void to_json(nlohmann::json& json_output, const TotalEnergyUnitsEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case TotalEnergyUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case TotalEnergyUnitsEnum::EV: json_output = "eV"; break;
        case TotalEnergyUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case TotalEnergyUnitsEnum::HARTREE: json_output = "hartree"; break;
        case TotalEnergyUnitsEnum::CM_1: json_output = "cm-1"; break;
        case TotalEnergyUnitsEnum::RY: json_output = "Ry"; break;
        case TotalEnergyUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
        case TotalEnergyUnitsEnum::EV_A_2: json_output = "eV/A^2"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = TotalEnergyUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = TotalEnergyUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = TotalEnergyUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = TotalEnergyUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = TotalEnergyUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = TotalEnergyUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = TotalEnergyUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = TotalEnergyUnitsEnum::EV_ATOM;
    } else if (string_value == "eV/A^2") {
        enum_value = TotalEnergyUnitsEnum::EV_A_2;
    } else {
        throw std::runtime_error("Unknown TotalEnergyUnitsEnum: " + string_value);
    }
}

/**
 *
 * Schema: properties_directory/scalar/total_energy.json
 */
struct TotalEnergy {
    TotalEnergyNameEnum name;
    TotalEnergyUnitsEnum units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergy& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["units"] = data.units;
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergy& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
    json_input.at("value").get_to(data.value);
}

} // namespace q3::generated
