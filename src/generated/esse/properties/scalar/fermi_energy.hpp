#pragma once

// @generated from properties_directory/scalar/fermi_energy.json — DO NOT EDIT


#include <nlohmann/json.hpp>

namespace q3::generated {

enum class FermiEnergyNameEnum {
    FERMI_ENERGY
};

inline void to_json(nlohmann::json& json_output, const FermiEnergyNameEnum& enum_value) {
    switch (enum_value) {
        case FermiEnergyNameEnum::FERMI_ENERGY: json_output = "fermi_energy"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, FermiEnergyNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "fermi_energy") {
        enum_value = FermiEnergyNameEnum::FERMI_ENERGY;
    } else {
        throw std::runtime_error("Unknown FermiEnergyNameEnum: " + string_value);
    }
}

enum class FermiEnergyUnitsEnum {
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

inline void to_json(nlohmann::json& json_output, const FermiEnergyUnitsEnum& enum_value) {
    switch (enum_value) {
        case FermiEnergyUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case FermiEnergyUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case FermiEnergyUnitsEnum::EV: json_output = "eV"; break;
        case FermiEnergyUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case FermiEnergyUnitsEnum::HARTREE: json_output = "hartree"; break;
        case FermiEnergyUnitsEnum::CM_1: json_output = "cm-1"; break;
        case FermiEnergyUnitsEnum::RY: json_output = "Ry"; break;
        case FermiEnergyUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
        case FermiEnergyUnitsEnum::EV_A_2: json_output = "eV/A^2"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, FermiEnergyUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = FermiEnergyUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = FermiEnergyUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = FermiEnergyUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = FermiEnergyUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = FermiEnergyUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = FermiEnergyUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = FermiEnergyUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = FermiEnergyUnitsEnum::EV_ATOM;
    } else if (string_value == "eV/A^2") {
        enum_value = FermiEnergyUnitsEnum::EV_A_2;
    } else {
        throw std::runtime_error("Unknown FermiEnergyUnitsEnum: " + string_value);
    }
}

/**
 *
 * Schema: properties_directory/scalar/fermi_energy.json
 */
struct FermiEnergy {
    FermiEnergyNameEnum name;
    FermiEnergyUnitsEnum units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const FermiEnergy& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["units"] = data.units;
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, FermiEnergy& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
    json_input.at("value").get_to(data.value);
}

} // namespace q3::generated
