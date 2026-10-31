#pragma once

// @generated from properties_directory/workflow/convergence/electronic.json — DO NOT EDIT

#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class ConvergenceElectronicNameEnum {
    CONVERGENCE_ELECTRONIC
};

inline void to_json(nlohmann::json& json_output, const ConvergenceElectronicNameEnum& enum_value) {
    switch (enum_value) {
        case ConvergenceElectronicNameEnum::CONVERGENCE_ELECTRONIC: json_output = "convergence_electronic"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, ConvergenceElectronicNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "convergence_electronic") {
        enum_value = ConvergenceElectronicNameEnum::CONVERGENCE_ELECTRONIC;
    } else {
        throw std::runtime_error("Unknown ConvergenceElectronicNameEnum: " + string_value);
    }
}

enum class ConvergenceElectronicUnitsEnum {
    EV,
    RY,
    HARTREE
};

inline void to_json(nlohmann::json& json_output, const ConvergenceElectronicUnitsEnum& enum_value) {
    switch (enum_value) {
        case ConvergenceElectronicUnitsEnum::EV: json_output = "eV"; break;
        case ConvergenceElectronicUnitsEnum::RY: json_output = "Ry"; break;
        case ConvergenceElectronicUnitsEnum::HARTREE: json_output = "hartree"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, ConvergenceElectronicUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "eV") {
        enum_value = ConvergenceElectronicUnitsEnum::EV;
    } else if (string_value == "Ry") {
        enum_value = ConvergenceElectronicUnitsEnum::RY;
    } else if (string_value == "hartree") {
        enum_value = ConvergenceElectronicUnitsEnum::HARTREE;
    } else {
        throw std::runtime_error("Unknown ConvergenceElectronicUnitsEnum: " + string_value);
    }
}

/**
 *
 * Schema: properties_directory/workflow/convergence/electronic.json
 */
struct ConvergenceElectronic {
    ConvergenceElectronicNameEnum name;
    ConvergenceElectronicUnitsEnum units;
    std::vector<std::vector<double>> data;
};

inline void to_json(nlohmann::json& json_output, const ConvergenceElectronic& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["units"] = data.units;
    json_output["data"] = data.data;
}

inline void from_json(const nlohmann::json& json_input, ConvergenceElectronic& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
    json_input.at("data").get_to(data.data);
}

} // namespace q3::generated
