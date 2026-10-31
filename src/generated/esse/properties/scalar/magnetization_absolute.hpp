#pragma once

// @generated from properties_directory/scalar/magnetization_absolute.json — DO NOT EDIT


#include <nlohmann/json.hpp>

namespace q3::generated {

enum class MagnetizationAbsoluteNameEnum {
    MAGNETIZATION_ABSOLUTE
};

inline void to_json(nlohmann::json& json_output, const MagnetizationAbsoluteNameEnum& enum_value) {
    switch (enum_value) {
        case MagnetizationAbsoluteNameEnum::MAGNETIZATION_ABSOLUTE: json_output = "magnetization_absolute"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, MagnetizationAbsoluteNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "magnetization_absolute") {
        enum_value = MagnetizationAbsoluteNameEnum::MAGNETIZATION_ABSOLUTE;
    } else {
        throw std::runtime_error("Unknown MagnetizationAbsoluteNameEnum: " + string_value);
    }
}

enum class MagnetizationAbsoluteUnitsEnum {
    UB
};

inline void to_json(nlohmann::json& json_output, const MagnetizationAbsoluteUnitsEnum& enum_value) {
    switch (enum_value) {
        case MagnetizationAbsoluteUnitsEnum::UB: json_output = "uB"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, MagnetizationAbsoluteUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "uB") {
        enum_value = MagnetizationAbsoluteUnitsEnum::UB;
    } else {
        throw std::runtime_error("Unknown MagnetizationAbsoluteUnitsEnum: " + string_value);
    }
}

/**
 * integral of the absolute value of the magnetization density over the cell
 *
 * Schema: properties_directory/scalar/magnetization_absolute.json
 */
struct MagnetizationAbsolute {
    MagnetizationAbsoluteNameEnum name;
    MagnetizationAbsoluteUnitsEnum units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const MagnetizationAbsolute& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["units"] = data.units;
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, MagnetizationAbsolute& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
    json_input.at("value").get_to(data.value);
}

} // namespace q3::generated
