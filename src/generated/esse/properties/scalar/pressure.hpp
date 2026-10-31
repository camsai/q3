#pragma once

// @generated from properties_directory/scalar/pressure.json — DO NOT EDIT


#include <nlohmann/json.hpp>

namespace q3::generated {

enum class PressureNameEnum {
    PRESSURE
};

inline void to_json(nlohmann::json& json_output, const PressureNameEnum& enum_value) {
    switch (enum_value) {
        case PressureNameEnum::PRESSURE: json_output = "pressure"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, PressureNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "pressure") {
        enum_value = PressureNameEnum::PRESSURE;
    } else {
        throw std::runtime_error("Unknown PressureNameEnum: " + string_value);
    }
}

enum class PressureUnitsEnum {
    KBAR,
    PA
};

inline void to_json(nlohmann::json& json_output, const PressureUnitsEnum& enum_value) {
    switch (enum_value) {
        case PressureUnitsEnum::KBAR: json_output = "kbar"; break;
        case PressureUnitsEnum::PA: json_output = "pa"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, PressureUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kbar") {
        enum_value = PressureUnitsEnum::KBAR;
    } else if (string_value == "pa") {
        enum_value = PressureUnitsEnum::PA;
    } else {
        throw std::runtime_error("Unknown PressureUnitsEnum: " + string_value);
    }
}

/**
 * average pressure in unit cell
 *
 * Schema: properties_directory/scalar/pressure.json
 */
struct Pressure {
    PressureNameEnum name;
    PressureUnitsEnum units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const Pressure& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["units"] = data.units;
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, Pressure& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
    json_input.at("value").get_to(data.value);
}

} // namespace q3::generated
