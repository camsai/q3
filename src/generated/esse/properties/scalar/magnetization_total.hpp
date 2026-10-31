#pragma once

// @generated from properties_directory/scalar/magnetization_total.json — DO NOT EDIT


#include <nlohmann/json.hpp>

namespace q3::generated {

enum class MagnetizationTotalNameEnum {
    MAGNETIZATION_TOTAL
};

inline void to_json(nlohmann::json& json_output, const MagnetizationTotalNameEnum& enum_value) {
    switch (enum_value) {
        case MagnetizationTotalNameEnum::MAGNETIZATION_TOTAL: json_output = "magnetization_total"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, MagnetizationTotalNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "magnetization_total") {
        enum_value = MagnetizationTotalNameEnum::MAGNETIZATION_TOTAL;
    } else {
        throw std::runtime_error("Unknown MagnetizationTotalNameEnum: " + string_value);
    }
}

enum class MagnetizationTotalUnitsEnum {
    UB
};

inline void to_json(nlohmann::json& json_output, const MagnetizationTotalUnitsEnum& enum_value) {
    switch (enum_value) {
        case MagnetizationTotalUnitsEnum::UB: json_output = "uB"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, MagnetizationTotalUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "uB") {
        enum_value = MagnetizationTotalUnitsEnum::UB;
    } else {
        throw std::runtime_error("Unknown MagnetizationTotalUnitsEnum: " + string_value);
    }
}

/**
 * net magnetization of the cell: the integral of the magnetization density
 *
 * Schema: properties_directory/scalar/magnetization_total.json
 */
struct MagnetizationTotal {
    MagnetizationTotalNameEnum name;
    MagnetizationTotalUnitsEnum units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const MagnetizationTotal& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["units"] = data.units;
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, MagnetizationTotal& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
    json_input.at("value").get_to(data.value);
}

} // namespace q3::generated
