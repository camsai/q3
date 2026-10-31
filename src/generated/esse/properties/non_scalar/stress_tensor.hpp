#pragma once

// @generated from properties_directory/non_scalar/stress_tensor.json — DO NOT EDIT

#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class StressTensorNameEnum {
    STRESS_TENSOR
};

inline void to_json(nlohmann::json& json_output, const StressTensorNameEnum& enum_value) {
    switch (enum_value) {
        case StressTensorNameEnum::STRESS_TENSOR: json_output = "stress_tensor"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, StressTensorNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "stress_tensor") {
        enum_value = StressTensorNameEnum::STRESS_TENSOR;
    } else {
        throw std::runtime_error("Unknown StressTensorNameEnum: " + string_value);
    }
}

enum class StressTensorUnitsEnum {
    KBAR,
    PA
};

inline void to_json(nlohmann::json& json_output, const StressTensorUnitsEnum& enum_value) {
    switch (enum_value) {
        case StressTensorUnitsEnum::KBAR: json_output = "kbar"; break;
        case StressTensorUnitsEnum::PA: json_output = "pa"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, StressTensorUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kbar") {
        enum_value = StressTensorUnitsEnum::KBAR;
    } else if (string_value == "pa") {
        enum_value = StressTensorUnitsEnum::PA;
    } else {
        throw std::runtime_error("Unknown StressTensorUnitsEnum: " + string_value);
    }
}

/**
 *
 * Schema: properties_directory/non_scalar/stress_tensor.json
 */
struct StressTensor {
    std::vector<std::vector<double>> value;
    StressTensorNameEnum name;
    StressTensorUnitsEnum units;
};

inline void to_json(nlohmann::json& json_output, const StressTensor& data) {
    json_output = nlohmann::json::object();
    json_output["value"] = data.value;
    json_output["name"] = data.name;
    json_output["units"] = data.units;
}

inline void from_json(const nlohmann::json& json_input, StressTensor& data) {
    json_input.at("value").get_to(data.value);
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
}

} // namespace q3::generated
