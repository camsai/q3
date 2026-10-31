#pragma once

// @generated from properties_directory/non_scalar/density_of_states.json — DO NOT EDIT

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class DensityOfStatesXAxisLabelEnum {
    ENERGY
};

inline void to_json(nlohmann::json& json_output, const DensityOfStatesXAxisLabelEnum& enum_value) {
    switch (enum_value) {
        case DensityOfStatesXAxisLabelEnum::ENERGY: json_output = "energy"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, DensityOfStatesXAxisLabelEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "energy") {
        enum_value = DensityOfStatesXAxisLabelEnum::ENERGY;
    } else {
        throw std::runtime_error("Unknown DensityOfStatesXAxisLabelEnum: " + string_value);
    }
}

enum class DensityOfStatesXAxisUnitsEnum {
    KCAL_MOL,
    KJ_MOL,
    EV,
    J_MOL,
    HARTREE,
    CM_1,
    RY,
    EV_ATOM
};

inline void to_json(nlohmann::json& json_output, const DensityOfStatesXAxisUnitsEnum& enum_value) {
    switch (enum_value) {
        case DensityOfStatesXAxisUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case DensityOfStatesXAxisUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case DensityOfStatesXAxisUnitsEnum::EV: json_output = "eV"; break;
        case DensityOfStatesXAxisUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case DensityOfStatesXAxisUnitsEnum::HARTREE: json_output = "hartree"; break;
        case DensityOfStatesXAxisUnitsEnum::CM_1: json_output = "cm-1"; break;
        case DensityOfStatesXAxisUnitsEnum::RY: json_output = "Ry"; break;
        case DensityOfStatesXAxisUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, DensityOfStatesXAxisUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = DensityOfStatesXAxisUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = DensityOfStatesXAxisUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = DensityOfStatesXAxisUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = DensityOfStatesXAxisUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = DensityOfStatesXAxisUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = DensityOfStatesXAxisUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = DensityOfStatesXAxisUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = DensityOfStatesXAxisUnitsEnum::EV_ATOM;
    } else {
        throw std::runtime_error("Unknown DensityOfStatesXAxisUnitsEnum: " + string_value);
    }
}

enum class DensityOfStatesYAxisLabelEnum {
    DENSITY_OF_STATES
};

inline void to_json(nlohmann::json& json_output, const DensityOfStatesYAxisLabelEnum& enum_value) {
    switch (enum_value) {
        case DensityOfStatesYAxisLabelEnum::DENSITY_OF_STATES: json_output = "density of states"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, DensityOfStatesYAxisLabelEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "density of states") {
        enum_value = DensityOfStatesYAxisLabelEnum::DENSITY_OF_STATES;
    } else {
        throw std::runtime_error("Unknown DensityOfStatesYAxisLabelEnum: " + string_value);
    }
}

enum class DensityOfStatesYAxisUnitsEnum {
    STATES_UNITCELL
};

inline void to_json(nlohmann::json& json_output, const DensityOfStatesYAxisUnitsEnum& enum_value) {
    switch (enum_value) {
        case DensityOfStatesYAxisUnitsEnum::STATES_UNITCELL: json_output = "states/unitcell"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, DensityOfStatesYAxisUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "states/unitcell") {
        enum_value = DensityOfStatesYAxisUnitsEnum::STATES_UNITCELL;
    } else {
        throw std::runtime_error("Unknown DensityOfStatesYAxisUnitsEnum: " + string_value);
    }
}

enum class DensityOfStatesNameEnum {
    DENSITY_OF_STATES
};

inline void to_json(nlohmann::json& json_output, const DensityOfStatesNameEnum& enum_value) {
    switch (enum_value) {
        case DensityOfStatesNameEnum::DENSITY_OF_STATES: json_output = "density_of_states"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, DensityOfStatesNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "density_of_states") {
        enum_value = DensityOfStatesNameEnum::DENSITY_OF_STATES;
    } else {
        throw std::runtime_error("Unknown DensityOfStatesNameEnum: " + string_value);
    }
}

/**
 */
struct DensityOfStatesXAxis {
    // label of an axis object
    DensityOfStatesXAxisLabelEnum label;
    // units for an axis
    std::optional<DensityOfStatesXAxisUnitsEnum> units;
};

inline void to_json(nlohmann::json& json_output, const DensityOfStatesXAxis& data) {
    json_output = nlohmann::json::object();
    json_output["label"] = data.label;
    if (data.units) {
        json_output["units"] = *data.units;
    }
}

inline void from_json(const nlohmann::json& json_input, DensityOfStatesXAxis& data) {
    json_input.at("label").get_to(data.label);
    if (json_input.contains("units")) {
        data.units = json_input.at("units").get<DensityOfStatesXAxisUnitsEnum>();
    }
}

/**
 */
struct DensityOfStatesYAxis {
    // label of an axis object
    DensityOfStatesYAxisLabelEnum label;
    // units for an axis
    std::optional<DensityOfStatesYAxisUnitsEnum> units;
};

inline void to_json(nlohmann::json& json_output, const DensityOfStatesYAxis& data) {
    json_output = nlohmann::json::object();
    json_output["label"] = data.label;
    if (data.units) {
        json_output["units"] = *data.units;
    }
}

inline void from_json(const nlohmann::json& json_input, DensityOfStatesYAxis& data) {
    json_input.at("label").get_to(data.label);
    if (json_input.contains("units")) {
        data.units = json_input.at("units").get<DensityOfStatesYAxisUnitsEnum>();
    }
}

/**
 */
struct DensityOfStatesLegendItem {
    // chemical element
    std::optional<std::string> element;
    // index inside sub-array of atoms of the same element type
    std::optional<int64_t> index;
    // electronic character and shell of PDOS, such as `1s` or `s`, or `total`
    std::optional<std::string> electronic_state;
    // spin of the electronic state
    std::optional<double> spin;
};

inline void to_json(nlohmann::json& json_output, const DensityOfStatesLegendItem& data) {
    json_output = nlohmann::json::object();
    if (data.element) {
        json_output["element"] = *data.element;
    }
    if (data.index) {
        json_output["index"] = *data.index;
    }
    if (data.electronic_state) {
        json_output["electronicState"] = *data.electronic_state;
    }
    if (data.spin) {
        json_output["spin"] = *data.spin;
    }
}

inline void from_json(const nlohmann::json& json_input, DensityOfStatesLegendItem& data) {
    if (json_input.contains("element")) {
        data.element = json_input.at("element").get<std::string>();
    }
    if (json_input.contains("index")) {
        data.index = json_input.at("index").get<int64_t>();
    }
    if (json_input.contains("electronicState")) {
        data.electronic_state = json_input.at("electronicState").get<std::string>();
    }
    if (json_input.contains("spin")) {
        data.spin = json_input.at("spin").get<double>();
    }
}

/**
 *
 * Schema: properties_directory/non_scalar/density_of_states.json
 */
struct DensityOfStates {
    DensityOfStatesXAxis x_axis;
    DensityOfStatesYAxis y_axis;
    DensityOfStatesNameEnum name;
    std::vector<DensityOfStatesLegendItem> legend;
    // array containing values of x Axis
    std::vector<nlohmann::json> x_data_array;
    std::vector<std::vector<double>> y_data_series;
};

inline void to_json(nlohmann::json& json_output, const DensityOfStates& data) {
    json_output = nlohmann::json::object();
    json_output["xAxis"] = data.x_axis;
    json_output["yAxis"] = data.y_axis;
    json_output["name"] = data.name;
    json_output["legend"] = data.legend;
    json_output["xDataArray"] = data.x_data_array;
    json_output["yDataSeries"] = data.y_data_series;
}

inline void from_json(const nlohmann::json& json_input, DensityOfStates& data) {
    json_input.at("xAxis").get_to(data.x_axis);
    json_input.at("yAxis").get_to(data.y_axis);
    json_input.at("name").get_to(data.name);
    json_input.at("legend").get_to(data.legend);
    json_input.at("xDataArray").get_to(data.x_data_array);
    json_input.at("yDataSeries").get_to(data.y_data_series);
}

} // namespace q3::generated
