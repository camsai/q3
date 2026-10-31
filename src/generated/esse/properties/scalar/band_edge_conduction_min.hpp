#pragma once

// @generated from properties_directory/scalar/band_edge_conduction_min.json — DO NOT EDIT

#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class BandEdgeConductionMinNameEnum {
    BAND_EDGE_CONDUCTION_MIN
};

inline void to_json(nlohmann::json& json_output, const BandEdgeConductionMinNameEnum& enum_value) {
    switch (enum_value) {
        case BandEdgeConductionMinNameEnum::BAND_EDGE_CONDUCTION_MIN: json_output = "band_edge_conduction_min"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, BandEdgeConductionMinNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "band_edge_conduction_min") {
        enum_value = BandEdgeConductionMinNameEnum::BAND_EDGE_CONDUCTION_MIN;
    } else {
        throw std::runtime_error("Unknown BandEdgeConductionMinNameEnum: " + string_value);
    }
}

enum class BandEdgeConductionMinUnitsEnum {
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

inline void to_json(nlohmann::json& json_output, const BandEdgeConductionMinUnitsEnum& enum_value) {
    switch (enum_value) {
        case BandEdgeConductionMinUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case BandEdgeConductionMinUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case BandEdgeConductionMinUnitsEnum::EV: json_output = "eV"; break;
        case BandEdgeConductionMinUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case BandEdgeConductionMinUnitsEnum::HARTREE: json_output = "hartree"; break;
        case BandEdgeConductionMinUnitsEnum::CM_1: json_output = "cm-1"; break;
        case BandEdgeConductionMinUnitsEnum::RY: json_output = "Ry"; break;
        case BandEdgeConductionMinUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
        case BandEdgeConductionMinUnitsEnum::EV_A_2: json_output = "eV/A^2"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, BandEdgeConductionMinUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = BandEdgeConductionMinUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = BandEdgeConductionMinUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = BandEdgeConductionMinUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = BandEdgeConductionMinUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = BandEdgeConductionMinUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = BandEdgeConductionMinUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = BandEdgeConductionMinUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = BandEdgeConductionMinUnitsEnum::EV_ATOM;
    } else if (string_value == "eV/A^2") {
        enum_value = BandEdgeConductionMinUnitsEnum::EV_A_2;
    } else {
        throw std::runtime_error("Unknown BandEdgeConductionMinUnitsEnum: " + string_value);
    }
}

/**
 * lowest unoccupied band energy of a crystal
 *
 * Schema: properties_directory/scalar/band_edge_conduction_min.json
 */
struct BandEdgeConductionMin {
    BandEdgeConductionMinNameEnum name;
    // A k-point is a point in reciprocal space of a crystal.
    std::optional<std::vector<double>> kpoint;
    // spin of the channel the band edge belongs to
    std::optional<double> spin;
    BandEdgeConductionMinUnitsEnum units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const BandEdgeConductionMin& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    if (data.kpoint) {
        json_output["kpoint"] = *data.kpoint;
    }
    if (data.spin) {
        json_output["spin"] = *data.spin;
    }
    json_output["units"] = data.units;
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, BandEdgeConductionMin& data) {
    json_input.at("name").get_to(data.name);
    if (json_input.contains("kpoint")) {
        data.kpoint = json_input.at("kpoint").get<std::vector<double>>();
    }
    if (json_input.contains("spin")) {
        data.spin = json_input.at("spin").get<double>();
    }
    json_input.at("units").get_to(data.units);
    json_input.at("value").get_to(data.value);
}

} // namespace q3::generated
