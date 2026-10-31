#pragma once

// @generated from properties_directory/scalar/band_edge_valence_max.json — DO NOT EDIT

#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class BandEdgeValenceMaxNameEnum {
    BAND_EDGE_VALENCE_MAX
};

inline void to_json(nlohmann::json& json_output, const BandEdgeValenceMaxNameEnum& enum_value) {
    switch (enum_value) {
        case BandEdgeValenceMaxNameEnum::BAND_EDGE_VALENCE_MAX: json_output = "band_edge_valence_max"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, BandEdgeValenceMaxNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "band_edge_valence_max") {
        enum_value = BandEdgeValenceMaxNameEnum::BAND_EDGE_VALENCE_MAX;
    } else {
        throw std::runtime_error("Unknown BandEdgeValenceMaxNameEnum: " + string_value);
    }
}

enum class BandEdgeValenceMaxUnitsEnum {
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

inline void to_json(nlohmann::json& json_output, const BandEdgeValenceMaxUnitsEnum& enum_value) {
    switch (enum_value) {
        case BandEdgeValenceMaxUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case BandEdgeValenceMaxUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case BandEdgeValenceMaxUnitsEnum::EV: json_output = "eV"; break;
        case BandEdgeValenceMaxUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case BandEdgeValenceMaxUnitsEnum::HARTREE: json_output = "hartree"; break;
        case BandEdgeValenceMaxUnitsEnum::CM_1: json_output = "cm-1"; break;
        case BandEdgeValenceMaxUnitsEnum::RY: json_output = "Ry"; break;
        case BandEdgeValenceMaxUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
        case BandEdgeValenceMaxUnitsEnum::EV_A_2: json_output = "eV/A^2"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, BandEdgeValenceMaxUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = BandEdgeValenceMaxUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = BandEdgeValenceMaxUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = BandEdgeValenceMaxUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = BandEdgeValenceMaxUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = BandEdgeValenceMaxUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = BandEdgeValenceMaxUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = BandEdgeValenceMaxUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = BandEdgeValenceMaxUnitsEnum::EV_ATOM;
    } else if (string_value == "eV/A^2") {
        enum_value = BandEdgeValenceMaxUnitsEnum::EV_A_2;
    } else {
        throw std::runtime_error("Unknown BandEdgeValenceMaxUnitsEnum: " + string_value);
    }
}

/**
 * highest occupied band energy of a crystal
 *
 * Schema: properties_directory/scalar/band_edge_valence_max.json
 */
struct BandEdgeValenceMax {
    BandEdgeValenceMaxNameEnum name;
    // A k-point is a point in reciprocal space of a crystal.
    std::optional<std::vector<double>> kpoint;
    // spin of the channel the band edge belongs to
    std::optional<double> spin;
    BandEdgeValenceMaxUnitsEnum units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const BandEdgeValenceMax& data) {
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

inline void from_json(const nlohmann::json& json_input, BandEdgeValenceMax& data) {
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
