#pragma once

// @generated from properties_directory/non_scalar/band_gaps.json — DO NOT EDIT

#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class BandGapsNameEnum {
    BAND_GAPS
};

inline void to_json(nlohmann::json& json_output, const BandGapsNameEnum& enum_value) {
    switch (enum_value) {
        case BandGapsNameEnum::BAND_GAPS: json_output = "band_gaps"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, BandGapsNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "band_gaps") {
        enum_value = BandGapsNameEnum::BAND_GAPS;
    } else {
        throw std::runtime_error("Unknown BandGapsNameEnum: " + string_value);
    }
}

enum class BandGapsBandGapTypeEnum {
    DIRECT,
    INDIRECT
};

inline void to_json(nlohmann::json& json_output, const BandGapsBandGapTypeEnum& enum_value) {
    switch (enum_value) {
        case BandGapsBandGapTypeEnum::DIRECT: json_output = "direct"; break;
        case BandGapsBandGapTypeEnum::INDIRECT: json_output = "indirect"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, BandGapsBandGapTypeEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "direct") {
        enum_value = BandGapsBandGapTypeEnum::DIRECT;
    } else if (string_value == "indirect") {
        enum_value = BandGapsBandGapTypeEnum::INDIRECT;
    } else {
        throw std::runtime_error("Unknown BandGapsBandGapTypeEnum: " + string_value);
    }
}

enum class BandGapsBandGapUnitsEnum {
    KCAL_MOL,
    KJ_MOL,
    EV,
    J_MOL,
    HARTREE,
    CM_1,
    RY,
    EV_ATOM
};

inline void to_json(nlohmann::json& json_output, const BandGapsBandGapUnitsEnum& enum_value) {
    switch (enum_value) {
        case BandGapsBandGapUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case BandGapsBandGapUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case BandGapsBandGapUnitsEnum::EV: json_output = "eV"; break;
        case BandGapsBandGapUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case BandGapsBandGapUnitsEnum::HARTREE: json_output = "hartree"; break;
        case BandGapsBandGapUnitsEnum::CM_1: json_output = "cm-1"; break;
        case BandGapsBandGapUnitsEnum::RY: json_output = "Ry"; break;
        case BandGapsBandGapUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, BandGapsBandGapUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = BandGapsBandGapUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = BandGapsBandGapUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = BandGapsBandGapUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = BandGapsBandGapUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = BandGapsBandGapUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = BandGapsBandGapUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = BandGapsBandGapUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = BandGapsBandGapUnitsEnum::EV_ATOM;
    } else {
        throw std::runtime_error("Unknown BandGapsBandGapUnitsEnum: " + string_value);
    }
}

/**
 */
struct BandGapsBandGap {
    // A k-point is a point in reciprocal space of a crystal.
    std::optional<std::vector<double>> kpoint_conduction;
    // A k-point is a point in reciprocal space of a crystal.
    std::optional<std::vector<double>> kpoint_valence;
    // eigenvalue at k-point in conduction band
    std::optional<double> eigenvalue_conduction;
    // eigenvalue at k-point in valence band
    std::optional<double> eigenvalue_valence;
    std::optional<double> spin;
    BandGapsBandGapTypeEnum type;
    std::optional<BandGapsBandGapUnitsEnum> units;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const BandGapsBandGap& data) {
    json_output = nlohmann::json::object();
    if (data.kpoint_conduction) {
        json_output["kpointConduction"] = *data.kpoint_conduction;
    }
    if (data.kpoint_valence) {
        json_output["kpointValence"] = *data.kpoint_valence;
    }
    if (data.eigenvalue_conduction) {
        json_output["eigenvalueConduction"] = *data.eigenvalue_conduction;
    }
    if (data.eigenvalue_valence) {
        json_output["eigenvalueValence"] = *data.eigenvalue_valence;
    }
    if (data.spin) {
        json_output["spin"] = *data.spin;
    }
    json_output["type"] = data.type;
    if (data.units) {
        json_output["units"] = *data.units;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, BandGapsBandGap& data) {
    if (json_input.contains("kpointConduction")) {
        data.kpoint_conduction = json_input.at("kpointConduction").get<std::vector<double>>();
    }
    if (json_input.contains("kpointValence")) {
        data.kpoint_valence = json_input.at("kpointValence").get<std::vector<double>>();
    }
    if (json_input.contains("eigenvalueConduction")) {
        data.eigenvalue_conduction = json_input.at("eigenvalueConduction").get<double>();
    }
    if (json_input.contains("eigenvalueValence")) {
        data.eigenvalue_valence = json_input.at("eigenvalueValence").get<double>();
    }
    if (json_input.contains("spin")) {
        data.spin = json_input.at("spin").get<double>();
    }
    json_input.at("type").get_to(data.type);
    if (json_input.contains("units")) {
        data.units = json_input.at("units").get<BandGapsBandGapUnitsEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * the band energies and occupations of one spin channel at one k-point
 */
struct BandGapsBandEnergies {
    // spin of the channel
    std::optional<double> spin;
    // band energies, in band order
    std::vector<double> energies;
    // band occupations, in the same order as energies
    std::optional<std::vector<double>> occupations;
};

inline void to_json(nlohmann::json& json_output, const BandGapsBandEnergies& data) {
    json_output = nlohmann::json::object();
    if (data.spin) {
        json_output["spin"] = *data.spin;
    }
    json_output["energies"] = data.energies;
    if (data.occupations) {
        json_output["occupations"] = *data.occupations;
    }
}

inline void from_json(const nlohmann::json& json_input, BandGapsBandEnergies& data) {
    if (json_input.contains("spin")) {
        data.spin = json_input.at("spin").get<double>();
    }
    json_input.at("energies").get_to(data.energies);
    if (json_input.contains("occupations")) {
        data.occupations = json_input.at("occupations").get<std::vector<double>>();
    }
}

/**
 * the eigenvalues at one k-point: its coordinates, its integration weight, and the band energies and occupations of each spin channel
 */
struct BandGapsEigenvaluesAtKPoint {
    // A k-point is a point in reciprocal space of a crystal.
    std::optional<std::vector<double>> kpoint;
    // integration weight of the k-point
    std::optional<double> weight;
    // one entry per spin channel; a spin-unpolarized calculation has a single entry
    std::vector<BandGapsBandEnergies> eigenvalues;
};

inline void to_json(nlohmann::json& json_output, const BandGapsEigenvaluesAtKPoint& data) {
    json_output = nlohmann::json::object();
    if (data.kpoint) {
        json_output["kpoint"] = *data.kpoint;
    }
    if (data.weight) {
        json_output["weight"] = *data.weight;
    }
    json_output["eigenvalues"] = data.eigenvalues;
}

inline void from_json(const nlohmann::json& json_input, BandGapsEigenvaluesAtKPoint& data) {
    if (json_input.contains("kpoint")) {
        data.kpoint = json_input.at("kpoint").get<std::vector<double>>();
    }
    if (json_input.contains("weight")) {
        data.weight = json_input.at("weight").get<double>();
    }
    json_input.at("eigenvalues").get_to(data.eigenvalues);
}

/**
 * contains band gap values
 *
 * Schema: properties_directory/non_scalar/band_gaps.json
 */
struct BandGaps {
    BandGapsNameEnum name;
    std::vector<BandGapsBandGap> values;
    std::optional<std::vector<BandGapsEigenvaluesAtKPoint>> eigenvalues;
};

inline void to_json(nlohmann::json& json_output, const BandGaps& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["values"] = data.values;
    if (data.eigenvalues) {
        json_output["eigenvalues"] = *data.eigenvalues;
    }
}

inline void from_json(const nlohmann::json& json_input, BandGaps& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("values").get_to(data.values);
    if (json_input.contains("eigenvalues")) {
        data.eigenvalues = json_input.at("eigenvalues").get<std::vector<BandGapsEigenvaluesAtKPoint>>();
    }
}

} // namespace q3::generated
