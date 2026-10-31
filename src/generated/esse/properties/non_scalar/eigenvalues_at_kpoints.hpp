#pragma once

// @generated from properties_directory/non_scalar/eigenvalues_at_kpoints.json — DO NOT EDIT

#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class EigenvaluesAtKpointsNameEnum {
    EIGENVALUES_AT_KPOINTS
};

inline void to_json(nlohmann::json& json_output, const EigenvaluesAtKpointsNameEnum& enum_value) {
    switch (enum_value) {
        case EigenvaluesAtKpointsNameEnum::EIGENVALUES_AT_KPOINTS: json_output = "eigenvalues_at_kpoints"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, EigenvaluesAtKpointsNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "eigenvalues_at_kpoints") {
        enum_value = EigenvaluesAtKpointsNameEnum::EIGENVALUES_AT_KPOINTS;
    } else {
        throw std::runtime_error("Unknown EigenvaluesAtKpointsNameEnum: " + string_value);
    }
}

enum class EigenvaluesAtKpointsUnitsEnum {
    KCAL_MOL,
    KJ_MOL,
    EV,
    J_MOL,
    HARTREE,
    CM_1,
    RY,
    EV_ATOM
};

inline void to_json(nlohmann::json& json_output, const EigenvaluesAtKpointsUnitsEnum& enum_value) {
    switch (enum_value) {
        case EigenvaluesAtKpointsUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case EigenvaluesAtKpointsUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case EigenvaluesAtKpointsUnitsEnum::EV: json_output = "eV"; break;
        case EigenvaluesAtKpointsUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case EigenvaluesAtKpointsUnitsEnum::HARTREE: json_output = "hartree"; break;
        case EigenvaluesAtKpointsUnitsEnum::CM_1: json_output = "cm-1"; break;
        case EigenvaluesAtKpointsUnitsEnum::RY: json_output = "Ry"; break;
        case EigenvaluesAtKpointsUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, EigenvaluesAtKpointsUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = EigenvaluesAtKpointsUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = EigenvaluesAtKpointsUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = EigenvaluesAtKpointsUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = EigenvaluesAtKpointsUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = EigenvaluesAtKpointsUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = EigenvaluesAtKpointsUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = EigenvaluesAtKpointsUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = EigenvaluesAtKpointsUnitsEnum::EV_ATOM;
    } else {
        throw std::runtime_error("Unknown EigenvaluesAtKpointsUnitsEnum: " + string_value);
    }
}

/**
 * the band energies and occupations of one spin channel at one k-point
 */
struct EigenvaluesAtKpointsBandEnergies {
    // spin of the channel
    std::optional<double> spin;
    // band energies, in band order
    std::vector<double> energies;
    // band occupations, in the same order as energies
    std::optional<std::vector<double>> occupations;
};

inline void to_json(nlohmann::json& json_output, const EigenvaluesAtKpointsBandEnergies& data) {
    json_output = nlohmann::json::object();
    if (data.spin) {
        json_output["spin"] = *data.spin;
    }
    json_output["energies"] = data.energies;
    if (data.occupations) {
        json_output["occupations"] = *data.occupations;
    }
}

inline void from_json(const nlohmann::json& json_input, EigenvaluesAtKpointsBandEnergies& data) {
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
struct EigenvaluesAtKpointsEigenvaluesAtKPoint {
    // A k-point is a point in reciprocal space of a crystal.
    std::optional<std::vector<double>> kpoint;
    // integration weight of the k-point
    std::optional<double> weight;
    // one entry per spin channel; a spin-unpolarized calculation has a single entry
    std::vector<EigenvaluesAtKpointsBandEnergies> eigenvalues;
};

inline void to_json(nlohmann::json& json_output, const EigenvaluesAtKpointsEigenvaluesAtKPoint& data) {
    json_output = nlohmann::json::object();
    if (data.kpoint) {
        json_output["kpoint"] = *data.kpoint;
    }
    if (data.weight) {
        json_output["weight"] = *data.weight;
    }
    json_output["eigenvalues"] = data.eigenvalues;
}

inline void from_json(const nlohmann::json& json_input, EigenvaluesAtKpointsEigenvaluesAtKPoint& data) {
    if (json_input.contains("kpoint")) {
        data.kpoint = json_input.at("kpoint").get<std::vector<double>>();
    }
    if (json_input.contains("weight")) {
        data.weight = json_input.at("weight").get<double>();
    }
    json_input.at("eigenvalues").get_to(data.eigenvalues);
}

/**
 * the band energies (and occupations) at each k-point of a calculation, in one energy unit
 *
 * Schema: properties_directory/non_scalar/eigenvalues_at_kpoints.json
 */
struct EigenvaluesAtKpoints {
    EigenvaluesAtKpointsNameEnum name;
    EigenvaluesAtKpointsUnitsEnum units;
    std::vector<EigenvaluesAtKpointsEigenvaluesAtKPoint> values;
};

inline void to_json(nlohmann::json& json_output, const EigenvaluesAtKpoints& data) {
    json_output = nlohmann::json::object();
    json_output["name"] = data.name;
    json_output["units"] = data.units;
    json_output["values"] = data.values;
}

inline void from_json(const nlohmann::json& json_input, EigenvaluesAtKpoints& data) {
    json_input.at("name").get_to(data.name);
    json_input.at("units").get_to(data.units);
    json_input.at("values").get_to(data.values);
}

} // namespace q3::generated
