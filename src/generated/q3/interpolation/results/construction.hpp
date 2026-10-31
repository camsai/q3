#pragma once

// @generated from interpolation/results/construction.json — DO NOT EDIT

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "generated/q3/interpolation/results/construction/disentanglement.hpp"
#include "generated/q3/interpolation/results/construction/spread.hpp"

namespace q3::generated {

/**
 * How the evaluated model was built and how big it is: the method, the record it was read from, its number of bands, and what the construction measured of it (the spread of its Wannier functions and, when the subspace was disentangled, the classifier's verdict).
 */
struct InterpolationResultsConstruction {
    // The record's method name ("wannier").
    std::string method;
    // How the model was built, in words: the method, the subspace choice, whether and how far the gauge was localised, and the model's size.
    std::string description;
    // The path of the record that was evaluated.
    std::string record;
    // The number of bands the model interpolates: the number of Wannier functions.
    int64_t number_of_bands{};
    // The Wigner-Seitz weight identity: the sum of the Bloch-sum weights over the model's lattice vectors, which equals the number of k-points of the mesh the record was built on.
    double total_bloch_sum_weight{};
    // Any further diagnostic the interpolator reports, by name. The local-orbital interpolator reports none beyond total_bloch_sum_weight, which has its own key because readers and the reference name it.
    std::optional<std::map<std::string, double>> diagnostics;
    // The Marzari-Vanderbilt spread of the model's Wannier functions at the gauge it was built with, in bohr and bohr squared. Omega_I is reported always; the rest whenever the record carries the neighbour overlaps that define it and the mesh has no two-point axis — the projection gauge has a spread too, and a driver comparing two models wants it either way.
    InterpolationResultsSpread spread;
    // What the classifier decided when the subspace was chosen by projectability disentanglement: the window it resolved and which bands the singular-gauge rule kept out of the freeze. Present only when the model was disentangled.
    std::optional<InterpolationResultsDisentanglement> disentanglement;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsConstruction& data) {
    json_output = nlohmann::json::object();
    json_output["method"] = data.method;
    json_output["description"] = data.description;
    json_output["record"] = data.record;
    json_output["number_of_bands"] = data.number_of_bands;
    json_output["total_bloch_sum_weight"] = data.total_bloch_sum_weight;
    if (data.diagnostics) {
        json_output["diagnostics"] = *data.diagnostics;
    }
    json_output["spread"] = data.spread;
    if (data.disentanglement) {
        json_output["disentanglement"] = *data.disentanglement;
    }
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsConstruction& data) {
    json_input.at("method").get_to(data.method);
    json_input.at("description").get_to(data.description);
    json_input.at("record").get_to(data.record);
    json_input.at("number_of_bands").get_to(data.number_of_bands);
    json_input.at("total_bloch_sum_weight").get_to(data.total_bloch_sum_weight);
    if (json_input.contains("diagnostics")) {
        data.diagnostics = json_input.at("diagnostics").get<std::map<std::string, double>>();
    }
    json_input.at("spread").get_to(data.spread);
    if (json_input.contains("disentanglement")) {
        data.disentanglement = json_input.at("disentanglement").get<InterpolationResultsDisentanglement>();
    }
}

} // namespace q3::generated
