#pragma once

// @generated from interpolation/results/construction/spread.json — DO NOT EDIT

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * The Marzari-Vanderbilt spread of the model's Wannier functions at the gauge it was built with, in bohr and bohr squared. Omega_I is reported always; the rest whenever the record carries the neighbour overlaps that define it and the mesh has no two-point axis — the projection gauge has a spread too, and a driver comparing two models wants it either way.
 */
struct InterpolationResultsSpread {
    // Omega_I: the part of the spread the gauge cannot change. From the spread when it was evaluated, otherwise from the subspace iteration when there was one; zero for an isolated manifold without overlaps.
    double gauge_invariant_bohr_squared{};
    // Whether the gauge was rotated by the spread minimiser at all.
    bool localized{};
    // Omega = Omega_I + Omega_D + Omega_OD at the final gauge.
    std::optional<double> total_bohr_squared;
    // What fixed the branch of each Berry phase: "trial orbitals" when the record carries their positions, "projection gauge" otherwise.
    std::optional<std::string> guiding_centers;
    // The record's trial-orbital positions, Cartesian bohr, one [x, y, z] per function; empty when it carries none.
    std::optional<std::vector<std::vector<double>>> trial_orbital_centers_bohr;
    // Omega_D + Omega_OD at the projection gauge, before any rotation: what localisation bought is this minus the final value.
    std::optional<double> initial_minimizable_bohr_squared;
    // The Wannier centres at the final gauge, Cartesian bohr, one [x, y, z] per function.
    std::optional<std::vector<std::vector<double>>> centers_bohr;
    // Omega_n per function: <r^2>_n - |<r>_n|^2, summing to the total.
    std::optional<std::vector<double>> spreads_bohr_squared;
    // How many iterations the spread minimiser ran; present when it ran.
    std::optional<int64_t> localization_iterations;
    // Whether the minimiser reached a stationary gauge rather than merely stopped; false after a stall, where the gauge is the best one found but not a minimum.
    std::optional<bool> localization_converged;
    // The norm of the spread gradient where the minimiser stopped, a diagnostic beside the answer rather than the convergence test.
    std::optional<double> localization_gradient_norm;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsSpread& data) {
    json_output = nlohmann::json::object();
    json_output["gauge_invariant_bohr_squared"] = data.gauge_invariant_bohr_squared;
    json_output["localized"] = data.localized;
    if (data.total_bohr_squared) {
        json_output["total_bohr_squared"] = *data.total_bohr_squared;
    }
    if (data.guiding_centers) {
        json_output["guiding_centers"] = *data.guiding_centers;
    }
    if (data.trial_orbital_centers_bohr) {
        json_output["trial_orbital_centers_bohr"] = *data.trial_orbital_centers_bohr;
    }
    if (data.initial_minimizable_bohr_squared) {
        json_output["initial_minimizable_bohr_squared"] = *data.initial_minimizable_bohr_squared;
    }
    if (data.centers_bohr) {
        json_output["centers_bohr"] = *data.centers_bohr;
    }
    if (data.spreads_bohr_squared) {
        json_output["spreads_bohr_squared"] = *data.spreads_bohr_squared;
    }
    if (data.localization_iterations) {
        json_output["localization_iterations"] = *data.localization_iterations;
    }
    if (data.localization_converged) {
        json_output["localization_converged"] = *data.localization_converged;
    }
    if (data.localization_gradient_norm) {
        json_output["localization_gradient_norm"] = *data.localization_gradient_norm;
    }
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsSpread& data) {
    json_input.at("gauge_invariant_bohr_squared").get_to(data.gauge_invariant_bohr_squared);
    json_input.at("localized").get_to(data.localized);
    if (json_input.contains("total_bohr_squared")) {
        data.total_bohr_squared = json_input.at("total_bohr_squared").get<double>();
    }
    if (json_input.contains("guiding_centers")) {
        data.guiding_centers = json_input.at("guiding_centers").get<std::string>();
    }
    if (json_input.contains("trial_orbital_centers_bohr")) {
        data.trial_orbital_centers_bohr = json_input.at("trial_orbital_centers_bohr").get<std::vector<std::vector<double>>>();
    }
    if (json_input.contains("initial_minimizable_bohr_squared")) {
        data.initial_minimizable_bohr_squared = json_input.at("initial_minimizable_bohr_squared").get<double>();
    }
    if (json_input.contains("centers_bohr")) {
        data.centers_bohr = json_input.at("centers_bohr").get<std::vector<std::vector<double>>>();
    }
    if (json_input.contains("spreads_bohr_squared")) {
        data.spreads_bohr_squared = json_input.at("spreads_bohr_squared").get<std::vector<double>>();
    }
    if (json_input.contains("localization_iterations")) {
        data.localization_iterations = json_input.at("localization_iterations").get<int64_t>();
    }
    if (json_input.contains("localization_converged")) {
        data.localization_converged = json_input.at("localization_converged").get<bool>();
    }
    if (json_input.contains("localization_gradient_norm")) {
        data.localization_gradient_norm = json_input.at("localization_gradient_norm").get<double>();
    }
}

} // namespace q3::generated
