#pragma once

// @generated from results/runtime_log.json — DO NOT EDIT

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "generated/q3/results/runtime_log/execution.hpp"
#include "generated/q3/results/runtime_log/incomplete.hpp"

namespace q3::generated {

/**
 * How the calculation ran: which flavor, whether it converged and in how many iterations, the parallel topology and where the time went, and which derivatives are incomplete.
 */
struct ResultsRuntimeLog {
    // Which calculation produced this record: 'self_consistent_field', 'non_self_consistent_field' or 'verification'.
    std::string flavor;
    // Whether the run met its convergence criterion. A non-self-consistent pass reports true, having no criterion to miss.
    bool converged{};
    // Iterations actually performed: the one it converged on, or the whole budget when it did not. Absent for a flavor that does not iterate.
    std::optional<int64_t> self_consistent_field_iterations;
    // The density residual after each self-consistent field iteration, in order, the convergence test's own measure (the squared norm of the change in the density); its energies are the convergence_electronic record. Absent for a flavor that does not iterate to self-consistency.
    std::optional<std::vector<double>> density_residuals;
    // Where the run spent its time, in milliseconds, plus the call counts those times divide. Absent when nothing ran.
    std::optional<nlohmann::json> timings_ms;
    // The parallel topology the run ended up with, and its measured footprint.
    ResultsExecution execution;
    // Present only when a derivative was left out, each entry saying why. The Python relaxation workflows refuse to run on a result whose forces are incomplete.
    std::optional<ResultsIncomplete> incomplete;
};

inline void to_json(nlohmann::json& json_output, const ResultsRuntimeLog& data) {
    json_output = nlohmann::json::object();
    json_output["flavor"] = data.flavor;
    json_output["converged"] = data.converged;
    if (data.self_consistent_field_iterations) {
        json_output["self_consistent_field_iterations"] = *data.self_consistent_field_iterations;
    }
    if (data.density_residuals) {
        json_output["density_residuals"] = *data.density_residuals;
    }
    if (data.timings_ms) {
        json_output["timings_ms"] = *data.timings_ms;
    }
    json_output["execution"] = data.execution;
    if (data.incomplete) {
        json_output["incomplete"] = *data.incomplete;
    }
}

inline void from_json(const nlohmann::json& json_input, ResultsRuntimeLog& data) {
    json_input.at("flavor").get_to(data.flavor);
    json_input.at("converged").get_to(data.converged);
    if (json_input.contains("self_consistent_field_iterations")) {
        data.self_consistent_field_iterations = json_input.at("self_consistent_field_iterations").get<int64_t>();
    }
    if (json_input.contains("density_residuals")) {
        data.density_residuals = json_input.at("density_residuals").get<std::vector<double>>();
    }
    if (json_input.contains("timings_ms")) {
        data.timings_ms = json_input.at("timings_ms").get<nlohmann::json>();
    }
    json_input.at("execution").get_to(data.execution);
    if (json_input.contains("incomplete")) {
        data.incomplete = json_input.at("incomplete").get<ResultsIncomplete>();
    }
}

} // namespace q3::generated
