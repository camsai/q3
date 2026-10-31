#pragma once

// @generated from results/runtime_log/execution.json — DO NOT EDIT

#include <cstdint>
#include <optional>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * The parallel topology the run ended up with, and its measured footprint.
 */
struct ResultsExecution {
    int64_t number_of_processes{};
    // OpenMP threads each process may use (OMP_NUM_THREADS); 1 without OpenMP. Absent from results written before it was recorded.
    std::optional<int64_t> number_of_threads_per_process;
    int64_t number_of_kpoint_pools{};
    int64_t band_group_size{};
    bool reciprocal_space_vectors_are_distributed{};
    // The largest peak resident set over the ranks, not their sum: what a node has to fit is the biggest rank. Zero when no platform source could be read.
    int64_t peak_resident_memory_bytes{};
};

inline void to_json(nlohmann::json& json_output, const ResultsExecution& data) {
    json_output = nlohmann::json::object();
    json_output["number_of_processes"] = data.number_of_processes;
    if (data.number_of_threads_per_process) {
        json_output["number_of_threads_per_process"] = *data.number_of_threads_per_process;
    }
    json_output["number_of_kpoint_pools"] = data.number_of_kpoint_pools;
    json_output["band_group_size"] = data.band_group_size;
    json_output["reciprocal_space_vectors_are_distributed"] = data.reciprocal_space_vectors_are_distributed;
    json_output["peak_resident_memory_bytes"] = data.peak_resident_memory_bytes;
}

inline void from_json(const nlohmann::json& json_input, ResultsExecution& data) {
    json_input.at("number_of_processes").get_to(data.number_of_processes);
    if (json_input.contains("number_of_threads_per_process")) {
        data.number_of_threads_per_process = json_input.at("number_of_threads_per_process").get<int64_t>();
    }
    json_input.at("number_of_kpoint_pools").get_to(data.number_of_kpoint_pools);
    json_input.at("band_group_size").get_to(data.band_group_size);
    json_input.at("reciprocal_space_vectors_are_distributed").get_to(data.reciprocal_space_vectors_are_distributed);
    json_input.at("peak_resident_memory_bytes").get_to(data.peak_resident_memory_bytes);
}

} // namespace q3::generated
