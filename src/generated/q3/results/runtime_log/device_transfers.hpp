#pragma once

// @generated from results/runtime_log/device_transfers.json — DO NOT EDIT

#include <cstdint>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * What the run copied between host memory and device memory, counted where each copy is made. A copy inside the device, or inside the host, is not counted. On a multi-process run these are the totals of the process that wrote the results.
 */
struct ResultsDeviceTransfers {
    // Bytes copied from host memory into device buffers.
    int64_t host_to_device_bytes{};
    // Bytes copied from device buffers into host memory.
    int64_t device_to_host_bytes{};
    // The number of copies that host_to_device_bytes adds up: the same bytes in many small copies is a different finding from one bulk copy.
    int64_t host_to_device_copies{};
    // The number of copies that device_to_host_bytes adds up.
    int64_t device_to_host_copies{};
};

inline void to_json(nlohmann::json& json_output, const ResultsDeviceTransfers& data) {
    json_output = nlohmann::json::object();
    json_output["host_to_device_bytes"] = data.host_to_device_bytes;
    json_output["device_to_host_bytes"] = data.device_to_host_bytes;
    json_output["host_to_device_copies"] = data.host_to_device_copies;
    json_output["device_to_host_copies"] = data.device_to_host_copies;
}

inline void from_json(const nlohmann::json& json_input, ResultsDeviceTransfers& data) {
    json_input.at("host_to_device_bytes").get_to(data.host_to_device_bytes);
    json_input.at("device_to_host_bytes").get_to(data.device_to_host_bytes);
    json_input.at("host_to_device_copies").get_to(data.host_to_device_copies);
    json_input.at("device_to_host_copies").get_to(data.device_to_host_copies);
}

} // namespace q3::generated
