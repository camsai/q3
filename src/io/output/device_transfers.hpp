#pragma once

/**
 * What a device backend copied between host and device during a run, as the
 * runtime log's device_transfers block.
 *
 * Only the entry points that own a device backend call this, after the flavor
 * has returned its results: a run with no device has no such block, rather
 * than one of zeros that reads as "a device run that moved nothing".
 */

#include <cstdint>

#include "generated/q3/results.hpp"
#include "io/logger.hpp"
#include "layer_0_core/execution/gpu/device/device_transfer_totals.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"

namespace q3 {

/// Write the backend's transfer totals so far into results.runtime_log.device_transfers.
inline void record_device_transfers(generated::Results& results,
                                    const execution::gpu::device::GPUDeviceBackend& backend) {
    const execution::gpu::device::DeviceTransferTotals totals = backend.transfer_totals();
    generated::ResultsDeviceTransfers device_transfers;
    device_transfers.host_to_device_bytes = static_cast<std::int64_t>(totals.host_to_device_bytes);
    device_transfers.device_to_host_bytes = static_cast<std::int64_t>(totals.device_to_host_bytes);
    device_transfers.host_to_device_copies =
        static_cast<std::int64_t>(totals.host_to_device_copies);
    device_transfers.device_to_host_copies =
        static_cast<std::int64_t>(totals.device_to_host_copies);
    results.runtime_log.device_transfers = device_transfers;
}

/// One line per direction, for the end-of-run summary: gibibytes moved and in how many copies.
inline void log_device_transfers(const execution::gpu::device::GPUDeviceBackend& backend) {
    const execution::gpu::device::DeviceTransferTotals totals = backend.transfer_totals();
    constexpr double bytes_per_gibibyte = 1024.0 * 1024.0 * 1024.0;
    log::info("     Copied host to device:          {:.3f} GiB in {} copies",
              static_cast<double>(totals.host_to_device_bytes) / bytes_per_gibibyte,
              totals.host_to_device_copies);
    log::info("     Copied device to host:          {:.3f} GiB in {} copies",
              static_cast<double>(totals.device_to_host_bytes) / bytes_per_gibibyte,
              totals.device_to_host_copies);
}

} // namespace q3
