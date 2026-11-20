#pragma once

/**
 * DeviceTransferTotals — how much data has crossed between host and device.
 *
 * A reading of GPUDeviceBackend's counters. The counters only grow, so the
 * traffic of one stage (one SCF iteration, one solve) is the difference of
 * the readings taken either side of it.
 */

#include <cstdint>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

/// One reading of a device backend's host-to-device and device-to-host copy counters.
struct DeviceTransferTotals {
    /// Bytes copied from host memory into device buffers.
    std::uint64_t host_to_device_bytes = 0;
    /// Bytes copied from device buffers into host memory.
    std::uint64_t device_to_host_bytes = 0;
    /// Number of copies that make up host_to_device_bytes.
    std::uint64_t host_to_device_copies = 0;
    /// Number of copies that make up device_to_host_bytes.
    std::uint64_t device_to_host_copies = 0;

    /// The traffic between an earlier reading and this one.
    [[nodiscard]] DeviceTransferTotals since(const DeviceTransferTotals& earlier) const {
        DeviceTransferTotals difference;
        difference.host_to_device_bytes = host_to_device_bytes - earlier.host_to_device_bytes;
        difference.device_to_host_bytes = device_to_host_bytes - earlier.device_to_host_bytes;
        difference.host_to_device_copies = host_to_device_copies - earlier.host_to_device_copies;
        difference.device_to_host_copies = device_to_host_copies - earlier.device_to_host_copies;
        return difference;
    }
};

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
