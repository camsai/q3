#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/output/device_transfers.hpp"
#include "io/output/results.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/types/init.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"

using namespace q3;
using test_support::CPUBackedGPUDeviceBackend;

TEST(DeviceTransfersRecord, ARunWithoutADeviceHasNoBlock) {
    generated::Results results = new_results("self_consistent_field");

    const nlohmann::json serialized = results;

    EXPECT_FALSE(serialized["runtime_log"].contains("device_transfers"));
}

TEST(DeviceTransfersRecord, TheBackendsTotalsGoIntoTheRuntimeLog) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    execution::gpu::device::GPUDeviceBuffer<Real> device_buffer(4, device_backend);
    std::vector<Real> host_array(4, 1.0);
    device_backend->copy(device_buffer.data(), host_array.data(), 4 * sizeof(Real));
    device_backend->copy(device_buffer.data(), host_array.data(), 4 * sizeof(Real));
    device_backend->copy(host_array.data(), device_buffer.data(), 2 * sizeof(Real));
    generated::Results results = new_results("self_consistent_field");

    record_device_transfers(results, *device_backend);

    const nlohmann::json serialized = results;
    const nlohmann::json& device_transfers = serialized["runtime_log"]["device_transfers"];
    EXPECT_EQ(device_transfers["host_to_device_bytes"], 8 * sizeof(Real));
    EXPECT_EQ(device_transfers["host_to_device_copies"], 2);
    EXPECT_EQ(device_transfers["device_to_host_bytes"], 2 * sizeof(Real));
    EXPECT_EQ(device_transfers["device_to_host_copies"], 1);
}
