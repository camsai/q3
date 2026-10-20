#pragma once

/**
 * Factory for creating GPU execution policies.
 *
 * Creates a GPUExecutionPolicy backed by the given device backend.
 * In Phase 1, this factory will also accept a GPUBLAS reference for
 * the policy to delegate vector operations to.
 *
 * The concrete GPUExecutionPolicy is vendor-neutral — vendor selection
 * happens at the device backend level via create_gpu_device_backend().
 */

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/init.hpp"

#include <memory>

namespace q3 {
namespace execution {
namespace gpu {

/**
 * Create a GPU execution policy backed by the given device backend.
 *
 * @param device_backend  Shared pointer to the GPU device backend.
 * @return                Unique pointer to the GPU execution policy.
 */
inline std::unique_ptr<GPUExecutionPolicy>
create_gpu_execution_policy(std::shared_ptr<device::GPUDeviceBackend> device_backend) {
    return std::make_unique<GPUExecutionPolicy>(std::move(device_backend));
}

} // namespace gpu
} // namespace execution
} // namespace q3
