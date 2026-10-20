#pragma once

/**
 * Factory for creating GPU device backends.
 *
 * Returns std::shared_ptr (not unique_ptr) because GPUDeviceBuffer
 * stores a shared_ptr<GPUDeviceBackend> to guarantee the backend
 * outlives all buffers allocated through it.
 *
 * The concrete implementation is selected at link time via CMake:
 * CUDA builds link the CUDA factory, Metal builds link the Metal factory.
 * No #ifdef guards in this header.
 */

#include "layer_0_core/execution/gpu/device/init.hpp"

#include <memory>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

/**
 * Create a GPU device backend for the platform detected at build time.
 *
 * @return  Shared pointer to the concrete GPUDeviceBackend implementation.
 * @throws std::runtime_error  If no GPU device is available.
 */
std::shared_ptr<GPUDeviceBackend> create_gpu_device_backend();

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
