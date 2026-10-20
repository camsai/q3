#pragma once

/**
 * GPUDeviceBackend — abstract interface for GPU device management.
 *
 * Provides device lifecycle, unified memory allocation/deallocation,
 * and synchronization. Has no knowledge of BLAS, FFT, or any math
 * library — those wrappers own their own vendor handles.
 *
 * No CUDA headers. No Metal headers. Vendor-specific implementations
 * live in cuda.hpp/cuda.cpp and metal.hpp/metal.cpp.
 */

#include <cstddef>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

class GPUDeviceBackend {
  public:
    virtual ~GPUDeviceBackend() = default;

    /**
     * Human-readable name of the GPU device (e.g., "Apple M2 Max", "NVIDIA A100").
     */
    virtual std::string device_name() const = 0;

    /**
     * Total device memory in bytes.
     */
    virtual size_t total_memory_bytes() const = 0;

    /**
     * Whether the device supports native double precision (FP64) computation.
     *
     * Returns false for Apple Silicon (Metal has no FP64 compute shaders).
     * Returns true for CUDA GPUs (all supported compute capabilities).
     */
    virtual bool supports_double_precision() const = 0;

    /**
     * Allocate unified (managed) memory accessible from both host and device.
     *
     * On CUDA, this maps to cudaMallocManaged(). On Metal, this maps to
     * MTLDevice::newBuffer() with MTLResourceStorageModeShared.
     *
     * The returned memory is NOT zero-initialized.
     *
     * @param size_bytes  Number of bytes to allocate.
     * @return            Pointer to the allocated memory.
     * @throws std::runtime_error  If allocation fails.
     */
    virtual void* allocate_unified(size_t size_bytes) = 0;

    /**
     * Deallocate memory previously allocated via allocate_unified().
     *
     * @param pointer  Pointer returned by allocate_unified(). Must not be null.
     */
    virtual void deallocate(void* pointer) = 0;

    /**
     * Block until all pending GPU operations have completed.
     *
     * Must be called before reading GPU-produced results on the host,
     * and before precision casts or MPI collectives.
     */
    virtual void synchronize() = 0;
};

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
