#pragma once

/**
 * MetalGPUDeviceBackend forward declaration header.
 *
 * Provides the Metal-specific accessors (device, command queue, buffer lookup)
 * that MPS BLAS and MPSGraph FFT need. This header can be included from
 * other .mm files without pulling in Metal/Metal.h.
 */

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#include "layer_0_core/execution/gpu/device/init.hpp"

#include <memory>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

/**
 * MetalGPUDeviceBackend — abstract interface for Metal-specific accessors.
 *
 * Concrete implementation lives in device/metal.mm.
 */
class MetalGPUDeviceBackend : public GPUDeviceBackend {
  public:
    virtual ~MetalGPUDeviceBackend() = default;

#ifdef __OBJC__
    /**
     * Get the Metal device.
     */
    virtual id<MTLDevice> metal_device() const = 0;

    /**
     * Get the Metal command queue.
     */
    virtual id<MTLCommandQueue> metal_command_queue() const = 0;

    /**
     * Look up the MTLBuffer for a pointer allocated by this backend.
     *
     * Returns nil if the pointer was not allocated by this backend.
     */
    virtual id<MTLBuffer> get_metal_buffer(const void* pointer) const = 0;
#endif
};

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
