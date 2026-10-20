/**
 * MetalGPUDeviceBackend — Apple Silicon GPU device backend via Metal.
 *
 * Uses MTLDevice and MTLBuffer with storageModeShared for unified memory
 * allocation. All allocated buffers are tracked in a map so that MPS
 * operations can reference the MTLBuffer for a given pointer.
 *
 * Device info (name, memory) is queried from the MTLDevice object.
 */

#import <Metal/Metal.h>

#include "layer_0_core/execution/gpu/device/metal.hpp"
#include "layer_0_core/execution/gpu/device/factory.hpp"

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unordered_map>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

class MetalGPUDeviceBackendImpl : public MetalGPUDeviceBackend {
  public:
    MetalGPUDeviceBackendImpl() {
        device_ = MTLCreateSystemDefaultDevice();
        /**
         * The "system default" device is nil in a process with no window-server
         * session -- an SSH login, a launch agent, a tool's shell -- even on a
         * machine with a perfectly good GPU. Enumerating the devices still
         * works there, so fall back to the first one rather than refusing on
         * hardware that is present. That is how an M1 Max came to report "no
         * Metal device found".
         */
        if (device_ == nil) {
            NSArray<id<MTLDevice>>* all_devices = MTLCopyAllDevices();
            if ([all_devices count] > 0) {
                device_ = all_devices[0];
            }
        }
        if (device_ == nil) {
            throw std::runtime_error("MetalGPUDeviceBackendImpl: no Metal device found");
        }
        command_queue_ = [device_ newCommandQueue];
        if (command_queue_ == nil) {
            throw std::runtime_error("MetalGPUDeviceBackendImpl: failed to create command queue");
        }
    }

    ~MetalGPUDeviceBackendImpl() override = default;

    /**
     * Human-readable name of the Metal GPU device.
     *
     * Returns the MTLDevice name (e.g., "Apple M1 Max").
     */
    std::string device_name() const override {
        return std::string([device_.name UTF8String]);
    }

    /**
     * Total system memory in bytes (unified with GPU on Apple Silicon).
     *
     * Queries hw.memsize via sysctlbyname().
     */
    size_t total_memory_bytes() const override {
        uint64_t memory_size = 0;
        size_t memory_size_length = sizeof(memory_size);
        if (sysctlbyname("hw.memsize", &memory_size, &memory_size_length, nullptr, 0) != 0) {
            throw std::runtime_error("MetalGPUDeviceBackendImpl: failed to query hw.memsize");
        }
        return static_cast<size_t>(memory_size);
    }

    /**
     * Apple Silicon GPUs do not support native FP64 compute shaders.
     */
    bool supports_double_precision() const override {
        return false;
    }

    /**
     * Allocate unified memory via MTLBuffer with storageModeShared.
     *
     * The returned pointer is accessible by both CPU and GPU with zero
     * copy on Apple Silicon. The MTLBuffer is tracked internally so
     * MPS operations can look up the buffer for a given pointer.
     */
    void* allocate_unified(size_t size_bytes) override {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:size_bytes
                                                    options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::runtime_error("MetalGPUDeviceBackendImpl: MTLBuffer allocation failed for " +
                                     std::to_string(size_bytes) + " bytes");
        }
        void* pointer = buffer.contents;
        {
            std::lock_guard<std::mutex> lock(buffer_map_mutex_);
            buffer_map_[pointer] = buffer;
        }
        return pointer;
    }

    /**
     * Deallocate memory previously allocated via allocate_unified().
     *
     * Removes the MTLBuffer from the tracking map. ARC releases the buffer.
     */
    void deallocate(void* pointer) override {
        std::lock_guard<std::mutex> lock(buffer_map_mutex_);
        buffer_map_.erase(pointer);
    }

    /**
     * No-op: storageModeShared ensures CPU/GPU coherence on Apple Silicon.
     */
    void synchronize() override {
        // Unified memory with storageModeShared — CPU/GPU are coherent.
    }

    // ── Metal-specific accessors (used by MPS BLAS and MPSGraph FFT) ─────

    /**
     * Get the Metal device.
     */
    id<MTLDevice> metal_device() const override {
        return device_;
    }

    /**
     * Get the Metal command queue.
     */
    id<MTLCommandQueue> metal_command_queue() const override {
        return command_queue_;
    }

    /**
     * Look up the MTLBuffer for a pointer allocated by this backend.
     *
     * Returns nil if the pointer was not allocated by this backend.
     */
    id<MTLBuffer> get_metal_buffer(const void* pointer) const override {
        std::lock_guard<std::mutex> lock(buffer_map_mutex_);
        auto iterator = buffer_map_.find(const_cast<void*>(pointer));
        if (iterator != buffer_map_.end()) {
            return iterator->second;
        }
        return nil;
    }

  private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    mutable std::mutex buffer_map_mutex_;
    std::unordered_map<void*, id<MTLBuffer>> buffer_map_;
};

// ── Factory ─────────────────────────────────────────────────────────────────

std::shared_ptr<GPUDeviceBackend> create_gpu_device_backend() {
    return std::make_shared<MetalGPUDeviceBackendImpl>();
}

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
