/**
 * SYCLGPUDeviceBackendImpl — concrete SYCL/oneAPI GPU device backend.
 *
 * Provides device lifecycle management, unified shared memory allocation
 * via sycl::malloc_shared(), and device synchronization via queue.wait_and_throw().
 *
 * Device info (name, memory) is queried via SYCL device info descriptors.
 * The default selector picks the best available device (GPU if present,
 * otherwise falls back to CPU for development/testing).
 */

#include "layer_0_core/execution/gpu/device/sycl.hpp"
#include "layer_0_core/execution/gpu/device/factory.hpp"

#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

class SYCLGPUDeviceBackendImpl : public SYCLGPUDeviceBackend {
  public:
    /**
     * Construct and initialize the SYCL device backend.
     *
     * Uses the default selector, which prefers GPU devices when available
     * and falls back to CPU for development/testing on systems without
     * a discrete GPU.
     */
    SYCLGPUDeviceBackendImpl() : queue_(sycl::default_selector_v) {}

    /**
     * Human-readable name of the SYCL device (e.g., "Intel(R) Data Center GPU Max 1550").
     */
    std::string device_name() const override {
        return queue_.get_device().get_info<sycl::info::device::name>();
    }

    /**
     * Total device global memory in bytes.
     */
    size_t total_memory_bytes() const override {
        return queue_.get_device().get_info<sycl::info::device::global_mem_size>();
    }

    /**
     * Whether the device supports native double precision (FP64) computation.
     *
     * Checks for the sycl::aspect::fp64 capability. Most Intel Data Center GPUs
     * support FP64; some consumer GPUs may not.
     */
    bool supports_double_precision() const override {
        return queue_.get_device().has(sycl::aspect::fp64);
    }

    /**
     * Allocate unified shared memory via sycl::malloc_shared().
     *
     * Shared memory is accessible from both host and device with automatic
     * data migration managed by the SYCL runtime. This is the SYCL
     * equivalent of CUDA's cudaMallocManaged().
     */
    void* allocate_unified(size_t size_bytes) override {
        void* pointer = sycl::malloc_shared(size_bytes, queue_);
        if (pointer == nullptr) {
            throw std::runtime_error("SYCLGPUDeviceBackendImpl: sycl::malloc_shared failed for " +
                                     std::to_string(size_bytes) + " bytes");
        }
        return pointer;
    }

    /**
     * Deallocate memory previously allocated via allocate_unified().
     */
    void deallocate(void* pointer) override {
        sycl::free(pointer, queue_);
    }

    /**
     * Block until all pending GPU operations on this queue have completed.
     *
     * Uses wait_and_throw() to surface any asynchronous exceptions from
     * previously submitted kernel or memory operations.
     */
    void synchronize() override {
        queue_.wait_and_throw();
    }

    /**
     * Access the underlying SYCL queue for vendor-specific operations.
     */
    sycl::queue& queue() override {
        return queue_;
    }

    /**
     * Access the underlying SYCL queue (const).
     */
    const sycl::queue& queue() const override {
        return queue_;
    }

  private:
    sycl::queue queue_;
};

// ── Factory ─────────────────────────────────────────────────────────────────

std::shared_ptr<GPUDeviceBackend> create_gpu_device_backend() {
    return std::make_shared<SYCLGPUDeviceBackendImpl>();
}

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
