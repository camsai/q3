/**
 * HIPGPUDeviceBackend — HIP/ROCm GPU device backend.
 *
 * Provides device lifecycle management, unified memory allocation via
 * hipMallocManaged(), and device synchronization via hipDeviceSynchronize().
 *
 * Device info (name, memory) is queried via hipGetDeviceProperties().
 */

#include "layer_0_core/execution/gpu/device/factory.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

namespace {

/**
 * Check a HIP runtime API return code and throw on error.
 */
void check_hip_error(hipError_t error_code, const char* operation) {
    if (error_code != hipSuccess) {
        throw std::runtime_error(std::string("HIPGPUDeviceBackend: ") + operation +
                                 " failed: " + hipGetErrorString(error_code));
    }
}

} // anonymous namespace

class HIPGPUDeviceBackend : public GPUDeviceBackend {
  public:
    /**
     * Construct and initialize the HIP device backend.
     *
     * Selects device 0 and queries its properties.
     */
    HIPGPUDeviceBackend() {
        check_hip_error(hipSetDevice(0), "hipSetDevice");
        check_hip_error(hipGetDeviceProperties(&device_properties_, 0), "hipGetDeviceProperties");
    }

    /**
     * Human-readable name of the HIP device (e.g., "AMD Instinct MI250X").
     */
    std::string device_name() const override {
        return std::string(device_properties_.name);
    }

    /**
     * Total device global memory in bytes.
     */
    size_t total_memory_bytes() const override {
        return device_properties_.totalGlobalMem;
    }

    /**
     * AMD Instinct GPUs support native FP64 compute on all supported architectures.
     */
    bool supports_double_precision() const override {
        return true;
    }

    /**
     * Allocate unified (managed) memory via hipMallocManaged().
     *
     * Managed memory is accessible from both host and device with automatic
     * page migration. This simplifies the programming model at the cost of
     * some potential performance overhead compared to explicit transfers.
     */
    void* allocate_unified(size_t size_bytes) override {
        void* pointer = nullptr;
        check_hip_error(hipMallocManaged(&pointer, size_bytes), "hipMallocManaged");
        return pointer;
    }

    /**
     * Deallocate memory previously allocated via allocate_unified().
     */
    void deallocate(void* pointer) override {
        check_hip_error(hipFree(pointer), "hipFree");
    }

    /**
     * Block until all pending GPU operations have completed.
     */
    void synchronize() override {
        check_hip_error(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }

  private:
    hipDeviceProp_t device_properties_;
};

// ── Factory ─────────────────────────────────────────────────────────────────

std::shared_ptr<GPUDeviceBackend> create_gpu_device_backend() {
    return std::make_shared<HIPGPUDeviceBackend>();
}

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
