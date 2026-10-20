/**
 * CudaGPUDeviceBackend — CUDA GPU device backend.
 *
 * Provides device lifecycle management, unified memory allocation via
 * cudaMallocManaged(), and device synchronization via cudaDeviceSynchronize().
 *
 * Device info (name, memory) is queried via cudaGetDeviceProperties().
 */

#include "layer_0_core/execution/gpu/device/factory.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

namespace {

/**
 * Check a CUDA runtime API return code and throw on error.
 */
void check_cuda_error(cudaError_t error_code, const char* operation) {
    if (error_code != cudaSuccess) {
        throw std::runtime_error(std::string("CudaGPUDeviceBackend: ") + operation +
                                 " failed: " + cudaGetErrorString(error_code));
    }
}

} // anonymous namespace

class CudaGPUDeviceBackend : public GPUDeviceBackend {
  public:
    /**
     * Construct and initialize the CUDA device backend.
     *
     * Selects device 0 and queries its properties.
     */
    CudaGPUDeviceBackend() {
        check_cuda_error(cudaSetDevice(0), "cudaSetDevice");
        check_cuda_error(cudaGetDeviceProperties(&device_properties_, 0),
                         "cudaGetDeviceProperties");
    }

    /**
     * Human-readable name of the CUDA device (e.g., "NVIDIA A100").
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
     * CUDA GPUs support native FP64 compute on all supported architectures.
     */
    bool supports_double_precision() const override {
        return true;
    }

    /**
     * Allocate unified (managed) memory via cudaMallocManaged().
     *
     * Managed memory is accessible from both host and device with automatic
     * page migration. This simplifies the programming model at the cost of
     * some potential performance overhead compared to explicit transfers.
     */
    void* allocate_unified(size_t size_bytes) override {
        void* pointer = nullptr;
        check_cuda_error(cudaMallocManaged(&pointer, size_bytes), "cudaMallocManaged");
        return pointer;
    }

    /**
     * Deallocate memory previously allocated via allocate_unified().
     */
    void deallocate(void* pointer) override {
        check_cuda_error(cudaFree(pointer), "cudaFree");
    }

    /**
     * Block until all pending GPU operations have completed.
     */
    void synchronize() override {
        check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

  private:
    cudaDeviceProp device_properties_;
};

// ── Factory ─────────────────────────────────────────────────────────────────

std::shared_ptr<GPUDeviceBackend> create_gpu_device_backend() {
    return std::make_shared<CudaGPUDeviceBackend>();
}

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
