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
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
    void* allocate_unified(size_t requested_bytes) override {
        const size_t size_bytes = block_size_for(requested_bytes);
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto cached = cached_blocks_.find(size_bytes);
        if (cached != cached_blocks_.end()) {
            void* pointer = cached->second;
            cached_blocks_.erase(cached);
            cached_bytes_ -= size_bytes;
            block_sizes_[pointer] = size_bytes;
            return pointer;
        }
        void* pointer = nullptr;
        check_cuda_error(cudaMallocManaged(&pointer, size_bytes), "cudaMallocManaged");
        block_sizes_[pointer] = size_bytes;
        return pointer;
    }

    /**
     * Return memory from allocate_unified() to the cache, or free it.
     *
     * cudaFree synchronizes the device, and the Davidson solver makes and drops small
     * GPUDeviceBuffers (coefficient matrices, factors) many times per call: 399 frees took 2.9 s
     * of a 64 s 64-atom silicon run on one A100. A freed block is kept for the next request of
     * the same size, up to cache_limit_bytes in all.
     */
    void deallocate(void* pointer) override {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto entry = block_sizes_.find(pointer);
        size_t size_bytes = entry != block_sizes_.end() ? entry->second : 0;
        if (entry != block_sizes_.end()) {
            block_sizes_.erase(entry);
        }
        if (size_bytes > 0 && cached_bytes_ + size_bytes <= cache_limit_bytes) {
            cached_blocks_.emplace(size_bytes, pointer);
            cached_bytes_ += size_bytes;
            return;
        }
        check_cuda_error(cudaFree(pointer), "cudaFree");
    }

    ~CudaGPUDeviceBackend() override {
        for (auto& [size_bytes, pointer] : cached_blocks_) {
            cudaFree(pointer);
        }
    }

    /**
     * Copy on the device: cudaMemcpyDefault resolves managed pointers to where their pages are,
     * so device-resident data is not migrated to the host to be copied.
     */
    void copy_bytes(void* destination, const void* source, size_t size_bytes) override {
        check_cuda_error(cudaMemcpy(destination, source, size_bytes, cudaMemcpyDefault),
                         "cudaMemcpy");
    }

    /// Zero on the device, so that device-resident pages are not migrated to be cleared.
    void fill_zero(void* destination, size_t size_bytes) override {
        if (size_bytes == 0) {
            return;
        }
        check_cuda_error(cudaMemset(destination, 0, size_bytes), "cudaMemset");
    }

    /**
     * Block until all pending GPU operations have completed.
     */
    void synchronize() override {
        check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

  private:
    cudaDeviceProp device_properties_;

    /**
     * The size actually allocated for a request: the next power of two up to
     * 64 MiB, the next multiple of 64 MiB above.
     *
     * A freed block is reused only by a request of its own size, and the
     * eigensolver's small buffers are sized by the number of unconverged
     * bands, which changes on every iteration -- so exact sizes almost never
     * met again, and each miss is a fresh managed allocation the device then
     * faults in page by page. Rounding makes neighbouring sizes share a block.
     */
    static size_t block_size_for(size_t requested_bytes) {
        constexpr size_t smallest_block = size_t{1} << 12;
        constexpr size_t large_block_step = size_t{1} << 26;
        if (requested_bytes == 0) {
            return 0;
        }
        if (requested_bytes > large_block_step) {
            return (requested_bytes + large_block_step - 1) / large_block_step * large_block_step;
        }
        size_t block = smallest_block;
        while (block < requested_bytes) {
            block <<= 1;
        }
        return block;
    }

    /// Most bytes of freed managed memory kept for reuse; see deallocate().
    static constexpr size_t cache_limit_bytes = size_t{1} << 30;
    std::mutex cache_mutex_;
    std::unordered_map<void*, size_t> block_sizes_;
    std::multimap<size_t, void*> cached_blocks_;
    size_t cached_bytes_ = 0;
};

// ── Factory ─────────────────────────────────────────────────────────────────

std::shared_ptr<GPUDeviceBackend> create_gpu_device_backend() {
    return std::make_shared<CudaGPUDeviceBackend>();
}

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
