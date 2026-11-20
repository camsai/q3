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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <string>

#include "layer_0_core/execution/gpu/device/device_transfer_totals.hpp"

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

template <typename T> class GPUDeviceBuffer;

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

    /**
     * Copy bytes between two buffers from allocate_unified() (or between host memory and one).
     *
     * A host memcpy between managed buffers that live on the device first migrates both to the
     * host, page by page, and the next kernel migrates them back; copy_bytes() is where a backend
     * whose managed memory is not simply host memory copies on the device instead. Synchronous:
     * the data is in place when it returns.
     *
     * Every copy with exactly one end in a GPUDeviceBuffer is a crossing between host and device,
     * and is counted in transfer_totals(). A copy between two device buffers, or between two
     * host arrays, is not.
     */
    void copy(void* destination, const void* source, size_t size_bytes) {
        const bool destination_is_device_memory = is_device_memory(destination);
        const bool source_is_device_memory = is_device_memory(source);
        if (destination_is_device_memory && !source_is_device_memory) {
            host_to_device_bytes_.fetch_add(size_bytes, std::memory_order_relaxed);
            host_to_device_copies_.fetch_add(1, std::memory_order_relaxed);
        } else if (source_is_device_memory && !destination_is_device_memory) {
            device_to_host_bytes_.fetch_add(size_bytes, std::memory_order_relaxed);
            device_to_host_copies_.fetch_add(1, std::memory_order_relaxed);
        }
        copy_bytes(destination, source, size_bytes);
    }

    /**
     * What copy() has moved between host and device since the backend was made.
     *
     * Totals only grow; a caller that wants the traffic of one stage takes the difference of two
     * readings.
     */
    [[nodiscard]] DeviceTransferTotals transfer_totals() const {
        DeviceTransferTotals totals;
        totals.host_to_device_bytes = host_to_device_bytes_.load(std::memory_order_relaxed);
        totals.device_to_host_bytes = device_to_host_bytes_.load(std::memory_order_relaxed);
        totals.host_to_device_copies = host_to_device_copies_.load(std::memory_order_relaxed);
        totals.device_to_host_copies = device_to_host_copies_.load(std::memory_order_relaxed);
        return totals;
    }

    /// Whether the pointer lies inside a live GPUDeviceBuffer of this backend.
    [[nodiscard]] bool is_device_memory(const void* pointer) const {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        std::lock_guard<std::mutex> lock(device_ranges_mutex_);
        auto next_range = device_ranges_.upper_bound(address);
        if (next_range == device_ranges_.begin()) {
            return false;
        }
        return address < std::prev(next_range)->second;
    }

    /**
     * Zero a range that may be device-resident unified memory.
     *
     * The default writes from the host. A backend whose unified memory lives on a device
     * overrides it to zero there: a host write to a device-resident range pulls every page
     * across first, and the next kernel pushes them back.
     */
    virtual void fill_zero(void* destination, size_t size_bytes) {
        std::memset(destination, 0, size_bytes);
    }

  protected:
    /**
     * The copy itself, behind copy()'s accounting. The default is a host memcpy, which is right
     * where unified memory is host memory; a backend whose unified memory lives on a device
     * overrides it to copy there.
     */
    virtual void copy_bytes(void* destination, const void* source, size_t size_bytes) {
        std::memcpy(destination, source, size_bytes);
    }

  private:
    /**
     * GPUDeviceBuffer tells the backend which address ranges are its device memory, for as long
     * as each buffer lives. That is what lets copy() name the direction of a transfer from its
     * two pointers, including pointers into the middle of a buffer.
     */
    template <typename T> friend class GPUDeviceBuffer;

    void register_device_range(const void* pointer, size_t size_bytes) {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        std::lock_guard<std::mutex> lock(device_ranges_mutex_);
        device_ranges_[address] = address + size_bytes;
    }

    void unregister_device_range(const void* pointer) {
        std::lock_guard<std::mutex> lock(device_ranges_mutex_);
        device_ranges_.erase(reinterpret_cast<std::uintptr_t>(pointer));
    }

    /// First address of each live device buffer, mapped to the address one past its end.
    std::map<std::uintptr_t, std::uintptr_t> device_ranges_;
    mutable std::mutex device_ranges_mutex_;

    std::atomic<std::uint64_t> host_to_device_bytes_{0};
    std::atomic<std::uint64_t> device_to_host_bytes_{0};
    std::atomic<std::uint64_t> host_to_device_copies_{0};
    std::atomic<std::uint64_t> device_to_host_copies_{0};
};

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
