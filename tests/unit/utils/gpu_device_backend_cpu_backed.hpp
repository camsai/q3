#pragma once

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>

#include "layer_0_core/execution/gpu/device/init.hpp"

namespace q3::test_support {

/**
 * A device backend whose device is the host: unified memory is ordinary heap
 * memory and there is nothing to wait for. With the serial kernels every
 * non-GPU build links, it lets device code paths run where the tests do.
 *
 * It keeps the base class's transfer accounting, so a test can read from
 * transfer_totals() how many bytes a code path moved between host arrays and
 * device buffers, although no byte left the host. A copy observer, when set,
 * is also told of each copy, for a test that needs to know which data moved
 * and not only how much.
 */
class CPUBackedGPUDeviceBackend : public execution::gpu::device::GPUDeviceBackend {
  public:
    std::string device_name() const override {
        return "host";
    }
    size_t total_memory_bytes() const override {
        return total_memory_bytes_;
    }
    /// Report this much device memory from now on; zero, the default, is "unknown".
    void set_total_memory_bytes(size_t total_memory_bytes) {
        total_memory_bytes_ = total_memory_bytes;
    }
    bool supports_double_precision() const override {
        return true;
    }
    void* allocate_unified(size_t size_bytes) override {
        return std::malloc(size_bytes);
    }
    void deallocate(void* pointer) override {
        std::free(pointer);
    }
    void synchronize() override {}

    /// What a copy observer is given: the arguments of one GPUDeviceBackend::copy call.
    using CopyObserver =
        std::function<void(const void* destination, const void* source, size_t size_bytes)>;

    /// Call the observer for every copy from now on, before the bytes move. Empty to stop.
    void set_copy_observer(CopyObserver copy_observer) {
        copy_observer_ = std::move(copy_observer);
    }

  protected:
    void copy_bytes(void* destination, const void* source, size_t size_bytes) override {
        if (copy_observer_) {
            copy_observer_(destination, source, size_bytes);
        }
        GPUDeviceBackend::copy_bytes(destination, source, size_bytes);
    }

  private:
    CopyObserver copy_observer_;
    size_t total_memory_bytes_ = 0;
};

} // namespace q3::test_support
