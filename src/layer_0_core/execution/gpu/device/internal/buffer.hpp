#pragma once

/**
 * GPUDeviceBuffer — RAII wrapper for GPU unified (managed) memory.
 *
 * Templated on element type T. Uses GPUDeviceBackend::allocate_unified()
 * for allocation and GPUDeviceBackend::deallocate() for deallocation.
 *
 * Stores a std::shared_ptr<GPUDeviceBackend> to guarantee that the
 * backend outlives all buffers — critical for correct destruction order.
 *
 * No CUDA headers. No Metal headers. Uses the backend abstraction only.
 */

#include "layer_0_core/execution/gpu/device/init.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

/**
 * Layout compatibility assertions for GPU interop — complex types must be
 * layout-compatible with float[2] / double[2] for BLAS and FFT libraries.
 */
static_assert(sizeof(std::complex<float>) == 2 * sizeof(float),
              "complex<float> must be layout-compatible with float[2]");
static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
              "complex<double> must be layout-compatible with double[2]");

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

template <typename T> class GPUDeviceBuffer {
  public:
    /**
     * Construct a buffer of the given number of elements.
     *
     * @param element_count  Number of elements of type T to allocate.
     *                       If zero, no allocation is performed and data() returns nullptr.
     * @param backend        Shared pointer to the GPU device backend.
     *                       Must not be null.
     * @throws std::runtime_error  If allocation fails (e.g., out of GPU memory).
     */
    GPUDeviceBuffer(size_t element_count, std::shared_ptr<GPUDeviceBackend> backend)
        : element_count_(element_count), backend_(std::move(backend)), data_(nullptr) {
        if (!backend_) {
            throw std::runtime_error("GPUDeviceBuffer: backend must not be null");
        }
        if (element_count_ > 0) {
            data_ = static_cast<T*>(backend_->allocate_unified(element_count_ * sizeof(T)));
        }
    }

    // Non-copyable — unified memory pointers cannot be meaningfully duplicated.
    GPUDeviceBuffer(const GPUDeviceBuffer&) = delete;
    GPUDeviceBuffer& operator=(const GPUDeviceBuffer&) = delete;

    /**
     * Move constructor. Transfers ownership of the allocation.
     * The source buffer is left in a valid but empty state.
     */
    GPUDeviceBuffer(GPUDeviceBuffer&& other) noexcept
        : element_count_(other.element_count_)
        , backend_(std::move(other.backend_))
        , data_(other.data_) {
        other.element_count_ = 0;
        other.data_ = nullptr;
    }

    /**
     * Move assignment operator. Releases the current allocation (if any)
     * before taking ownership of the source's allocation.
     */
    GPUDeviceBuffer& operator=(GPUDeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            element_count_ = other.element_count_;
            backend_ = std::move(other.backend_);
            data_ = other.data_;
            other.element_count_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    ~GPUDeviceBuffer() {
        release();
    }

    /**
     * Raw pointer to the underlying unified memory.
     * Returns nullptr if the buffer was zero-sized or moved-from.
     */
    T* data() noexcept {
        return data_;
    }

    /**
     * Raw const pointer to the underlying unified memory.
     */
    const T* data() const noexcept {
        return data_;
    }

    /**
     * Number of elements of type T in this buffer.
     */
    size_t size() const noexcept {
        return element_count_;
    }

    /**
     * Total size of the allocation in bytes.
     */
    size_t size_bytes() const noexcept {
        return element_count_ * sizeof(T);
    }

  private:
    void release() noexcept {
        if (data_ && backend_) {
            backend_->deallocate(data_);
            data_ = nullptr;
        }
        element_count_ = 0;
    }

    size_t element_count_;
    std::shared_ptr<GPUDeviceBackend> backend_;
    T* data_;
};

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
