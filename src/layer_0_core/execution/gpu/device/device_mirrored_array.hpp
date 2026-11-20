#pragma once

/**
 * DeviceMirroredArray — one array, a host copy and an optional device copy of it,
 * and the knowledge of which of the two is current.
 *
 * The owner of data that both host code and device kernels use. A reader asks
 * for the side it runs on, host() or device(backend); if that copy is stale it
 * is brought up to date with one bulk GPUDeviceBackend::copy, and both are
 * then current. A writer asks for mutable_host() or mutable_device(backend),
 * which marks the other copy stale at that moment. Nothing else moves the
 * data: no host code touches the device buffer, so no crossing happens by page
 * migration, and every crossing is a copy the backend counts.
 *
 * Neither copy is storage the array must hold. The device copy is made when
 * something asks for the array on a device. The host copy is made when
 * something asks for it on the host: an array that a device kernel fills, and
 * that only device code reads, never takes host memory at all. An array whose
 * elements nothing has written yet holds value-initialized elements (zeros,
 * for arithmetic types) and may hold no storage on either side.
 *
 * The array is a plain host array until something asks for it on a device.
 * Whoever asks names the device: a device consumer always has its backend, so
 * nothing has to be attached to the array beforehand, and no order of set-up
 * calls can leave it without one. The device copy can be dropped at any time
 * with release_device(), which keeps the values: the device side is a cache of
 * the array, never the only place it lives once released.
 *
 * A pointer from any accessor is good until the next call that writes the
 * other side, or resizes or releases; do not keep one across such a call.
 *
 * Threads. Bringing a stale copy up to date is serialized, so any number of
 * threads may read the same side at once, even when the first of them finds
 * it stale: a parallel loop over bands can call host() per band. Writers are
 * not serialized against anything: no thread may ask for a writable view, or
 * for the other side, while another is reading.
 */

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

/// An array with a host copy, an optional device copy, and which of them is current.
template <typename Element> class DeviceMirroredArray {
  public:
    DeviceMirroredArray() = default;

    /// An array of value-initialized elements (zeros, for arithmetic types), current on the host.
    explicit DeviceMirroredArray(std::size_t number_of_elements)
        : number_of_elements_(number_of_elements), host_elements_(number_of_elements) {}

    /// A copy holds the same values on the host, and has no device copy until asked for one.
    DeviceMirroredArray(const DeviceMirroredArray& other)
        : number_of_elements_(other.number_of_elements_)
        , host_elements_(other.current_host_elements()) {}

    DeviceMirroredArray& operator=(const DeviceMirroredArray& other) {
        if (this != &other) {
            host_elements_ = other.current_host_elements();
            number_of_elements_ = other.number_of_elements_;
            device_elements_.reset();
            device_backend_.reset();
            host_is_current_.store(true);
            device_is_current_.store(false);
        }
        return *this;
    }

    DeviceMirroredArray(DeviceMirroredArray&& other) noexcept
        : number_of_elements_(other.number_of_elements_)
        , host_elements_(std::move(other.host_elements_))
        , device_elements_(std::move(other.device_elements_))
        , device_backend_(std::move(other.device_backend_))
        , host_is_current_(other.host_is_current_.load())
        , device_is_current_(other.device_is_current_.load()) {
        other.reset_to_empty();
    }

    DeviceMirroredArray& operator=(DeviceMirroredArray&& other) noexcept {
        if (this != &other) {
            number_of_elements_ = other.number_of_elements_;
            host_elements_ = std::move(other.host_elements_);
            device_elements_ = std::move(other.device_elements_);
            device_backend_ = std::move(other.device_backend_);
            host_is_current_.store(other.host_is_current_.load());
            device_is_current_.store(other.device_is_current_.load());
            other.reset_to_empty();
        }
        return *this;
    }

    ~DeviceMirroredArray() = default;

    /// Number of elements.
    [[nodiscard]] std::size_t size() const noexcept {
        return number_of_elements_;
    }

    /**
     * Change the number of elements, as std::vector::resize does: the leading elements keep
     * their values and new ones are value-initialized. The device copy is dropped.
     */
    void resize(std::size_t number_of_elements) {
        release_device();
        if (host_is_current_.load(std::memory_order_relaxed)) {
            host_elements_.resize(number_of_elements);
        }
        // Otherwise nothing has been written and nothing is held: only the size changes.
        number_of_elements_ = number_of_elements;
    }

    /**
     * Change the number of elements and forget their values: the array is then as a new one of
     * that size, holding value-initialized elements and no storage on either side. Nothing is
     * copied, which is what sets this apart from resize(): an array whose only current copy is
     * on a device is not brought down first. For an owner about to refill the array, or done
     * with it.
     */
    void reset(std::size_t number_of_elements) {
        device_elements_.reset();
        device_backend_.reset();
        std::vector<Element>().swap(host_elements_);
        number_of_elements_ = number_of_elements;
        device_is_current_.store(false, std::memory_order_release);
        // An array of no elements has nothing to allocate, so its host copy is current as it is.
        host_is_current_.store(number_of_elements == 0, std::memory_order_release);
    }

    /// Whether a device copy takes memory, current or not.
    [[nodiscard]] bool device_is_allocated() const noexcept {
        return device_elements_.has_value() && number_of_elements_ > 0;
    }

    /// Whether the host copy takes memory. False for an array only ever written on a device.
    [[nodiscard]] bool host_is_allocated() const noexcept {
        return host_is_current_.load(std::memory_order_acquire) ||
               (number_of_elements_ > 0 && host_elements_.size() == number_of_elements_);
    }

    /// Whether the host copy holds the array's values.
    [[nodiscard]] bool host_is_current() const noexcept {
        return host_is_current_.load(std::memory_order_acquire);
    }

    /// Whether a device copy exists and holds the array's values.
    [[nodiscard]] bool device_is_current() const noexcept {
        return device_is_current_.load(std::memory_order_acquire);
    }

    /// The array on the host, to read. Downloads it first if the host copy is stale.
    [[nodiscard]] const Element* host() const {
        if (!host_is_current_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(refresh_mutex_);
            if (!host_is_current_.load(std::memory_order_relaxed)) {
                refresh_host();
            }
        }
        return host_elements_.data();
    }

    /**
     * The array on the host, to write. The device copy is stale from this call on. A stale host
     * copy is downloaded first, so a caller that writes only part of the array keeps the rest.
     */
    [[nodiscard]] Element* mutable_host() {
        if (!host_is_current_.load(std::memory_order_relaxed)) {
            refresh_host();
        }
        device_is_current_.store(false, std::memory_order_release);
        return host_elements_.data();
    }

    /**
     * The array on a device, to read. Makes the device copy if there is none on this backend
     * and uploads if it is stale; a copy held on another backend is released first.
     *
     * @throws std::invalid_argument  If the backend is null.
     */
    [[nodiscard]] const Element*
    device(const std::shared_ptr<GPUDeviceBackend>& device_backend) const {
        if (!device_is_current_.load(std::memory_order_acquire) ||
            device_backend != device_backend_) {
            std::lock_guard<std::mutex> lock(refresh_mutex_);
            hold_device_copy_on(device_backend);
            if (!device_is_current_.load(std::memory_order_relaxed)) {
                upload();
            }
        }
        return device_elements_->data();
    }

    /**
     * The array on a device, to write. The host copy is stale from this call on. A stale
     * device copy is uploaded first, so a kernel that writes only part of the array keeps
     * the rest.
     *
     * @throws std::invalid_argument  If the backend is null.
     */
    [[nodiscard]] Element* mutable_device(const std::shared_ptr<GPUDeviceBackend>& device_backend) {
        hold_device_copy_on(device_backend);
        if (!device_is_current_.load(std::memory_order_relaxed)) {
            upload();
        }
        host_is_current_.store(number_of_elements_ == 0, std::memory_order_release);
        return device_elements_->data();
    }

    /**
     * The array on a device, for a writer that assigns every element. Nothing is uploaded:
     * what the device copy holds on return is unspecified, and the host copy is stale. The
     * caller writes the whole array before anything reads it, on either side.
     *
     * @throws std::invalid_argument  If the backend is null.
     */
    [[nodiscard]] Element*
    overwrite_device(const std::shared_ptr<GPUDeviceBackend>& device_backend) {
        require(device_backend);
        if (!device_elements_ || device_backend_ != device_backend) {
            // Nothing is downloaded from a copy held elsewhere: all of it is about to be replaced.
            device_elements_.reset();
            device_backend_ = device_backend;
            device_elements_.emplace(number_of_elements_, device_backend_);
        }
        device_is_current_.store(true, std::memory_order_release);
        // An array of no elements has no host copy to go stale.
        host_is_current_.store(number_of_elements_ == 0, std::memory_order_release);
        return device_elements_->data();
    }

    /**
     * Free the device copy. The values are kept: if only the device copy was current, the host
     * copy is downloaded first. A later device() uploads again.
     */
    void release_device() {
        if (!device_elements_) {
            return;
        }
        if (!host_is_current_.load(std::memory_order_relaxed)) {
            refresh_host();
        }
        device_elements_.reset();
        device_backend_.reset();
        device_is_current_.store(false, std::memory_order_release);
    }

  private:
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return number_of_elements_ * sizeof(Element);
    }

    /// The host elements, downloaded first if stale.
    [[nodiscard]] const std::vector<Element>& current_host_elements() const {
        (void)host();
        return host_elements_;
    }

    /**
     * Make the host copy current. It is allocated here if it never was, value-initialized, and
     * then filled by one copy from the device if the values are there. When neither copy is
     * current nothing has been written, and the value-initialized elements are the values.
     */
    void refresh_host() const {
        if (host_elements_.size() != number_of_elements_) {
            host_elements_.resize(number_of_elements_);
        }
        if (device_is_current_.load(std::memory_order_relaxed) && size_bytes() > 0) {
            device_backend_->copy(host_elements_.data(), device_elements_->data(), size_bytes());
        }
        host_is_current_.store(true, std::memory_order_release);
    }

    /**
     * Make the device copy current: one copy from the host, or, when nothing has been written
     * on either side, zeros written on the device without a host copy being made for them.
     */
    void upload() const {
        if (size_bytes() > 0) {
            if (host_is_current_.load(std::memory_order_relaxed)) {
                device_backend_->copy(
                    device_elements_->data(), host_elements_.data(), size_bytes());
            } else {
                device_backend_->fill_zero(device_elements_->data(), size_bytes());
            }
        }
        device_is_current_.store(true, std::memory_order_release);
    }

    /**
     * Make sure a device buffer exists on this backend. A copy held on another backend is
     * released, after downloading it if it was the only current one; the buffer made here is
     * not current until something fills it.
     */
    void hold_device_copy_on(const std::shared_ptr<GPUDeviceBackend>& device_backend) const {
        require(device_backend);
        if (device_elements_ && device_backend_ == device_backend) {
            return;
        }
        if (device_elements_) {
            if (!host_is_current_.load(std::memory_order_relaxed)) {
                refresh_host();
            }
            device_elements_.reset();
        }
        device_backend_ = device_backend;
        device_elements_.emplace(number_of_elements_, device_backend_);
        device_is_current_.store(false, std::memory_order_release);
    }

    /// Refuse a null backend before anything is changed.
    static void require(const std::shared_ptr<GPUDeviceBackend>& device_backend) {
        if (!device_backend) {
            throw std::invalid_argument(
                "DeviceMirroredArray: a device view was requested with a null device backend");
        }
    }

    /// What a moved-from array is left as: no elements and no device copy.
    void reset_to_empty() noexcept {
        number_of_elements_ = 0;
        host_elements_.clear();
        device_elements_.reset();
        device_backend_.reset();
        host_is_current_.store(true);
        device_is_current_.store(false);
    }

    /**
     * The copies, where the device copy is held, and their state are mutable because a
     * read-only view may have to bring its side up to date: that changes where the values are
     * held, not the values. The two flags are atomic so that readers can test them without
     * the mutex, which is taken only to bring a stale side up to date.
     */
    std::size_t number_of_elements_ = 0;
    mutable std::vector<Element> host_elements_;
    mutable std::optional<GPUDeviceBuffer<Element>> device_elements_;
    mutable std::shared_ptr<GPUDeviceBackend> device_backend_;
    mutable std::atomic<bool> host_is_current_{true};
    mutable std::atomic<bool> device_is_current_{false};
    mutable std::mutex refresh_mutex_;
};

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
