#pragma once

/**
 * SYCLGPUDeviceBackend header.
 *
 * Provides the SYCL-specific queue() accessor that oneMKL BLAS, LAPACK, and
 * DFT wrappers need. This header can be included from other SYCL .cpp files
 * without pulling in the full device implementation.
 *
 * The concrete implementation lives in device/sycl.cpp.
 */

#include "layer_0_core/execution/gpu/device/init.hpp"

#include <sycl/sycl.hpp>

namespace q3 {
namespace execution {
namespace gpu {
namespace device {

/**
 * SYCLGPUDeviceBackend — abstract interface for SYCL-specific accessors.
 *
 * Exposes the SYCL queue needed by oneMKL BLAS/LAPACK/DFT operations.
 * Concrete implementation lives in device/sycl.cpp.
 */
class SYCLGPUDeviceBackend : public GPUDeviceBackend {
  public:
    virtual ~SYCLGPUDeviceBackend() = default;

    /**
     * Access the underlying SYCL queue for vendor-specific operations.
     *
     * Used by SYCL BLAS/LAPACK/FFT wrappers that need the queue to submit
     * oneMKL operations.
     */
    virtual sycl::queue& queue() = 0;

    /**
     * Access the underlying SYCL queue (const).
     */
    virtual const sycl::queue& queue() const = 0;
};

} // namespace device
} // namespace gpu
} // namespace execution
} // namespace q3
