#pragma once

/**
 * GPUFFTGrid — vendor-neutral GPU FFT grid interface.
 *
 * Extends the existing FFTGrid interface with single-precision (FP32)
 * FFT capabilities for mixed-precision GPU computation. Provides both
 * FP64 (inherited from FFTGrid) and FP32 (new) forward/inverse transforms.
 *
 * Each concrete implementation owns its own vendor FFT handles (e.g.,
 * cuFFT plans or FFTW/vDSP plans).
 *
 * No CUDA headers. No Metal headers. Vendor-neutral interface only.
 * Concrete implementations live in:
 *   - cuda.hpp / cuda.cpp   (CudaGPUFFTGrid — cuFFT)
 *   - metal.hpp / metal.cpp (MetalGPUFFTGrid — FFTW FP64 + vDSP FP32)
 */

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/types/single_precision.hpp"
#include "layer_0_core/types/type_aliases.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"

#include <complex>
#include <memory>

namespace q3 {

/**
 * GPUFFTGrid lives in q3::gpu (not q3::execution::gpu) because it belongs to layer_3
 * (representation), while q3::execution is in layer_0 (core). Keeping it in q3::gpu
 * avoids an upward layer dependency.
 */
namespace gpu {

class GPUFFTGrid : public FFTGrid {
  public:
    /**
     * Construct a GPU FFT grid backed by the given device.
     *
     * FFT plans are created during init(), not in the constructor.
     * The device backend is stored for memory management and
     * synchronization.
     *
     * @param device_backend  Shared pointer to the GPU device backend.
     *                        Must not be null.
     */
    explicit GPUFFTGrid(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : device_backend_(std::move(device_backend)) {}

    ~GPUFFTGrid() override = default;

    // ── Single-precision (FP32) FFT operations ───────────────────────────

    /**
     * Forward single-precision FFT: real-space → reciprocal-space (unnormalized).
     *
     * @param data  In-out: single-precision complex grid data.
     */
    virtual void forward_single(ComplexSingle* data) const = 0;

    /**
     * Inverse single-precision FFT without normalization (raw backward).
     *
     * @param data  In-out: single-precision complex grid data.
     */
    virtual void inverse_raw_single(ComplexSingle* data) const = 0;

    /**
     * Batched forward single-precision FFT on multiple contiguous grids.
     *
     * Each grid is total_grid_points() contiguous ComplexSingle values.
     * Default loops over forward_single(); CUDA overrides with cufftPlanMany.
     *
     * @param data             Pointer to batch_size contiguous FFT grids.
     * @param batch_size       Number of grids to transform.
     */
    virtual void forward_single_batch(ComplexSingle* data, int batch_size) const {
        int grid_size = total_grid_points();
        for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
            forward_single(data + batch_index * grid_size);
        }
    }

    /**
     * Batched inverse single-precision FFT on multiple contiguous grids.
     *
     * Each grid is total_grid_points() contiguous ComplexSingle values.
     * Default loops over inverse_raw_single(); CUDA overrides with cufftPlanMany.
     *
     * @param data             Pointer to batch_size contiguous FFT grids.
     * @param batch_size       Number of grids to transform.
     */
    virtual void inverse_raw_single_batch(ComplexSingle* data, int batch_size) const {
        int grid_size = total_grid_points();
        for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
            inverse_raw_single(data + batch_index * grid_size);
        }
    }

    // ── Double-precision (FP64) FFT operations ───────────────────────────

    /**
     * Forward FFT (FP64). On CUDA, delegates to cuFFT. On Metal, falls back to
     * CPU FFTW since Apple GPUs lack native FP64 compute.
     *
     * Default implementation delegates to the base FFTGrid (CPU FFTW).
     * Subclasses with native FP64 GPU FFT support should override.
     */
    virtual void forward_transform(Complex* data) const {
        // Delegates via virtual dispatch to the concrete subclass (e.g., SerialFFTGrid).
        this->forward(data);
    }

    /**
     * Inverse FFT (FP64). On CUDA, delegates to cuFFT. On Metal, falls back to
     * CPU FFTW since Apple GPUs lack native FP64 compute.
     *
     * Default implementation delegates to the base FFTGrid (CPU FFTW).
     * Subclasses with native FP64 GPU FFT support should override.
     */
    virtual void inverse_transform(Complex* data) const {
        // Delegates via virtual dispatch to the concrete subclass (e.g., SerialFFTGrid).
        this->inverse_raw(data);
    }

    /**
     * Batched forward FP64 FFT on multiple contiguous grids.
     *
     * Each grid is total_grid_points() contiguous Complex values.
     * Default loops over forward_transform(); CUDA overrides with cufftPlanMany.
     *
     * @param data             Pointer to batch_size contiguous FFT grids.
     * @param batch_size       Number of grids to transform.
     */
    virtual void forward_transform_batch(Complex* data, int batch_size) const {
        int grid_size = total_grid_points();
        for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
            forward_transform(data + batch_index * grid_size);
        }
    }

    /**
     * Batched inverse FP64 FFT on multiple contiguous grids.
     *
     * Each grid is total_grid_points() contiguous Complex values.
     * Default loops over inverse_transform(); CUDA overrides with cufftPlanMany.
     *
     * @param data             Pointer to batch_size contiguous FFT grids.
     * @param batch_size       Number of grids to transform.
     */
    virtual void inverse_transform_batch(Complex* data, int batch_size) const {
        int grid_size = total_grid_points();
        for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
            inverse_transform(data + batch_index * grid_size);
        }
    }

    /**
     * Access the underlying GPU device backend.
     */
    execution::gpu::device::GPUDeviceBackend& device_backend() {
        return *device_backend_;
    }

    /**
     * Access the underlying GPU device backend (const).
     */
    const execution::gpu::device::GPUDeviceBackend& device_backend() const {
        return *device_backend_;
    }

  protected:
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
};

} // namespace gpu
} // namespace q3
