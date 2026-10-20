#pragma once

/**
 * MetalGPUFFTGrid — GPUFFTGrid implementation for Apple Silicon.
 *
 * FP64 transforms delegate to the internal SerialFFTGrid (CPU FFTW),
 * since Metal has no FP64 compute.
 *
 * FP32 transforms use Accelerate vDSP (native single-precision FFT on
 * the AMX coprocessor) when grid dimensions are powers of 2. Otherwise,
 * falls back to the FP64 upcast/downcast approach via FFTW.
 *
 * Implementation is in metal.mm to avoid leaking Accelerate headers
 * into consumers.
 */

#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"

#include <memory>
#include <vector>

namespace q3 {
namespace gpu {

class MetalGPUFFTGrid : public GPUFFTGrid {
  public:
    explicit MetalGPUFFTGrid(
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend = nullptr);

    ~MetalGPUFFTGrid() override;

    // ── FFTGrid base interface (delegated to serial_fft_grid_) ───────────

    void init(int dimension_1,
              int dimension_2,
              int dimension_3,
              bool use_measured_planning = false) override;

    [[nodiscard]] int grid_dimension_1() const override;
    [[nodiscard]] int grid_dimension_2() const override;
    [[nodiscard]] int grid_dimension_3() const override;
    [[nodiscard]] int total_grid_points() const override;
    [[nodiscard]] int local_grid_points() const override;
    [[nodiscard]] int local_offset() const override;

    void forward(Complex* data) const override;
    void inverse(Complex* data) const override;
    void inverse_raw(Complex* data) const override;

    [[nodiscard]] int
    g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const override;
    void reduce_g_space(Complex* data, int number_of_elements) const override;
    Complex* input_buffer() const override;
    Complex* output_buffer() const override;
    void execute_inverse_raw_on_buffers() const override;
    void execute_forward_on_buffers() const override;
    void prepare_forward_from_output() const override;

    // ── FP32 single-precision FFT ────────────────────────────────────────

    void forward_single(ComplexSingle* data) const override;
    void inverse_raw_single(ComplexSingle* data) const override;

  private:
    SerialFFTGrid serial_fft_grid_;

    // PIMPL to hide vDSP/Accelerate types from consumers
    struct VDSPState;
    std::unique_ptr<VDSPState> vdsp_state_;
};

} // namespace gpu
} // namespace q3
