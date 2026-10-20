/**
 * SYCLGPUFFTGrid — GPUFFTGrid implementation for SYCL/oneAPI.
 *
 * Delegates all FFT operations to a composed SerialFFTGrid (FFTW on CPU).
 * This is the safe initial implementation — identical to the Metal backend
 * approach. On Intel CPUs, oneMKL internally uses FFTW anyway, so there
 * is no performance penalty for the CPU fallback path.
 *
 * FP64 FFTs delegate directly to SerialFFTGrid.
 * FP32 FFTs use the upcast-to-FP64 / downcast pattern (same as Metal fallback).
 *
 * oneMKL DFT descriptors can be added later as an optimization for native
 * GPU FFT on Intel Data Center GPUs.
 */

#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"

#include <vector>

namespace q3 {
namespace gpu {

class SYCLGPUFFTGrid : public GPUFFTGrid {
  public:
    /**
     * Construct a SYCL FFT grid backed by the given device.
     *
     * All FFT operations are delegated to the internal SerialFFTGrid,
     * so no SYCL-specific resources are created here.
     */
    explicit SYCLGPUFFTGrid(
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : GPUFFTGrid(std::move(device_backend)) {}

    ~SYCLGPUFFTGrid() override = default;

    // ── FFTGrid base interface (delegated to serial_fft_grid_) ───────────

    void init(int dimension_1,
              int dimension_2,
              int dimension_3,
              bool use_measured_planning = false) override {
        serial_fft_grid_.init(dimension_1, dimension_2, dimension_3, use_measured_planning);
    }

    [[nodiscard]] int grid_dimension_1() const override {
        return serial_fft_grid_.grid_dimension_1();
    }
    [[nodiscard]] int grid_dimension_2() const override {
        return serial_fft_grid_.grid_dimension_2();
    }
    [[nodiscard]] int grid_dimension_3() const override {
        return serial_fft_grid_.grid_dimension_3();
    }
    [[nodiscard]] int total_grid_points() const override {
        return serial_fft_grid_.total_grid_points();
    }
    [[nodiscard]] int local_grid_points() const override {
        return serial_fft_grid_.local_grid_points();
    }
    [[nodiscard]] int local_offset() const override {
        return serial_fft_grid_.local_offset();
    }

    void forward(Complex* data) const override {
        serial_fft_grid_.forward(data);
    }
    void inverse(Complex* data) const override {
        serial_fft_grid_.inverse(data);
    }
    void inverse_raw(Complex* data) const override {
        serial_fft_grid_.inverse_raw(data);
    }

    [[nodiscard]] int
    g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const override {
        return serial_fft_grid_.g_vector_to_fft_grid_index(miller_1, miller_2, miller_3);
    }

    void reduce_g_space(Complex* data, int number_of_elements) const override {
        serial_fft_grid_.reduce_g_space(data, number_of_elements);
    }

    Complex* input_buffer() const override {
        return serial_fft_grid_.input_buffer();
    }
    Complex* output_buffer() const override {
        return serial_fft_grid_.output_buffer();
    }

    void execute_inverse_raw_on_buffers() const override {
        serial_fft_grid_.execute_inverse_raw_on_buffers();
    }
    void execute_forward_on_buffers() const override {
        serial_fft_grid_.execute_forward_on_buffers();
    }
    void prepare_forward_from_output() const override {
        serial_fft_grid_.prepare_forward_from_output();
    }

    // ── FP32 FFT (upcast to FP64 via SerialFFTGrid) ─────────────────────

    /**
     * Forward FP32 FFT via upcast to FP64 + SerialFFTGrid + downcast.
     *
     * This matches the Metal backend fallback path. oneMKL DFT can be
     * added later for native FP32 GPU FFT on Intel hardware.
     */
    void forward_single(ComplexSingle* data) const override {
        const int grid_points = total_grid_points();
        std::vector<Complex> temporary_buffer(static_cast<size_t>(grid_points));

        for (int point_index = 0; point_index < grid_points; ++point_index) {
            temporary_buffer[static_cast<size_t>(point_index)] =
                Complex(static_cast<double>(data[point_index].real()),
                        static_cast<double>(data[point_index].imag()));
        }

        serial_fft_grid_.forward(temporary_buffer.data());

        for (int point_index = 0; point_index < grid_points; ++point_index) {
            data[point_index] = ComplexSingle(
                static_cast<float>(temporary_buffer[static_cast<size_t>(point_index)].real()),
                static_cast<float>(temporary_buffer[static_cast<size_t>(point_index)].imag()));
        }
    }

    /**
     * Inverse FP32 FFT (raw, unnormalized) via upcast to FP64 + SerialFFTGrid + downcast.
     */
    void inverse_raw_single(ComplexSingle* data) const override {
        const int grid_points = total_grid_points();
        std::vector<Complex> temporary_buffer(static_cast<size_t>(grid_points));

        for (int point_index = 0; point_index < grid_points; ++point_index) {
            temporary_buffer[static_cast<size_t>(point_index)] =
                Complex(static_cast<double>(data[point_index].real()),
                        static_cast<double>(data[point_index].imag()));
        }

        serial_fft_grid_.inverse_raw(temporary_buffer.data());

        for (int point_index = 0; point_index < grid_points; ++point_index) {
            data[point_index] = ComplexSingle(
                static_cast<float>(temporary_buffer[static_cast<size_t>(point_index)].real()),
                static_cast<float>(temporary_buffer[static_cast<size_t>(point_index)].imag()));
        }
    }

    // ── FP64 FFT (delegate to SerialFFTGrid) ─────────────────────────────

    /**
     * Forward FP64 FFT delegates to SerialFFTGrid (FFTW).
     */
    void forward_transform(Complex* data) const override {
        serial_fft_grid_.forward(data);
    }

    /**
     * Inverse FP64 FFT (raw, unnormalized) delegates to SerialFFTGrid (FFTW).
     */
    void inverse_transform(Complex* data) const override {
        serial_fft_grid_.inverse_raw(data);
    }

  private:
    SerialFFTGrid serial_fft_grid_;
};

std::unique_ptr<GPUFFTGrid>
create_sycl_gpu_fft_grid(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend) {
    return std::make_unique<SYCLGPUFFTGrid>(std::move(device_backend));
}

} // namespace gpu
} // namespace q3
