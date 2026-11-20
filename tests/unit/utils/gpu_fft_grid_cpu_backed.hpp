#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"

namespace q3::test_support {

/**
 * A device transform that runs on the host, through a serial grid of the same
 * dimensions. The batched double-precision transforms fall to the base class,
 * which loops over this grid's single transform. Single precision is refused.
 */
class CPUBackedGPUFFTGrid : public gpu::GPUFFTGrid {
  public:
    explicit CPUBackedGPUFFTGrid(
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : GPUFFTGrid(std::move(device_backend)) {}

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
    void forward_single(ComplexSingle* /*data*/) const override {
        throw std::logic_error("CPUBackedGPUFFTGrid: single precision is not implemented");
    }
    void inverse_raw_single(ComplexSingle* /*data*/) const override {
        throw std::logic_error("CPUBackedGPUFFTGrid: single precision is not implemented");
    }

  private:
    SerialFFTGrid serial_fft_grid_;
};

} // namespace q3::test_support
