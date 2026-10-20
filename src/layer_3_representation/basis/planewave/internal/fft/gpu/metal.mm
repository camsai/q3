/**
 * MetalGPUFFTGrid — implementation for Apple Silicon GPU FFT.
 *
 * FP64 delegates to SerialFFTGrid (FFTW on CPU).
 * FP32 uses vDSP for native single-precision FFT when all grid
 * dimensions are powers of 2, with fallback to upcast/downcast via FFTW.
 */

#include "layer_3_representation/basis/planewave/internal/fft/gpu/metal.hpp"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace q3 {
namespace gpu {

// ── PIMPL implementation for vDSP state ──────────────────────────────────

struct MetalGPUFFTGrid::VDSPState {
    FFTSetup fft_setup = nullptr;
    int dimensions[3] = {0, 0, 0};
    vDSP_Length log2_dimensions[3] = {0, 0, 0};
    bool use_vdsp = false;

    // Pre-allocated split-complex workspace
    mutable std::vector<float> split_real;
    mutable std::vector<float> split_imag;

    ~VDSPState() {
        if (fft_setup != nullptr) {
            vDSP_destroy_fftsetup(fft_setup);
        }
    }

    void init_vdsp(int dimension_1, int dimension_2, int dimension_3) {
        dimensions[0] = dimension_1;
        dimensions[1] = dimension_2;
        dimensions[2] = dimension_3;

        use_vdsp = true;
        for (int axis = 0; axis < 3; ++axis) {
            if (dimensions[axis] <= 0 || (dimensions[axis] & (dimensions[axis] - 1)) != 0) {
                use_vdsp = false;
                return;
            }
            log2_dimensions[axis] = static_cast<vDSP_Length>(
                std::round(std::log2(static_cast<double>(dimensions[axis]))));
        }

        int max_dimension = std::max({dimensions[0], dimensions[1], dimensions[2]});
        vDSP_Length log2_max = static_cast<vDSP_Length>(
            std::round(std::log2(static_cast<double>(max_dimension))));
        fft_setup = vDSP_create_fftsetup(log2_max, FFT_RADIX2);
        if (fft_setup == nullptr) {
            use_vdsp = false;
            return;
        }

        const int total = dimension_1 * dimension_2 * dimension_3;
        split_real.resize(static_cast<size_t>(total));
        split_imag.resize(static_cast<size_t>(total));
    }

    /**
     * Perform 3D FFT using vDSP 1D FFTs along each axis.
     *
     * Decomposed as batched 1D FFTs along axes 0, 1, 2.
     * Data layout: data[i + n1*(j + n2*k)] for indices (i,j,k).
     */
    void fft_3d(ComplexSingle* data, FFTDirection direction) const {
        const int n1 = dimensions[0];
        const int n2 = dimensions[1];
        const int n3 = dimensions[2];
        const int total = n1 * n2 * n3;

        // Convert interleaved complex to split complex
        DSPSplitComplex split;
        split.realp = split_real.data();
        split.imagp = split_imag.data();
        vDSP_ctoz(reinterpret_cast<const DSPComplex*>(data), 2,
                  &split, 1, static_cast<vDSP_Length>(total));

        // Axis 0: n2*n3 transforms of length n1, stride=1
        for (int batch = 0; batch < n2 * n3; ++batch) {
            DSPSplitComplex sub;
            sub.realp = split_real.data() + batch * n1;
            sub.imagp = split_imag.data() + batch * n1;
            vDSP_fft_zip(fft_setup, &sub, 1, log2_dimensions[0], direction);
        }

        // Axis 1: n1*n3 transforms of length n2, stride=n1
        for (int k = 0; k < n3; ++k) {
            for (int i = 0; i < n1; ++i) {
                const int offset = i + k * n1 * n2;
                DSPSplitComplex sub;
                sub.realp = split_real.data() + offset;
                sub.imagp = split_imag.data() + offset;
                vDSP_fft_zip(fft_setup, &sub, static_cast<vDSP_Stride>(n1),
                             log2_dimensions[1], direction);
            }
        }

        // Axis 2: n1*n2 transforms of length n3, stride=n1*n2
        {
            const int plane = n1 * n2;
            for (int j = 0; j < n2; ++j) {
                for (int i = 0; i < n1; ++i) {
                    const int offset = i + j * n1;
                    DSPSplitComplex sub;
                    sub.realp = split_real.data() + offset;
                    sub.imagp = split_imag.data() + offset;
                    vDSP_fft_zip(fft_setup, &sub, static_cast<vDSP_Stride>(plane),
                                 log2_dimensions[2], direction);
                }
            }
        }

        // Convert split complex back to interleaved
        vDSP_ztoc(&split, 1, reinterpret_cast<DSPComplex*>(data), 2,
                  static_cast<vDSP_Length>(total));

        /**
         * vDSP inverse divides by 2 per 1D FFT call.
         * With 3 axes, result is divided by 2^3 = 8.
         * Multiply by 8 to get unnormalized inverse (matching FFTW convention).
         */
        if (direction == FFT_INVERSE) {
            float scale_factor = 8.0f;
            auto* float_data = reinterpret_cast<float*>(data);
            vDSP_vsmul(float_data, 1, &scale_factor, float_data, 1,
                       static_cast<vDSP_Length>(total * 2));
        }
    }
};

// ── MetalGPUFFTGrid implementation ───────────────────────────────────────

MetalGPUFFTGrid::MetalGPUFFTGrid(
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
    : GPUFFTGrid(std::move(device_backend)),
      vdsp_state_(std::make_unique<VDSPState>()) {}

MetalGPUFFTGrid::~MetalGPUFFTGrid() = default;

void MetalGPUFFTGrid::init(int dimension_1,
                           int dimension_2,
                           int dimension_3,
                           bool use_measured_planning) {
    serial_fft_grid_.init(dimension_1, dimension_2, dimension_3, use_measured_planning);
    vdsp_state_->init_vdsp(dimension_1, dimension_2, dimension_3);
}

int MetalGPUFFTGrid::grid_dimension_1() const {
    return serial_fft_grid_.grid_dimension_1();
}
int MetalGPUFFTGrid::grid_dimension_2() const {
    return serial_fft_grid_.grid_dimension_2();
}
int MetalGPUFFTGrid::grid_dimension_3() const {
    return serial_fft_grid_.grid_dimension_3();
}
int MetalGPUFFTGrid::total_grid_points() const {
    return serial_fft_grid_.total_grid_points();
}
int MetalGPUFFTGrid::local_grid_points() const {
    return serial_fft_grid_.local_grid_points();
}
int MetalGPUFFTGrid::local_offset() const {
    return serial_fft_grid_.local_offset();
}

void MetalGPUFFTGrid::forward(Complex* data) const {
    serial_fft_grid_.forward(data);
}
void MetalGPUFFTGrid::inverse(Complex* data) const {
    serial_fft_grid_.inverse(data);
}
void MetalGPUFFTGrid::inverse_raw(Complex* data) const {
    serial_fft_grid_.inverse_raw(data);
}

int MetalGPUFFTGrid::g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const {
    return serial_fft_grid_.g_vector_to_fft_grid_index(miller_1, miller_2, miller_3);
}
void MetalGPUFFTGrid::reduce_g_space(Complex* data, int number_of_elements) const {
    serial_fft_grid_.reduce_g_space(data, number_of_elements);
}
Complex* MetalGPUFFTGrid::input_buffer() const {
    return serial_fft_grid_.input_buffer();
}
Complex* MetalGPUFFTGrid::output_buffer() const {
    return serial_fft_grid_.output_buffer();
}
void MetalGPUFFTGrid::execute_inverse_raw_on_buffers() const {
    serial_fft_grid_.execute_inverse_raw_on_buffers();
}
void MetalGPUFFTGrid::execute_forward_on_buffers() const {
    serial_fft_grid_.execute_forward_on_buffers();
}
void MetalGPUFFTGrid::prepare_forward_from_output() const {
    serial_fft_grid_.prepare_forward_from_output();
}

void MetalGPUFFTGrid::forward_single(ComplexSingle* data) const {
    if (vdsp_state_->use_vdsp) {
        vdsp_state_->fft_3d(data, FFT_FORWARD);
    } else {
        // Fallback: upcast to FP64 via FFTW
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
}

void MetalGPUFFTGrid::inverse_raw_single(ComplexSingle* data) const {
    if (vdsp_state_->use_vdsp) {
        vdsp_state_->fft_3d(data, FFT_INVERSE);
    } else {
        // Fallback: upcast to FP64 via FFTW
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
}

} // namespace gpu
} // namespace q3
