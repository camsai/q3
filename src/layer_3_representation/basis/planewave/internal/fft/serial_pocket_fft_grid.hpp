#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <pocketfft_hdronly.h>

#include "layer_0_core/profiling/counters.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"

namespace q3 {

/**
 * SerialPocketFFTGrid — node-local 3D FFT implementation using PocketFFT.
 *
 * PocketFFT (BSD-3-Clause, header-only; the C++ library NumPy and SciPy use)
 * lets a build ship without any GPL-licensed FFT library. It has the same
 * contract as the FFTW-backed serial grid: row-major [i1][i2][i3] storage,
 * an exp(-iGr) forward transform, unnormalized forward and raw-inverse
 * transforms, and out-of-place execution from input_buffer() to
 * output_buffer().
 *
 * PocketFFT has no planning step to tune, so use_measured_planning is
 * accepted and ignored: every transform is deterministic by construction.
 * Twiddle tables are cached by the library itself (POCKETFFT_CACHE_SIZE,
 * set by the build). Threads follow OMP_NUM_THREADS, as the FFTW grid's do.
 */
class SerialPocketFFTGrid : public FFTGrid {
  public:
    void init(int dimension_1,
              int dimension_2,
              int dimension_3,
              bool /*use_measured_planning*/ = false) override {
        grid_dimension_1_ = dimension_1;
        grid_dimension_2_ = dimension_2;
        grid_dimension_3_ = dimension_3;
        const auto total_points = static_cast<std::size_t>(total_grid_points());
        work_.assign(total_points, Complex(0.0, 0.0));
        output_buffer_.assign(total_points, Complex(0.0, 0.0));

        shape_ = {static_cast<std::size_t>(dimension_1),
                  static_cast<std::size_t>(dimension_2),
                  static_cast<std::size_t>(dimension_3)};
        const auto element_bytes = static_cast<std::ptrdiff_t>(sizeof(Complex));
        strides_ = {
            element_bytes * dimension_2 * dimension_3, element_bytes * dimension_3, element_bytes};

        number_of_threads_ = 1;
        const char* omp_threads_env = std::getenv("OMP_NUM_THREADS");
        if (omp_threads_env != nullptr && std::atoi(omp_threads_env) > 1) {
            number_of_threads_ = static_cast<std::size_t>(std::atoi(omp_threads_env));
        }
    }

    [[nodiscard]] int grid_dimension_1() const override {
        return grid_dimension_1_;
    }
    [[nodiscard]] int grid_dimension_2() const override {
        return grid_dimension_2_;
    }
    [[nodiscard]] int grid_dimension_3() const override {
        return grid_dimension_3_;
    }
    [[nodiscard]] int total_grid_points() const override {
        return grid_dimension_1_ * grid_dimension_2_ * grid_dimension_3_;
    }

    void forward(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        transform(data, data, /*is_forward=*/true);
    }

    void inverse(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        transform(data, data, /*is_forward=*/false, 1.0 / total_grid_points());
    }

    void inverse_raw(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        transform(data, data, /*is_forward=*/false);
    }

    [[nodiscard]] int
    g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const override {
        int wrapped_index_1 = (miller_1 >= 0) ? miller_1 : miller_1 + grid_dimension_1_;
        int wrapped_index_2 = (miller_2 >= 0) ? miller_2 : miller_2 + grid_dimension_2_;
        int wrapped_index_3 = (miller_3 >= 0) ? miller_3 : miller_3 + grid_dimension_3_;
        return (wrapped_index_1 * grid_dimension_2_ + wrapped_index_2) * grid_dimension_3_ +
               wrapped_index_3;
    }

    [[nodiscard]] int local_grid_points() const override {
        return total_grid_points();
    }

    [[nodiscard]] int local_offset() const override {
        return 0;
    }

    void reduce_g_space(Complex* /*data*/, int /*number_of_elements*/) const override {
        // No-op for serial: data is already redundant
    }

    Complex* input_buffer() const override {
        return work_.data();
    }

    Complex* output_buffer() const override {
        return output_buffer_.data();
    }

    void execute_inverse_raw_on_buffers() const override {
        ProfilingCounters::instance().fft_call_count++;
        transform(work_.data(), output_buffer_.data(), /*is_forward=*/false);
    }

    void execute_forward_on_buffers() const override {
        ProfilingCounters::instance().fft_call_count++;
        transform(work_.data(), output_buffer_.data(), /*is_forward=*/true);
    }

    void prepare_forward_from_output() const override {
        std::memcpy(work_.data(),
                    output_buffer_.data(),
                    static_cast<std::size_t>(total_grid_points()) * sizeof(Complex));
    }

  private:
    /// One 3D complex transform over all three axes; input and output may alias.
    void transform(const Complex* input, Complex* output, bool is_forward, Real scale = 1.0) const {
        pocketfft::c2c(shape_,
                       strides_,
                       strides_,
                       axes_,
                       is_forward,
                       input,
                       output,
                       scale,
                       number_of_threads_);
    }

    int grid_dimension_1_{}, grid_dimension_2_{}, grid_dimension_3_{};
    pocketfft::shape_t shape_;
    pocketfft::stride_t strides_;
    pocketfft::shape_t axes_{0, 1, 2};
    std::size_t number_of_threads_{1};
    mutable std::vector<Complex> work_;
    mutable std::vector<Complex> output_buffer_;
};

} // namespace q3
