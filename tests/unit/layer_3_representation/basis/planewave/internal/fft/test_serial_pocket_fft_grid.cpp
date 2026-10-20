#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/serial_pocket_fft_grid.hpp"

using namespace q3;

namespace {

std::vector<Complex> random_grid_values(int number_of_points, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<Real> distribution(-1.0, 1.0);
    std::vector<Complex> values(number_of_points);
    for (auto& value : values) {
        value = Complex(distribution(generator), distribution(generator));
    }
    return values;
}

Real max_abs_difference(const std::vector<Complex>& left, const std::vector<Complex>& right) {
    Real largest = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        largest = std::max(largest, std::abs(left[index] - right[index]));
    }
    return largest;
}

/// Even, mixed-radix and prime extents: PocketFFT takes a different code path for each.
const std::vector<std::vector<int>> kGridShapes = {
    {8, 8, 8}, {15, 18, 20}, {7, 11, 13}, {24, 24, 30}};

} // namespace

TEST(SerialPocketFFTGrid, RoundTripRecoversInput) {
    for (const auto& shape : kGridShapes) {
        SerialPocketFFTGrid grid;
        grid.init(shape[0], shape[1], shape[2]);
        const auto original = random_grid_values(grid.total_grid_points(), 7U);
        auto data = original;
        grid.forward(data.data());
        grid.inverse(data.data());
        EXPECT_LT(max_abs_difference(data, original), 1e-13);
    }
}

TEST(SerialPocketFFTGrid, PlaneWaveTransformsToSingleCoefficient) {
    const int dimension_1 = 12, dimension_2 = 10, dimension_3 = 9;
    SerialPocketFFTGrid grid;
    grid.init(dimension_1, dimension_2, dimension_3);
    const int miller_1 = 2, miller_2 = -3, miller_3 = 4;
    std::vector<Complex> data(grid.total_grid_points());
    for (int i1 = 0; i1 < dimension_1; ++i1) {
        for (int i2 = 0; i2 < dimension_2; ++i2) {
            for (int i3 = 0; i3 < dimension_3; ++i3) {
                const Real phase =
                    2.0 * M_PI *
                    (Real(miller_1 * i1) / dimension_1 + Real(miller_2 * i2) / dimension_2 +
                     Real(miller_3 * i3) / dimension_3);
                data[(i1 * dimension_2 + i2) * dimension_3 + i3] = std::polar(1.0, phase);
            }
        }
    }
    // Forward is exp(-iGr), so exp(+iGr) lands entirely on G with weight N.
    grid.forward(data.data());
    const int peak = grid.g_vector_to_fft_grid_index(miller_1, miller_2, miller_3);
    for (int index = 0; index < grid.total_grid_points(); ++index) {
        const Complex expected =
            (index == peak) ? Complex(grid.total_grid_points(), 0.0) : Complex(0.0, 0.0);
        EXPECT_LT(std::abs(data[index] - expected), 1e-10) << "index " << index;
    }
}

TEST(SerialPocketFFTGrid, BufferTransformsMatchInPlaceTransforms) {
    SerialPocketFFTGrid grid;
    grid.init(15, 18, 20);
    const int number_of_points = grid.total_grid_points();
    const auto original = random_grid_values(number_of_points, 11U);

    std::copy(original.begin(), original.end(), grid.input_buffer());
    grid.execute_inverse_raw_on_buffers();
    std::vector<Complex> buffered(grid.output_buffer(), grid.output_buffer() + number_of_points);
    auto in_place = original;
    grid.inverse_raw(in_place.data());
    EXPECT_LT(max_abs_difference(buffered, in_place), 1e-12);

    grid.prepare_forward_from_output();
    grid.execute_forward_on_buffers();
    std::vector<Complex> round_trip(grid.output_buffer(), grid.output_buffer() + number_of_points);
    for (auto& value : round_trip) {
        value /= static_cast<Real>(number_of_points);
    }
    EXPECT_LT(max_abs_difference(round_trip, original), 1e-13);
}

#ifndef Q3_FFT_BACKEND_POCKETFFT
/**
 * With FFTW as the build's backend, SerialFFTGrid is the FFTW grid, so the two
 * libraries can be run on identical input. They are not bit-identical; the
 * bound is round-off relative to the largest coefficient, which is O(N).
 */
TEST(SerialPocketFFTGrid, AgreesWithFFTWToRoundOff) {
    for (const auto& shape : kGridShapes) {
        SerialPocketFFTGrid pocketfft_grid;
        SerialFFTGrid fftw_grid;
        pocketfft_grid.init(shape[0], shape[1], shape[2]);
        fftw_grid.init(shape[0], shape[1], shape[2]);
        const int number_of_points = pocketfft_grid.total_grid_points();
        const auto original = random_grid_values(number_of_points, 23U);

        auto pocketfft_forward = original;
        auto fftw_forward = original;
        pocketfft_grid.forward(pocketfft_forward.data());
        fftw_grid.forward(fftw_forward.data());
        EXPECT_LT(max_abs_difference(pocketfft_forward, fftw_forward), 1e-14 * number_of_points);

        auto pocketfft_inverse = original;
        auto fftw_inverse = original;
        pocketfft_grid.inverse(pocketfft_inverse.data());
        fftw_grid.inverse(fftw_inverse.data());
        EXPECT_LT(max_abs_difference(pocketfft_inverse, fftw_inverse), 1e-15);
    }
}
#endif
