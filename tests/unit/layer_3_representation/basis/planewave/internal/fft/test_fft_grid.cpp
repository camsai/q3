#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"

using namespace q3;

// ── Forward then inverse FFT is identity (up to normalization) ──────
TEST(FFTGrid, RoundTrip_RealToComplex) {
    SerialFFTGrid grid;
    grid.init(16, 16, 16);
    int N = grid.total_grid_points();

    // Create a known real-space function: cos(2π x / L)
    std::vector<Complex> data(N);
    for (int i3 = 0; i3 < 16; i3++) {
        for (int i2 = 0; i2 < 16; i2++) {
            for (int i1 = 0; i1 < 16; i1++) {
                int idx = i1 + 16 * (i2 + 16 * i3);
                Real x = (Real)i1 / 16.0;
                data[idx] = Complex(std::cos(2.0 * M_PI * x), 0.0);
            }
        }
    }

    // Save original
    std::vector<Complex> original = data;

    // Forward FFT
    grid.forward(data.data());
    // Inverse FFT
    grid.inverse(data.data());

    // Should recover original (FFTW: forward then inverse = N * original)
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(data[i].real(), original[i].real(), 1e-10) << "Mismatch at index " << i;
        EXPECT_NEAR(data[i].imag(), original[i].imag(), 1e-10) << "Imaginary part at index " << i;
    }
}

// ── Grid dimensions ─────────────────────────────────────────────────
TEST(FFTGrid, Dimensions) {
    SerialFFTGrid grid;
    grid.init(20, 20, 20);
    EXPECT_EQ(grid.grid_dimension_1(), 20);
    EXPECT_EQ(grid.grid_dimension_2(), 20);
    EXPECT_EQ(grid.grid_dimension_3(), 20);
    EXPECT_EQ(grid.total_grid_points(), 8000);
}

// ── Parseval's theorem: sum |f(r)|² = sum |F(G)|² ──────────────────
TEST(FFTGrid, ParsevalTheorem) {
    SerialFFTGrid grid;
    grid.init(12, 12, 12);
    int N = grid.total_grid_points();

    std::vector<Complex> data(N);
    // Random-ish function
    for (int i = 0; i < N; i++) {
        data[i] = Complex(std::sin(0.3 * i + 0.7), std::cos(0.5 * i - 0.2));
    }

    // Sum |f(r)|² in real space
    Real sum_r = 0.0;
    for (int i = 0; i < N; i++) {
        sum_r += std::norm(data[i]);
    }

    // Forward FFT
    grid.forward(data.data());

    // Sum |F(G)|² in G-space
    Real sum_g = 0.0;
    for (int i = 0; i < N; i++) {
        sum_g += std::norm(data[i]);
    }

    // Parseval: sum|f|² = (1/N) sum|F|²  (FFTW unnormalized forward)
    EXPECT_NEAR(sum_r, sum_g / N, 1e-6 * sum_r);
}

// ── A grid that gives every thread to each transform ────────────────

namespace {

/// A field with a value of its own at every grid point.
std::vector<Complex> patterned_field(int number_of_points) {
    std::vector<Complex> field(static_cast<std::size_t>(number_of_points));
    for (int point = 0; point < number_of_points; point++) {
        field[static_cast<std::size_t>(point)] =
            Complex(std::sin(0.37 * point + 0.2), std::cos(0.11 * point - 0.5));
    }
    return field;
}

} // namespace

TEST(FFTGrid, GivingEveryThreadToEachTransformKeepsOneSlot) {
    SerialFFTGrid grid;
    grid.give_every_thread_to_each_transform();
    grid.init(12, 10, 8);

    EXPECT_EQ(grid.concurrent_transform_slots(), 1);
}

TEST(FFTGrid, AGridOfThreadedTransformsGivesTheTransformsOfTheDefaultGrid) {
    SerialFFTGrid default_grid;
    default_grid.init(12, 10, 8);
    SerialFFTGrid threaded_grid;
    threaded_grid.give_every_thread_to_each_transform();
    threaded_grid.init(12, 10, 8);
    const std::vector<Complex> field = patterned_field(default_grid.total_grid_points());

    std::vector<Complex> default_transform = field;
    std::vector<Complex> threaded_transform = field;
    default_grid.forward(default_transform.data());
    threaded_grid.forward(threaded_transform.data());

    for (std::size_t point = 0; point < field.size(); point++) {
        EXPECT_NEAR(threaded_transform[point].real(), default_transform[point].real(), 1e-11);
        EXPECT_NEAR(threaded_transform[point].imag(), default_transform[point].imag(), 1e-11);
    }

    // And back again: the round trip returns the field.
    threaded_grid.inverse(threaded_transform.data());
    for (std::size_t point = 0; point < field.size(); point++) {
        EXPECT_NEAR(threaded_transform[point].real(), field[point].real(), 1e-11);
        EXPECT_NEAR(threaded_transform[point].imag(), field[point].imag(), 1e-11);
    }
}

TEST(FFTGrid, AGridPlannedAfterAThreadedOneKeepsItsSlotsAndItsTransforms) {
    SerialFFTGrid first_default_grid;
    first_default_grid.init(12, 10, 8);
    const int default_slots = first_default_grid.concurrent_transform_slots();
    const std::vector<Complex> field = patterned_field(first_default_grid.total_grid_points());
    std::vector<Complex> expected = field;
    first_default_grid.forward(expected.data());

    SerialFFTGrid threaded_grid;
    threaded_grid.give_every_thread_to_each_transform();
    threaded_grid.init(12, 10, 8);
    SerialFFTGrid later_default_grid;
    later_default_grid.init(12, 10, 8);

    // The planner's thread count did not carry over from the threaded grid.
    EXPECT_EQ(later_default_grid.concurrent_transform_slots(), default_slots);
    std::vector<Complex> transform = field;
    later_default_grid.forward(transform.data());
    for (std::size_t point = 0; point < field.size(); point++) {
        EXPECT_EQ(transform[point], expected[point]) << point;
    }
}

TEST(FFTGrid, AGridUnderAPlanningCapKeepsItsSlotsAndItsTransforms) {
    SerialFFTGrid default_grid;
    default_grid.init(12, 10, 8, /*use_measured_planning=*/true);
    SerialFFTGrid capped_grid;
    capped_grid.honor_planning_time_limit();
    capped_grid.limit_planning_time(0.001);
    capped_grid.init(12, 10, 8, /*use_measured_planning=*/true);
    const std::vector<Complex> field = patterned_field(default_grid.total_grid_points());

    EXPECT_EQ(capped_grid.concurrent_transform_slots(), default_grid.concurrent_transform_slots());
    std::vector<Complex> default_transform = field;
    std::vector<Complex> capped_transform = field;
    default_grid.forward(default_transform.data());
    capped_grid.forward(capped_transform.data());
    for (std::size_t point = 0; point < field.size(); point++) {
        EXPECT_NEAR(capped_transform[point].real(), default_transform[point].real(), 1e-11);
        EXPECT_NEAR(capped_transform[point].imag(), default_transform[point].imag(), 1e-11);
    }
}
