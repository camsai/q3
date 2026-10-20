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
