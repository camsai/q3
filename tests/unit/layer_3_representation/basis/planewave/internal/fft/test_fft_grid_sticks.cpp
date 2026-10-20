#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_sticks.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

using namespace q3;

namespace {

int world_size() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

/// Deterministic pseudo-random value for a global grid point, identical on every rank.
Complex sample(int i1, int i2, int i3) {
    std::uint32_t seed =
        static_cast<std::uint32_t>((i1 * 73856093) ^ (i2 * 19349663) ^ (i3 * 83492791));
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    return Complex(uniform(generator), uniform(generator));
}

/// Fill this rank's stick buffer from the global sample, so every rank agrees on the field.
void fill_sticks(const FFTGridSticks& grid, Complex* sticks) {
    int d1 = grid.grid_dimension_1(), d2 = grid.grid_dimension_2(), d3 = grid.grid_dimension_3();
    for (int i1 = 0; i1 < d1; ++i1) {
        for (int i2 = 0; i2 < d2; ++i2) {
            for (int i3 = 0; i3 < d3; ++i3) {
                int index = grid.g_vector_to_fft_grid_index(i1, i2, i3);
                if (index >= 0) {
                    sticks[index] = sample(i1, i2, i3);
                }
            }
        }
    }
}

} // namespace

// ─── Round trip ─────────────────────────────────────────────────────────────

/**
 * Inverse then forward returns the input scaled by the point count, at every
 * rank count, on a grid whose dimensions are not divisible by the rank count.
 * This is the gate that has to hold before any physics runs through the class.
 */
TEST(FFTGridSticks, RoundTripIsIdentityUpToNormalisation) {
    FFTGridSticks grid(MPI_COMM_WORLD);
    grid.init(15, 14, 13);
    std::vector<Complex> data(
        static_cast<std::size_t>(std::max(grid.local_g_space_points(), grid.local_grid_points())));
    fill_sticks(grid, data.data());
    std::vector<Complex> original(data.begin(), data.begin() + grid.local_g_space_points());

    grid.inverse_raw(data.data());
    grid.forward(data.data());

    double scale = static_cast<double>(grid.total_grid_points());
    double worst = 0.0;
    for (int index = 0; index < grid.local_g_space_points(); ++index) {
        worst = std::max(worst,
                         std::abs(data[static_cast<std::size_t>(index)] / scale -
                                  original[static_cast<std::size_t>(index)]));
    }
    EXPECT_LT(worst, 1e-12) << "ranks=" << world_size();
}

// ─── Agreement with the serial transform ────────────────────────────────────

/**
 * The distributed transform must produce, plane for plane, what the serial
 * FFTW transform produces on the whole grid: same field in, same field out,
 * compared on the planes each rank holds.
 */
TEST(FFTGridSticks, MatchesTheSerialTransformOnLocalPlanes) {
    const int d1 = 12, d2 = 10, d3 = 11;
    FFTGridSticks grid(MPI_COMM_WORLD);
    grid.init(d1, d2, d3);
    std::vector<Complex> data(
        static_cast<std::size_t>(std::max(grid.local_g_space_points(), grid.local_grid_points())));
    fill_sticks(grid, data.data());
    grid.inverse_raw(data.data());

    SerialFFTGrid serial;
    serial.init(d1, d2, d3);
    std::vector<Complex> whole(static_cast<std::size_t>(d1 * d2 * d3));
    for (int i1 = 0; i1 < d1; ++i1)
        for (int i2 = 0; i2 < d2; ++i2)
            for (int i3 = 0; i3 < d3; ++i3)
                whole[static_cast<std::size_t>(serial.g_vector_to_fft_grid_index(i1, i2, i3))] =
                    sample(i1, i2, i3);
    serial.inverse_raw(whole.data());

    /**
     * The stick grid's real space is planes along the third dimension, laid
     * out [plane][i1][i2]; the serial grid is [i1][i2][i3]. Compare point by
     * point through both index conventions.
     */
    double worst = 0.0;
    for (int plane = 0; plane < grid.local_grid_points() / (d1 * d2); ++plane) {
        int i3 = grid.local_offset() + plane;
        for (int i1 = 0; i1 < d1; ++i1) {
            for (int i2 = 0; i2 < d2; ++i2) {
                Complex distributed = data[static_cast<std::size_t>(plane) * d1 * d2 +
                                           static_cast<std::size_t>(i1) * d2 + i2];
                Complex reference =
                    whole[static_cast<std::size_t>(serial.g_vector_to_fft_grid_index(i1, i2, i3))];
                worst = std::max(worst, std::abs(distributed - reference));
            }
        }
    }
    EXPECT_LT(worst, 1e-11) << "ranks=" << world_size();
}

// ─── Where a local real-space point sits ────────────────────────────────────

/**
 * global_real_space_index() names, for every point this rank holds in real
 * space, the serial grid's index of the same point, so the distributed field
 * read through it is the serial field. Code that needs a point's position --
 * the isolated-system corrections -- used to add local_offset() planes to the
 * local index instead, which reads the stick layout as a slab along the first
 * dimension. The serial grid's own answer is the index itself.
 */
TEST(FFTGridSticks, GlobalRealSpaceIndexNamesTheSerialPoint) {
    const int d1 = 12, d2 = 10, d3 = 11;
    FFTGridSticks grid(MPI_COMM_WORLD);
    grid.init(d1, d2, d3);
    std::vector<Complex> data(
        static_cast<std::size_t>(std::max(grid.local_g_space_points(), grid.local_grid_points())));
    fill_sticks(grid, data.data());
    grid.inverse_raw(data.data());

    SerialFFTGrid serial;
    serial.init(d1, d2, d3);
    std::vector<Complex> whole(static_cast<std::size_t>(d1 * d2 * d3));
    for (int i1 = 0; i1 < d1; ++i1)
        for (int i2 = 0; i2 < d2; ++i2)
            for (int i3 = 0; i3 < d3; ++i3)
                whole[static_cast<std::size_t>(serial.g_vector_to_fft_grid_index(i1, i2, i3))] =
                    sample(i1, i2, i3);
    serial.inverse_raw(whole.data());

    double worst = 0.0;
    for (int local_index = 0; local_index < grid.local_grid_points(); ++local_index) {
        int global_index = grid.global_real_space_index(local_index);
        ASSERT_GE(global_index, 0);
        ASSERT_LT(global_index, d1 * d2 * d3);
        worst = std::max(worst,
                         std::abs(data[static_cast<std::size_t>(local_index)] -
                                  whole[static_cast<std::size_t>(global_index)]));
    }
    EXPECT_LT(worst, 1e-11) << "ranks=" << world_size();

    for (int index = 0; index < serial.total_grid_points(); ++index) {
        ASSERT_EQ(serial.global_real_space_index(index), index);
    }
}

// ─── Parseval ───────────────────────────────────────────────────────────────

/**
 * Sum of |f(r)|^2 over all ranks equals the sum of |F(G)|^2 times the point
 * count. Catches a redistribution that drops or duplicates a stick, which the
 * round trip can miss if the same stick is lost both ways.
 */
TEST(FFTGridSticks, ParsevalHoldsAcrossRanks) {
    FFTGridSticks grid(MPI_COMM_WORLD);
    grid.init(9, 10, 11);
    std::vector<Complex> data(
        static_cast<std::size_t>(std::max(grid.local_g_space_points(), grid.local_grid_points())));
    fill_sticks(grid, data.data());

    double g_space = 0.0;
    for (int index = 0; index < grid.local_g_space_points(); ++index) {
        g_space += std::norm(data[static_cast<std::size_t>(index)]);
    }
    grid.inverse_raw(data.data());
    double real_space = 0.0;
    for (int index = 0; index < grid.local_grid_points(); ++index) {
        real_space += std::norm(data[static_cast<std::size_t>(index)]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &g_space, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &real_space, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double expected = g_space * grid.total_grid_points();
    EXPECT_NEAR(real_space, expected, 1e-9 * expected) << "ranks=" << world_size();
}

// ─── Ownership ──────────────────────────────────────────────────────────────

/**
 * Every stick is owned by exactly one rank, and a stick and its mirror by the
 * same one -- the property that lets gamma-only survive distribution.
 */
TEST(FFTGridSticks, EveryStickHasExactlyOneOwnerAndMirrorsAgree) {
    const int d1 = 13, d2 = 12, d3 = 8;
    FFTGridSticks grid(MPI_COMM_WORLD);
    grid.init(d1, d2, d3);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i1 = 0; i1 < d1; ++i1) {
        for (int i2 = 0; i2 < d2; ++i2) {
            int owned_here = grid.owns_stick(i1, i2) ? 1 : 0;
            int owners = owned_here;
            MPI_Allreduce(MPI_IN_PLACE, &owners, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            EXPECT_EQ(owners, 1) << "stick (" << i1 << "," << i2 << ")";
            EXPECT_EQ(grid.owns_stick(i1, i2), grid.owns_stick(-i1, -i2));
        }
    }
}

/**
 * The cutoff: sticks given zero weight are transported by nobody, which the
 * zero-copy path relies on to stay cheap. Their real-space contribution is
 * exactly zero and everything else is unchanged.
 */
TEST(FFTGridSticks, UnweightedSticksAreDroppedAndTheRestUntouched) {
    const int d1 = 10, d2 = 10, d3 = 10;
    FFTGridSticks full(MPI_COMM_WORLD);
    full.init(d1, d2, d3);
    FFTGridSticks cut(MPI_COMM_WORLD);
    cut.init(d1, d2, d3);
    std::vector<int> weights(static_cast<std::size_t>(d1 * d2), 1);
    weights[static_cast<std::size_t>(3 * d2 + 4)] = 0; // drop one stick and its mirror
    weights[static_cast<std::size_t>(((d1 - 3) % d1) * d2 + (d2 - 4) % d2)] = 0;
    cut.assign_sticks(weights);

    auto run = [&](const FFTGridSticks& grid, bool zero_dropped) {
        std::vector<Complex> data(static_cast<std::size_t>(
            std::max(grid.local_g_space_points(), grid.local_grid_points())));
        for (int i1 = 0; i1 < d1; ++i1)
            for (int i2 = 0; i2 < d2; ++i2)
                for (int i3 = 0; i3 < d3; ++i3) {
                    int index = grid.g_vector_to_fft_grid_index(i1, i2, i3);
                    if (index < 0)
                        continue;
                    bool dropped =
                        (i1 == 3 && i2 == 4) || (i1 == (d1 - 3) % d1 && i2 == (d2 - 4) % d2);
                    data[static_cast<std::size_t>(index)] =
                        (zero_dropped && dropped) ? Complex(0.0, 0.0) : sample(i1, i2, i3);
                }
        grid.inverse_raw(data.data());
        return data;
    };
    auto with_zeroed = run(full, true);
    auto with_dropped = run(cut, false);
    double worst = 0.0;
    for (int index = 0; index < full.local_grid_points(); ++index) {
        worst = std::max(worst,
                         std::abs(with_zeroed[static_cast<std::size_t>(index)] -
                                  with_dropped[static_cast<std::size_t>(index)]));
    }
    EXPECT_LT(worst, 1e-12);
}
