#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "layer_0_core/execution/gpu/device/device_mirrored_array.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator_serial.hpp"
#include "utils/one_kpoint_quantum_system.hpp"

using namespace q3;
using execution::gpu::device::DeviceMirroredArray;
using test_support::OneKpointQuantumSystem;

namespace {

constexpr int number_of_plane_waves = 29;
constexpr int number_of_bands = 6;
constexpr int number_of_columns = 4;

/// Columns and bands with a value of their own in every element, and the sums done directly.
struct ProjectionCase {
    KPointsGrid kpoint_grid;
    OneKpointQuantumSystem quantum_system;
    DeviceMirroredArray<Complex> columns;

    ProjectionCase()
        : quantum_system(make_grid(kpoint_grid), number_of_bands)
        , columns(static_cast<std::size_t>(number_of_plane_waves) * number_of_columns) {
        quantum_system.allocate_electronic_states({number_of_plane_waves});
        quantum_system.fill_bands_with_a_pattern();
        Complex* host_columns = columns.mutable_host();
        for (int column = 0; column < number_of_columns; ++column) {
            for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
                host_columns[static_cast<std::size_t>(column) * number_of_plane_waves +
                             plane_wave] = Complex(std::cos(0.23 * (plane_wave + 1) * (column + 2)),
                                                   std::sin(0.41 * plane_wave + 0.9 * column));
            }
        }
    }

    static const KPointsGrid& make_grid(KPointsGrid& grid) {
        grid.generate(1, 1, 1, 0, 0, 0);
        return grid;
    }

    [[nodiscard]] Complex column(int plane_wave, int column_index) const {
        return columns
            .host()[static_cast<std::size_t>(column_index) * number_of_plane_waves + plane_wave];
    }
    [[nodiscard]] Complex band(int plane_wave, int band_index) const {
        return quantum_system.wavefunction_band_data(0, band_index)[plane_wave];
    }
    [[nodiscard]] Complex direct_overlap(int column_index, int band_index) const {
        Complex sum(0.0, 0.0);
        for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
            sum += std::conj(column(plane_wave, column_index)) * band(plane_wave, band_index);
        }
        return sum;
    }
};

void expect_complex_near(Complex actual, Complex expected, Real tolerance) {
    EXPECT_NEAR(actual.real(), expected.real(), tolerance);
    EXPECT_NEAR(actual.imag(), expected.imag(), tolerance);
}

} // namespace

TEST(SerialBandProjectionCalculator, ARunOfBandsGivesTheDirectSums) {
    ProjectionCase projection_case;
    const std::vector<int> band_indices = {1, 2, 3, 4};
    std::vector<Complex> overlaps(static_cast<std::size_t>(number_of_columns) *
                                  band_indices.size());

    SerialBandProjectionCalculator().project(projection_case.columns,
                                             number_of_columns,
                                             number_of_plane_waves,
                                             projection_case.quantum_system,
                                             0,
                                             band_indices,
                                             overlaps.data());

    for (std::size_t position = 0; position < band_indices.size(); ++position) {
        for (int column = 0; column < number_of_columns; ++column) {
            expect_complex_near(overlaps[column + position * number_of_columns],
                                projection_case.direct_overlap(column, band_indices[position]),
                                1.0e-12);
        }
    }
}

TEST(SerialBandProjectionCalculator, BandsWithAGapAreGatheredInTheOrderGiven) {
    ProjectionCase projection_case;
    const std::vector<int> band_indices = {0, 2, 5};
    std::vector<Complex> overlaps(static_cast<std::size_t>(number_of_columns) *
                                  band_indices.size());

    SerialBandProjectionCalculator().project(projection_case.columns,
                                             number_of_columns,
                                             number_of_plane_waves,
                                             projection_case.quantum_system,
                                             0,
                                             band_indices,
                                             overlaps.data());

    for (std::size_t position = 0; position < band_indices.size(); ++position) {
        for (int column = 0; column < number_of_columns; ++column) {
            expect_complex_near(overlaps[column + position * number_of_columns],
                                projection_case.direct_overlap(column, band_indices[position]),
                                1.0e-12);
        }
    }
}

TEST(SerialBandProjectionCalculator, OnlyTheLeadingColumnsAskedForAreUsed) {
    ProjectionCase projection_case;
    const std::vector<int> band_indices = {0, 1};
    const int leading_columns = 2;
    std::vector<Complex> overlaps(static_cast<std::size_t>(leading_columns) * band_indices.size());

    SerialBandProjectionCalculator().project(projection_case.columns,
                                             leading_columns,
                                             number_of_plane_waves,
                                             projection_case.quantum_system,
                                             0,
                                             band_indices,
                                             overlaps.data());

    for (std::size_t position = 0; position < band_indices.size(); ++position) {
        for (int column = 0; column < leading_columns; ++column) {
            expect_complex_near(overlaps[column + position * leading_columns],
                                projection_case.direct_overlap(column, band_indices[position]),
                                1.0e-12);
        }
    }
}

TEST(SerialBandProjectionCalculator, ScaledBandsCarryTheWeightAndThePrefactor) {
    ProjectionCase projection_case;
    const std::vector<int> band_indices = {1, 2, 4};
    std::vector<Real> weights(number_of_plane_waves);
    for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
        weights[plane_wave] = 0.5 * plane_wave - 3.0;
    }
    const Complex prefactor(0.0, -1.0);
    std::vector<Complex> overlaps(static_cast<std::size_t>(number_of_columns) *
                                  band_indices.size());

    SerialBandProjectionCalculator().project_scaled_bands(projection_case.columns,
                                                          number_of_columns,
                                                          number_of_plane_waves,
                                                          projection_case.quantum_system,
                                                          0,
                                                          band_indices,
                                                          weights.data(),
                                                          prefactor,
                                                          overlaps.data());

    for (std::size_t position = 0; position < band_indices.size(); ++position) {
        for (int column = 0; column < number_of_columns; ++column) {
            Complex expected(0.0, 0.0);
            for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
                expected += std::conj(projection_case.column(plane_wave, column)) * prefactor *
                            weights[plane_wave] *
                            projection_case.band(plane_wave, band_indices[position]);
            }
            expect_complex_near(overlaps[column + position * number_of_columns], expected, 1.0e-11);
        }
    }
}

TEST(SerialBandProjectionCalculator, TheHalfSphereProductIsTwiceTheRealPart) {
    ProjectionCase projection_case;
    const std::vector<int> band_indices = {0, 1, 2, 3, 4, 5};
    std::vector<Real> overlaps(static_cast<std::size_t>(number_of_columns) * band_indices.size());

    SerialBandProjectionCalculator().project_half_sphere(projection_case.columns,
                                                         number_of_columns,
                                                         number_of_plane_waves,
                                                         projection_case.quantum_system,
                                                         0,
                                                         band_indices,
                                                         overlaps.data());

    for (std::size_t position = 0; position < band_indices.size(); ++position) {
        for (int column = 0; column < number_of_columns; ++column) {
            EXPECT_NEAR(overlaps[column + position * number_of_columns],
                        2.0 * projection_case.direct_overlap(column, band_indices[position]).real(),
                        1.0e-12);
        }
    }
}

TEST(SerialBandProjectionCalculator, NoBandsOrNoColumnsWritesNothing) {
    ProjectionCase projection_case;
    std::vector<Complex> overlaps(4, Complex(7.0, 7.0));

    SerialBandProjectionCalculator().project(projection_case.columns,
                                             number_of_columns,
                                             number_of_plane_waves,
                                             projection_case.quantum_system,
                                             0,
                                             {},
                                             overlaps.data());
    SerialBandProjectionCalculator().project(projection_case.columns,
                                             0,
                                             number_of_plane_waves,
                                             projection_case.quantum_system,
                                             0,
                                             {0, 1},
                                             overlaps.data());

    for (const Complex& overlap : overlaps) {
        EXPECT_EQ(overlap, Complex(7.0, 7.0));
    }
}
