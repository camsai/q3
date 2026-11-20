#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_0_core/execution/gpu/device/device_mirrored_array.hpp"
#include "layer_0_core/execution/gpu/device/device_transfer_totals.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator_serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/band_projection_calculator_gpu.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/one_kpoint_quantum_system.hpp"

using namespace q3;
using execution::gpu::device::DeviceMirroredArray;
using execution::gpu::device::DeviceTransferTotals;
using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::OneKpointQuantumSystem;

namespace {

constexpr int number_of_plane_waves = 31;
constexpr int number_of_bands = 7;
constexpr int number_of_columns = 5;
constexpr std::size_t column_bytes =
    static_cast<std::size_t>(number_of_plane_waves) * number_of_columns * sizeof(Complex);
constexpr std::size_t band_block_bytes =
    static_cast<std::size_t>(number_of_plane_waves) * number_of_bands * sizeof(Complex);

/// One k-point state and a set of columns, with a device whose device is the host.
struct DeviceProjectionCase {
    KPointsGrid kpoint_grid;
    OneKpointQuantumSystem quantum_system;
    DeviceMirroredArray<Complex> columns;
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas{device_backend};
    GPUBandProjectionCalculator device_calculator{device_backend, device_blas};
    SerialBandProjectionCalculator host_calculator;

    DeviceProjectionCase()
        : quantum_system(make_grid(kpoint_grid), number_of_bands)
        , columns(static_cast<std::size_t>(number_of_plane_waves) * number_of_columns) {
        quantum_system.allocate_electronic_states({number_of_plane_waves});
        quantum_system.fill_bands_with_a_pattern();
        Complex* host_columns = columns.mutable_host();
        for (int column = 0; column < number_of_columns; ++column) {
            for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
                host_columns[static_cast<std::size_t>(column) * number_of_plane_waves +
                             plane_wave] = Complex(std::cos(0.19 * (plane_wave + 2) * (column + 1)),
                                                   std::sin(0.53 * plane_wave - 0.4 * column));
            }
        }
    }

    static const KPointsGrid& make_grid(KPointsGrid& grid) {
        grid.generate(1, 1, 1, 0, 0, 0);
        return grid;
    }
};

void expect_all_near(const std::vector<Complex>& actual,
                     const std::vector<Complex>& expected,
                     Real tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t element = 0; element < actual.size(); ++element) {
        EXPECT_NEAR(actual[element].real(), expected[element].real(), tolerance) << element;
        EXPECT_NEAR(actual[element].imag(), expected[element].imag(), tolerance) << element;
    }
}

} // namespace

TEST(GPUBandProjectionCalculator, ARunOfBandsMatchesTheHostCalculator) {
    DeviceProjectionCase projection_case;
    const std::vector<int> band_indices = {2, 3, 4, 5};
    std::vector<Complex> device_overlaps(number_of_columns * band_indices.size());
    std::vector<Complex> host_overlaps(device_overlaps.size());

    projection_case.device_calculator.project(projection_case.columns,
                                              number_of_columns,
                                              number_of_plane_waves,
                                              projection_case.quantum_system,
                                              0,
                                              band_indices,
                                              device_overlaps.data());
    projection_case.host_calculator.project(projection_case.columns,
                                            number_of_columns,
                                            number_of_plane_waves,
                                            projection_case.quantum_system,
                                            0,
                                            band_indices,
                                            host_overlaps.data());

    expect_all_near(device_overlaps, host_overlaps, 1.0e-13);
}

TEST(GPUBandProjectionCalculator, BandsWithAGapMatchTheHostCalculator) {
    DeviceProjectionCase projection_case;
    const std::vector<int> band_indices = {0, 3, 6};
    std::vector<Complex> device_overlaps(number_of_columns * band_indices.size());
    std::vector<Complex> host_overlaps(device_overlaps.size());

    projection_case.device_calculator.project(projection_case.columns,
                                              number_of_columns,
                                              number_of_plane_waves,
                                              projection_case.quantum_system,
                                              0,
                                              band_indices,
                                              device_overlaps.data());
    projection_case.host_calculator.project(projection_case.columns,
                                            number_of_columns,
                                            number_of_plane_waves,
                                            projection_case.quantum_system,
                                            0,
                                            band_indices,
                                            host_overlaps.data());

    expect_all_near(device_overlaps, host_overlaps, 1.0e-13);
}

TEST(GPUBandProjectionCalculator, ScaledBandsMatchTheHostCalculatorWithAndWithoutAGap) {
    DeviceProjectionCase projection_case;
    std::vector<Real> weights(number_of_plane_waves);
    for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
        weights[plane_wave] = 0.25 * plane_wave - 2.0;
    }
    for (const std::vector<int>& band_indices :
         {std::vector<int>{1, 2, 3}, std::vector<int>{0, 2, 5, 6}}) {
        std::vector<Complex> device_overlaps(number_of_columns * band_indices.size());
        std::vector<Complex> host_overlaps(device_overlaps.size());

        projection_case.device_calculator.project_scaled_bands(projection_case.columns,
                                                               number_of_columns,
                                                               number_of_plane_waves,
                                                               projection_case.quantum_system,
                                                               0,
                                                               band_indices,
                                                               weights.data(),
                                                               Complex(0.0, -1.0),
                                                               device_overlaps.data());
        projection_case.host_calculator.project_scaled_bands(projection_case.columns,
                                                             number_of_columns,
                                                             number_of_plane_waves,
                                                             projection_case.quantum_system,
                                                             0,
                                                             band_indices,
                                                             weights.data(),
                                                             Complex(0.0, -1.0),
                                                             host_overlaps.data());

        expect_all_near(device_overlaps, host_overlaps, 1.0e-12);
    }
}

TEST(GPUBandProjectionCalculator, ScalingLeavesTheBandsThemselvesUnchanged) {
    DeviceProjectionCase projection_case;
    std::vector<Complex> bands_before(projection_case.quantum_system.band_block(0),
                                      projection_case.quantum_system.band_block(0) +
                                          number_of_plane_waves * number_of_bands);
    std::vector<Real> weights(number_of_plane_waves, 3.0);
    const std::vector<int> band_indices = {0, 1, 2, 3, 4, 5, 6};
    std::vector<Complex> overlaps(number_of_columns * band_indices.size());

    projection_case.device_calculator.project_scaled_bands(projection_case.columns,
                                                           number_of_columns,
                                                           number_of_plane_waves,
                                                           projection_case.quantum_system,
                                                           0,
                                                           band_indices,
                                                           weights.data(),
                                                           Complex(1.0, 0.0),
                                                           overlaps.data());

    const Complex* device_block =
        projection_case.quantum_system.device_band_block(0, projection_case.device_backend);
    for (std::size_t element = 0; element < bands_before.size(); ++element) {
        EXPECT_EQ(device_block[element], bands_before[element]) << element;
    }
}

TEST(GPUBandProjectionCalculator, TheHalfSphereProductMatchesTheHostCalculator) {
    DeviceProjectionCase projection_case;
    for (const std::vector<int>& band_indices :
         {std::vector<int>{0, 1, 2, 3}, std::vector<int>{1, 4, 6}}) {
        std::vector<Real> device_overlaps(number_of_columns * band_indices.size());
        std::vector<Real> host_overlaps(device_overlaps.size());

        projection_case.device_calculator.project_half_sphere(projection_case.columns,
                                                              number_of_columns,
                                                              number_of_plane_waves,
                                                              projection_case.quantum_system,
                                                              0,
                                                              band_indices,
                                                              device_overlaps.data());
        projection_case.host_calculator.project_half_sphere(projection_case.columns,
                                                            number_of_columns,
                                                            number_of_plane_waves,
                                                            projection_case.quantum_system,
                                                            0,
                                                            band_indices,
                                                            host_overlaps.data());

        for (std::size_t element = 0; element < device_overlaps.size(); ++element) {
            EXPECT_NEAR(device_overlaps[element], host_overlaps[element], 1.0e-13) << element;
        }
    }
}

// ─── What crosses ───────────────────────────────────────────────────────────

TEST(GPUBandProjectionCalculator, BandsLeftOnTheDeviceAreNotBroughtDown) {
    DeviceProjectionCase projection_case;
    // As after a device solve: the device block is the current one.
    (void)projection_case.quantum_system.mutable_device_band_block(0,
                                                                   projection_case.device_backend);
    (void)projection_case.columns.device(projection_case.device_backend);
    const std::vector<int> band_indices = {0, 1, 2, 3};
    std::vector<Complex> overlaps(number_of_columns * band_indices.size());
    const DeviceTransferTotals before = projection_case.device_backend->transfer_totals();

    projection_case.device_calculator.project(projection_case.columns,
                                              number_of_columns,
                                              number_of_plane_waves,
                                              projection_case.quantum_system,
                                              0,
                                              band_indices,
                                              overlaps.data());

    const DeviceTransferTotals moved =
        projection_case.device_backend->transfer_totals().since(before);
    EXPECT_EQ(moved.host_to_device_copies, 0u);
    EXPECT_EQ(moved.device_to_host_copies, 1u);
    EXPECT_EQ(moved.device_to_host_bytes, overlaps.size() * sizeof(Complex));

    // The host copy of the bands was left stale: asking for it now is what downloads it.
    const DeviceTransferTotals before_host_request =
        projection_case.device_backend->transfer_totals();
    (void)projection_case.quantum_system.band_block(0);
    EXPECT_EQ(projection_case.device_backend->transfer_totals()
                  .since(before_host_request)
                  .device_to_host_bytes,
              band_block_bytes);
}

TEST(GPUBandProjectionCalculator, ColumnsAndHostBandsAreUploadedOnceForManyProducts) {
    DeviceProjectionCase projection_case;
    const std::vector<int> band_indices = {0, 1, 2, 3, 4, 5, 6};
    std::vector<Complex> overlaps(number_of_columns * band_indices.size());
    const DeviceTransferTotals before = projection_case.device_backend->transfer_totals();

    for (int repeat = 0; repeat < 3; ++repeat) {
        projection_case.device_calculator.project(projection_case.columns,
                                                  number_of_columns,
                                                  number_of_plane_waves,
                                                  projection_case.quantum_system,
                                                  0,
                                                  band_indices,
                                                  overlaps.data());
    }

    const DeviceTransferTotals moved =
        projection_case.device_backend->transfer_totals().since(before);
    EXPECT_EQ(moved.host_to_device_bytes, column_bytes + band_block_bytes);
    EXPECT_EQ(moved.host_to_device_copies, 2u);
    EXPECT_EQ(moved.device_to_host_copies, 3u);
}

TEST(GPUBandProjectionCalculator, ColumnsRewrittenOnTheHostAreUploadedAgain) {
    DeviceProjectionCase projection_case;
    const std::vector<int> band_indices = {0, 1};
    std::vector<Complex> first(number_of_columns * band_indices.size());
    std::vector<Complex> second(first.size());
    projection_case.device_calculator.project(projection_case.columns,
                                              number_of_columns,
                                              number_of_plane_waves,
                                              projection_case.quantum_system,
                                              0,
                                              band_indices,
                                              first.data());

    Complex* host_columns = projection_case.columns.mutable_host();
    for (std::size_t element = 0;
         element < static_cast<std::size_t>(number_of_plane_waves) * number_of_columns;
         ++element) {
        host_columns[element] *= 2.0;
    }
    const DeviceTransferTotals before = projection_case.device_backend->transfer_totals();
    projection_case.device_calculator.project(projection_case.columns,
                                              number_of_columns,
                                              number_of_plane_waves,
                                              projection_case.quantum_system,
                                              0,
                                              band_indices,
                                              second.data());

    const DeviceTransferTotals moved =
        projection_case.device_backend->transfer_totals().since(before);
    EXPECT_EQ(moved.host_to_device_bytes, column_bytes);
    for (std::size_t element = 0; element < first.size(); ++element) {
        EXPECT_NEAR(second[element].real(), 2.0 * first[element].real(), 1.0e-12);
        EXPECT_NEAR(second[element].imag(), 2.0 * first[element].imag(), 1.0e-12);
    }
}

TEST(GPUBandProjectionCalculator, ANullBackendIsRefused) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas(device_backend);

    EXPECT_THROW(GPUBandProjectionCalculator(nullptr, device_blas), std::invalid_argument);
}
