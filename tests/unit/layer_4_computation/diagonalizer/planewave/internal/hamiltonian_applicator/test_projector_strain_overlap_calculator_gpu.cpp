#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_0_core/execution/gpu/device/device_transfer_totals.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator_serial.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_strain_overlap_calculator_serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/projector_strain_overlap_calculator_gpu.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/one_kpoint_quantum_system.hpp"
#include "utils/projector_strain_test_inputs.hpp"
#include "utils/two_species_projector_system.hpp"

using namespace q3;
using execution::gpu::device::DeviceTransferTotals;
using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::OneKpointQuantumSystem;
using test_support::ProjectorStrainTestInputs;
using test_support::TwoSpeciesProjectorSystem;

namespace {

constexpr int number_of_bands = 6;
constexpr int number_of_kinds = ProjectorStrainOverlapCalculator::number_of_kinds;
// Species A carries 4 projectors per atom and species B 18; the atoms are A, B, A, B.
constexpr int projectors_of_all_atoms = 4 + 18 + 4 + 18;

/// The two-species cell with bands over its k-point, and both strain overlap calculators.
struct DeviceStrainCase {
    TwoSpeciesProjectorSystem system;
    ProjectorStrainTestInputs strain_inputs{system.crystal_structure,
                                            system.planewave_basis,
                                            system.kpoint,
                                            system.g_vector_indices,
                                            /*maximum_angular_momentum=*/2};
    KPointsGrid kpoint_grid;
    OneKpointQuantumSystem quantum_system;
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas{device_backend};
    SerialBandProjectionCalculator host_band_projections;
    SerialProjectorStrainOverlapCalculator host_calculator{system.crystal_structure,
                                                           system.planewave_basis,
                                                           system.pseudopotential_basis(),
                                                           host_band_projections};
    GPUProjectorStrainOverlapCalculator device_calculator{system.crystal_structure,
                                                          system.planewave_basis,
                                                          system.pseudopotential_basis(),
                                                          device_backend,
                                                          device_blas};

    DeviceStrainCase() : quantum_system(make_grid(kpoint_grid), number_of_bands) {
        quantum_system.allocate_electronic_states({system.number_of_plane_waves()});
        quantum_system.fill_bands_with_a_pattern();
        host_calculator.begin_kpoint(strain_inputs.inputs(), projectors_of_all_atoms);
        device_calculator.begin_kpoint(strain_inputs.inputs(), projectors_of_all_atoms);
    }

    static const KPointsGrid& make_grid(KPointsGrid& grid) {
        grid.generate(1, 1, 1, 0, 0, 0);
        return grid;
    }

    void expect_block_matches(int first_atom,
                              int end_atom,
                              int projectors,
                              const std::vector<int>& band_indices) {
        const std::size_t number_of_overlaps =
            static_cast<std::size_t>(number_of_kinds) * projectors * band_indices.size();
        std::vector<Complex> host_overlaps(number_of_overlaps);
        std::vector<Complex> device_overlaps(number_of_overlaps);
        host_calculator.compute_block(first_atom,
                                      end_atom,
                                      projectors,
                                      quantum_system,
                                      0,
                                      band_indices,
                                      host_overlaps.data());
        device_calculator.compute_block(first_atom,
                                        end_atom,
                                        projectors,
                                        quantum_system,
                                        0,
                                        band_indices,
                                        device_overlaps.data());
        Real largest = 0.0;
        for (std::size_t element = 0; element < number_of_overlaps; ++element) {
            largest = std::max(largest, std::abs(host_overlaps[element]));
            ASSERT_NEAR(device_overlaps[element].real(), host_overlaps[element].real(), 1.0e-11)
                << "element " << element;
            ASSERT_NEAR(device_overlaps[element].imag(), host_overlaps[element].imag(), 1.0e-11)
                << "element " << element;
        }
        EXPECT_GT(largest, 1.0e-3);
    }
};

} // namespace

TEST(GPUProjectorStrainOverlapCalculator, EveryKindOfEveryAtomMatchesTheHostCalculator) {
    DeviceStrainCase strain_case;

    strain_case.expect_block_matches(0, 4, projectors_of_all_atoms, {0, 1, 2, 3, 4, 5});
}

TEST(GPUProjectorStrainOverlapCalculator, ABlockOfSomeAtomsAndSomeBandsMatchesToo) {
    DeviceStrainCase strain_case;

    // Atoms 1 and 2 (species B then A), a run of bands, then bands with a gap.
    strain_case.expect_block_matches(1, 3, 22, {1, 2, 3});
    strain_case.expect_block_matches(1, 3, 22, {0, 2, 5});
    strain_case.expect_block_matches(3, 4, 18, {4});
}

TEST(GPUProjectorStrainOverlapCalculator, ValuesAtAPlaneWaveMatchTheHostColumns) {
    DeviceStrainCase strain_case;
    const std::vector<int> band_indices = {0, 1};
    std::vector<Complex> overlaps(static_cast<std::size_t>(number_of_kinds) * 22 * 2);
    strain_case.host_calculator.compute_block(
        1, 3, 22, strain_case.quantum_system, 0, band_indices, overlaps.data());
    strain_case.device_calculator.compute_block(
        1, 3, 22, strain_case.quantum_system, 0, band_indices, overlaps.data());

    for (int plane_wave : {0, 7}) {
        std::vector<Complex> host_values;
        std::vector<Complex> device_values;
        strain_case.host_calculator.block_values_at_plane_wave(plane_wave, host_values);
        strain_case.device_calculator.block_values_at_plane_wave(plane_wave, device_values);

        ASSERT_EQ(device_values.size(), host_values.size());
        for (std::size_t row = 0; row < host_values.size(); ++row) {
            EXPECT_NEAR(device_values[row].real(), host_values[row].real(), 1.0e-13) << row;
            EXPECT_NEAR(device_values[row].imag(), host_values[row].imag(), 1.0e-13) << row;
        }
    }
}

TEST(GPUProjectorStrainOverlapCalculator, OnlyFactorsGoUpAndOnlyOverlapsComeDown) {
    DeviceStrainCase strain_case;
    const std::size_t plane_waves =
        static_cast<std::size_t>(strain_case.system.number_of_plane_waves());
    // The form factors went up when the k-point began: kinds × channels of each species.
    const std::size_t form_factor_bytes =
        static_cast<std::size_t>(number_of_kinds) * (4 + 18) * plane_waves * sizeof(Complex);
    EXPECT_EQ(strain_case.device_backend->transfer_totals().host_to_device_bytes,
              form_factor_bytes);

    // As after a device solve: the bands' device block is the current one.
    (void)strain_case.quantum_system.mutable_device_band_block(0, strain_case.device_backend);
    const std::vector<int> band_indices = {0, 1, 2, 3, 4, 5};
    std::vector<Complex> overlaps(static_cast<std::size_t>(number_of_kinds) *
                                  projectors_of_all_atoms * band_indices.size());
    const DeviceTransferTotals before = strain_case.device_backend->transfer_totals();

    strain_case.device_calculator.compute_block(0,
                                                4,
                                                projectors_of_all_atoms,
                                                strain_case.quantum_system,
                                                0,
                                                band_indices,
                                                overlaps.data());

    const DeviceTransferTotals moved = strain_case.device_backend->transfer_totals().since(before);
    // Up: the conjugate structure factor of each of the four atoms, as two real vectors.
    EXPECT_EQ(moved.host_to_device_bytes, 4 * 2 * plane_waves * sizeof(Real));
    EXPECT_EQ(moved.host_to_device_copies, 1u);
    // Down: the overlaps, and nothing of the bands.
    EXPECT_EQ(moved.device_to_host_bytes, overlaps.size() * sizeof(Complex));
    EXPECT_EQ(moved.device_to_host_copies, 1u);

    /**
     * Against what the host calculator's columns would have cost to upload: seven columns
     * per projector, each as long as the plane waves.
     */
    const std::size_t column_bytes = static_cast<std::size_t>(number_of_kinds) *
                                     projectors_of_all_atoms * plane_waves * sizeof(Complex);
    EXPECT_LT(form_factor_bytes + moved.host_to_device_bytes, column_bytes * 6 / 10);
}

TEST(GPUProjectorStrainOverlapCalculator, ANullBackendIsRefused) {
    TwoSpeciesProjectorSystem system;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas(device_backend);

    EXPECT_THROW(GPUProjectorStrainOverlapCalculator(system.crystal_structure,
                                                     system.planewave_basis,
                                                     system.pseudopotential_basis(),
                                                     nullptr,
                                                     device_blas),
                 std::invalid_argument);
}
