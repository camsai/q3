#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/execution/gpu/device/device_transfer_totals.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder_serial.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/projector_set_builder_gpu.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/silicon_projector_system.hpp"
#include "utils/two_species_projector_system.hpp"

using namespace q3;
using execution::gpu::device::DeviceTransferTotals;
using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::SiliconProjectorSystem;
using test_support::TwoSpeciesProjectorSystem;

namespace {

void expect_same_projectors(const KleinmanBylanderProjectorSet& device_built,
                            const KleinmanBylanderProjectorSet& host_built) {
    ASSERT_EQ(device_built.number_of_projectors(), host_built.number_of_projectors());
    ASSERT_EQ(device_built.number_of_plane_waves(), host_built.number_of_plane_waves());
    const std::size_t number_of_elements =
        static_cast<std::size_t>(host_built.number_of_projectors()) *
        static_cast<std::size_t>(host_built.number_of_plane_waves());
    const Complex* device_values = device_built.host();
    const Complex* host_values = host_built.host();
    for (std::size_t element = 0; element < number_of_elements; ++element) {
        ASSERT_EQ(device_values[element], host_values[element]) << "element " << element;
    }
}

} // namespace

TEST(GPUProjectorSetBuilder, BuildsTheProjectorsTheHostBuilderBuilds) {
    SiliconProjectorSystem silicon;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas(device_backend);
    GPUProjectorSetBuilder device_builder(silicon.crystal_structure,
                                          silicon.planewave_basis,
                                          silicon.pseudopotential_basis(),
                                          device_backend,
                                          device_blas);
    SerialProjectorSetBuilder host_builder(
        silicon.crystal_structure, silicon.planewave_basis, silicon.pseudopotential_basis());

    for (int kpoint_index = 0; kpoint_index < 2; ++kpoint_index) {
        KleinmanBylanderProjectorSet device_built;
        KleinmanBylanderProjectorSet host_built;
        device_builder.build(silicon.kpoint_grid[kpoint_index].coordinates,
                             silicon.g_vector_indices_by_kpoint[kpoint_index],
                             silicon.number_of_plane_waves(kpoint_index),
                             device_built);
        host_builder.build(silicon.kpoint_grid[kpoint_index].coordinates,
                           silicon.g_vector_indices_by_kpoint[kpoint_index],
                           silicon.number_of_plane_waves(kpoint_index),
                           host_built);

        EXPECT_TRUE(device_built.is_built_from(silicon.kpoint_grid[kpoint_index].coordinates,
                                               silicon.crystal_structure,
                                               silicon.number_of_plane_waves(kpoint_index)));
        expect_same_projectors(device_built, host_built);
    }
}

TEST(GPUProjectorSetBuilder, TwoSpeciesWithInterleavedAtomsEachGetTheirOwnProjectors) {
    TwoSpeciesProjectorSystem system;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas(device_backend);
    KleinmanBylanderProjectorSet device_built;
    KleinmanBylanderProjectorSet host_built;

    GPUProjectorSetBuilder(system.crystal_structure,
                           system.planewave_basis,
                           system.pseudopotential_basis(),
                           device_backend,
                           device_blas)
        .build(
            system.kpoint, system.g_vector_indices, system.number_of_plane_waves(), device_built);
    SerialProjectorSetBuilder(
        system.crystal_structure, system.planewave_basis, system.pseudopotential_basis())
        .build(system.kpoint, system.g_vector_indices, system.number_of_plane_waves(), host_built);

    // 2 × (1 + 3) for species A and 2 × 2 × (1 + 3 + 5) for species B.
    EXPECT_EQ(host_built.number_of_projectors(), 8 + 36);
    expect_same_projectors(device_built, host_built);
}

TEST(GPUProjectorSetBuilder, OnlyTheFactorsGoUpAndTheSetTakesNoHostMemory) {
    TwoSpeciesProjectorSystem system;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas(device_backend);
    GPUProjectorSetBuilder device_builder(system.crystal_structure,
                                          system.planewave_basis,
                                          system.pseudopotential_basis(),
                                          device_backend,
                                          device_blas);
    KleinmanBylanderProjectorSet projector_set;

    device_builder.build(
        system.kpoint, system.g_vector_indices, system.number_of_plane_waves(), projector_set);

    const std::size_t plane_waves = static_cast<std::size_t>(system.number_of_plane_waves());
    // One structure factor per atom, as its real and its imaginary part, all in one copy.
    const std::size_t structure_factor_bytes = 4 * 2 * plane_waves * sizeof(Real);
    // One form factor per channel of a species, real values held as complex: a copy per species.
    const std::size_t form_factor_bytes = (4 + 18) * plane_waves * sizeof(Complex);
    const DeviceTransferTotals totals = device_backend->transfer_totals();
    EXPECT_EQ(totals.host_to_device_bytes, structure_factor_bytes + form_factor_bytes);
    EXPECT_EQ(totals.host_to_device_copies, 1u + 2u);
    EXPECT_EQ(totals.device_to_host_copies, 0u);
    EXPECT_FALSE(projector_set.host_is_allocated());
    EXPECT_TRUE(projector_set.device_is_current());

    /**
     * Against the set itself, 44 projectors of complex numbers: under six tenths already
     * with four atoms, and the ratio falls as the atoms of a species grow in number, since
     * the form factors do not. For 64 atoms of one species with 18 channels it is 7%.
     */
    const std::size_t set_bytes = 44 * plane_waves * sizeof(Complex);
    EXPECT_LT(totals.host_to_device_bytes, set_bytes * 6 / 10);
}

TEST(GPUProjectorSetBuilder, RebuildingAfterAnAtomMovesReplacesTheDeviceCopyInPlace) {
    SiliconProjectorSystem silicon;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas(device_backend);
    GPUProjectorSetBuilder device_builder(silicon.crystal_structure,
                                          silicon.planewave_basis,
                                          silicon.pseudopotential_basis(),
                                          device_backend,
                                          device_blas);
    SerialProjectorSetBuilder host_builder(
        silicon.crystal_structure, silicon.planewave_basis, silicon.pseudopotential_basis());
    KleinmanBylanderProjectorSet device_built;
    device_builder.build(silicon.kpoint_grid[1].coordinates,
                         silicon.g_vector_indices_by_kpoint[1],
                         silicon.number_of_plane_waves(1),
                         device_built);
    const Complex* device_copy_before = device_built.device(device_backend);

    silicon.displace_second_atom();
    device_builder.build(silicon.kpoint_grid[1].coordinates,
                         silicon.g_vector_indices_by_kpoint[1],
                         silicon.number_of_plane_waves(1),
                         device_built);

    EXPECT_EQ(device_built.device(device_backend), device_copy_before);
    KleinmanBylanderProjectorSet host_built;
    host_builder.build(silicon.kpoint_grid[1].coordinates,
                       silicon.g_vector_indices_by_kpoint[1],
                       silicon.number_of_plane_waves(1),
                       host_built);
    expect_same_projectors(device_built, host_built);
}

TEST(GPUProjectorSetBuilder, ANullBackendIsRefused) {
    SiliconProjectorSystem silicon;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas(device_backend);

    EXPECT_THROW(GPUProjectorSetBuilder(silicon.crystal_structure,
                                        silicon.planewave_basis,
                                        silicon.pseudopotential_basis(),
                                        nullptr,
                                        device_blas),
                 std::invalid_argument);
}
