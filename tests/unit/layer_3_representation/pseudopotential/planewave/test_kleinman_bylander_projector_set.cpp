#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"

using namespace q3;
using test_support::CPUBackedGPUDeviceBackend;

namespace {

constexpr int number_of_projectors = 3;
constexpr int number_of_plane_waves = 5;
constexpr std::size_t set_bytes = number_of_projectors * number_of_plane_waves * sizeof(Complex);

/// A two-atom cell; only its geometry matters to a set's record of what it was built from.
Crystal two_atom_crystal() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.2);
    crystal_structure.add_species({"Si", 28.08, 4.0, "Si.upf"});
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

const Vector3D kpoint(0.25, 0.0, 0.5);

/// Build the set with projector p holding p + i·g at plane wave g, shifted by `offset`.
void build_labelled(KleinmanBylanderProjectorSet& projector_set,
                    const Crystal& crystal_structure,
                    Real offset = 0.0) {
    Complex* projectors = projector_set.begin_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves);
    for (int projector = 0; projector < number_of_projectors; ++projector) {
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            projectors[projector * number_of_plane_waves + g_index] =
                Complex(projector + offset, g_index);
        }
    }
}

} // namespace

// ─── What it holds ──────────────────────────────────────────────────────────

TEST(KleinmanBylanderProjectorSet, ANewSetIsNotBuilt) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();

    EXPECT_FALSE(projector_set.is_built());
    EXPECT_TRUE(projector_set.empty());
    EXPECT_EQ(projector_set.number_of_projectors(), 0);
    EXPECT_EQ(projector_set.number_of_plane_waves(), 0);
    EXPECT_FALSE(projector_set.is_built_from(kpoint, crystal_structure, 0));
}

TEST(KleinmanBylanderProjectorSet, BeginBuildSizesAndZeroesTheMatrix) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();

    const Complex* projectors = projector_set.begin_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves);

    EXPECT_TRUE(projector_set.is_built());
    EXPECT_FALSE(projector_set.empty());
    EXPECT_EQ(projector_set.number_of_projectors(), number_of_projectors);
    EXPECT_EQ(projector_set.number_of_plane_waves(), number_of_plane_waves);
    EXPECT_EQ(projector_set.host(), projectors);
    for (int element = 0; element < number_of_projectors * number_of_plane_waves; ++element) {
        EXPECT_EQ(projectors[element], Complex(0.0, 0.0));
    }
}

TEST(KleinmanBylanderProjectorSet, AProjectorIsARowOfTheMatrix) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);

    for (int projector = 0; projector < number_of_projectors; ++projector) {
        EXPECT_EQ(projector_set.projector(projector),
                  projector_set.host() + projector * number_of_plane_waves);
        EXPECT_EQ(projector_set.projector(projector)[4], Complex(projector, 4.0));
    }
}

TEST(KleinmanBylanderProjectorSet, ASetOfNoProjectorsIsBuiltButEmpty) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();

    (void)projector_set.begin_build(kpoint, crystal_structure, 0, number_of_plane_waves);

    EXPECT_TRUE(projector_set.is_built());
    EXPECT_TRUE(projector_set.empty());
    EXPECT_TRUE(projector_set.is_built_from(kpoint, crystal_structure, number_of_plane_waves));
}

TEST(KleinmanBylanderProjectorSet, RebuildingStartsFromZeroAgain) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);

    const Complex* projectors = projector_set.begin_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves);

    EXPECT_EQ(projectors[number_of_projectors * number_of_plane_waves - 1], Complex(0.0, 0.0));
}

// ─── What it was built from ─────────────────────────────────────────────────

TEST(KleinmanBylanderProjectorSet, IsBuiltFromTheKpointAndCrystalItWasBuiltWith) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);

    EXPECT_TRUE(projector_set.is_built_from(kpoint, crystal_structure, number_of_plane_waves));
    EXPECT_TRUE(projector_set.is_built_from(kpoint, two_atom_crystal(), number_of_plane_waves));
}

TEST(KleinmanBylanderProjectorSet, IsNotBuiltFromAnotherKpointOrPlaneWaveCount) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);

    EXPECT_FALSE(projector_set.is_built_from(
        Vector3D(0.25, 0.0, 0.25), crystal_structure, number_of_plane_waves));
    EXPECT_FALSE(projector_set.is_built_from(kpoint, crystal_structure, number_of_plane_waves + 1));
}

TEST(KleinmanBylanderProjectorSet, IsNotBuiltFromACrystalWhoseAtomMoved) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);

    crystal_structure.set_atomic_positions({Vector3D(0.0, 0.0, 0.0), Vector3D(0.26, 0.25, 0.25)});

    EXPECT_FALSE(projector_set.is_built_from(kpoint, crystal_structure, number_of_plane_waves));
}

TEST(KleinmanBylanderProjectorSet, IsNotBuiltFromACrystalWithAnotherCell) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);

    Crystal strained;
    strained.set_from_lattice_type(2, 10.3);
    strained.add_species({"Si", 28.08, 4.0, "Si.upf"});
    strained.add_atom({0, {0.0, 0.0, 0.0}});
    strained.add_atom({0, {0.25, 0.25, 0.25}});
    strained.compute_derived_quantities();

    EXPECT_FALSE(projector_set.is_built_from(kpoint, strained, number_of_plane_waves));
}

TEST(KleinmanBylanderProjectorSet, IsNotBuiltFromACrystalWithAnotherAtomCountOrSpecies) {
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);

    Crystal three_atoms = two_atom_crystal();
    three_atoms.add_atom({0, {0.5, 0.5, 0.5}});
    Crystal other_species = two_atom_crystal();
    other_species.add_species({"Ge", 72.63, 4.0, "Ge.upf"});
    other_species.add_atom({1, {0.5, 0.5, 0.5}});
    KleinmanBylanderProjectorSet three_atom_set;
    build_labelled(three_atom_set, three_atoms);

    EXPECT_FALSE(projector_set.is_built_from(kpoint, three_atoms, number_of_plane_waves));
    EXPECT_FALSE(three_atom_set.is_built_from(kpoint, other_species, number_of_plane_waves));
}

// ─── The device copy ────────────────────────────────────────────────────────

TEST(KleinmanBylanderProjectorSet, TheDeviceCopyIsUploadedOnceWhileTheSetIsNotRebuilt) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);

    const Complex* device_projectors = projector_set.device(device_backend);
    const Complex* device_projectors_again = projector_set.device(device_backend);

    EXPECT_TRUE(device_backend->is_device_memory(device_projectors));
    EXPECT_EQ(device_projectors, device_projectors_again);
    EXPECT_TRUE(projector_set.device_is_current());
    EXPECT_EQ(device_projectors[2 * number_of_plane_waves + 3], Complex(2.0, 3.0));
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_bytes, set_bytes);
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 1u);
}

TEST(KleinmanBylanderProjectorSet, ARebuildMakesTheDeviceCopyStaleAndTheNextReadUploadsAgain) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);
    (void)projector_set.device(device_backend);

    build_labelled(projector_set, crystal_structure, /*offset=*/10.0);

    EXPECT_FALSE(projector_set.device_is_current());
    EXPECT_EQ(projector_set.device(device_backend)[2 * number_of_plane_waves + 3],
              Complex(12.0, 3.0));
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 2u);
}

TEST(KleinmanBylanderProjectorSet, ReleasingFreesTheMatrixOnBothSides) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;
    Crystal crystal_structure = two_atom_crystal();
    build_labelled(projector_set, crystal_structure);
    const Complex* device_projectors = projector_set.device(device_backend);

    projector_set.release();

    EXPECT_FALSE(projector_set.is_built());
    EXPECT_TRUE(projector_set.empty());
    EXPECT_FALSE(projector_set.device_is_current());
    EXPECT_FALSE(device_backend->is_device_memory(device_projectors));
    EXPECT_FALSE(projector_set.is_built_from(kpoint, crystal_structure, number_of_plane_waves));
    // Nothing had to come back from the device: the host copy was current.
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 0u);
}

// ─── Built on a device ──────────────────────────────────────────────────────

TEST(KleinmanBylanderProjectorSet, ADeviceBuildTakesNoHostMemoryAndUploadsNothing) {
    Crystal crystal_structure = two_atom_crystal();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;

    Complex* device_projectors = projector_set.begin_device_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves, device_backend);
    for (int element = 0; element < number_of_projectors * number_of_plane_waves; ++element) {
        device_projectors[element] = Complex(element, -element);
    }

    EXPECT_TRUE(projector_set.is_built());
    EXPECT_TRUE(projector_set.is_built_from(kpoint, crystal_structure, number_of_plane_waves));
    EXPECT_EQ(projector_set.number_of_projectors(), number_of_projectors);
    EXPECT_FALSE(projector_set.host_is_allocated());
    EXPECT_EQ(projector_set.device_bytes(), set_bytes);
    EXPECT_EQ(projector_set.device(device_backend), device_projectors);
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 0u);
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 0u);
}

TEST(KleinmanBylanderProjectorSet, AHostReaderOfADeviceBuiltSetGetsItInOneDownload) {
    Crystal crystal_structure = two_atom_crystal();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;
    Complex* device_projectors = projector_set.begin_device_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves, device_backend);
    for (int element = 0; element < number_of_projectors * number_of_plane_waves; ++element) {
        device_projectors[element] = Complex(element, -element);
    }

    const Complex second_projector_third_wave = projector_set.projector(1)[2];

    EXPECT_EQ(second_projector_third_wave,
              Complex(number_of_plane_waves + 2, -(number_of_plane_waves + 2)));
    EXPECT_TRUE(projector_set.host_is_allocated());
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 1u);
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_bytes, set_bytes);
}

TEST(KleinmanBylanderProjectorSet, ADeviceRebuildOfTheSameSizeReusesTheDeviceCopy) {
    Crystal crystal_structure = two_atom_crystal();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;
    Complex* first = projector_set.begin_device_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves, device_backend);

    Complex* second = projector_set.begin_device_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves, device_backend);

    EXPECT_EQ(second, first);
}

TEST(KleinmanBylanderProjectorSet, EvictingASetTheHostHoldsKeepsItAndUploadsAgainLater) {
    Crystal crystal_structure = two_atom_crystal();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;
    build_labelled(projector_set, crystal_structure);
    (void)projector_set.device(device_backend);

    projector_set.evict_from_device();

    EXPECT_TRUE(projector_set.is_built());
    EXPECT_EQ(projector_set.device_bytes(), 0u);
    EXPECT_EQ(projector_set.projector(2)[1], Complex(2.0, 1.0));
    (void)projector_set.device(device_backend);
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 2u);
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 0u);
}

TEST(KleinmanBylanderProjectorSet, EvictingASetOnlyTheDeviceHoldsReleasesItWithoutADownload) {
    Crystal crystal_structure = two_atom_crystal();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;
    (void)projector_set.begin_device_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves, device_backend);

    projector_set.evict_from_device();

    EXPECT_FALSE(projector_set.is_built());
    EXPECT_EQ(projector_set.device_bytes(), 0u);
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 0u);
}

TEST(KleinmanBylanderProjectorSet, ReleasingADeviceBuiltSetBringsNothingDown) {
    Crystal crystal_structure = two_atom_crystal();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    KleinmanBylanderProjectorSet projector_set;
    (void)projector_set.begin_device_build(
        kpoint, crystal_structure, number_of_projectors, number_of_plane_waves, device_backend);

    projector_set.release();

    EXPECT_FALSE(projector_set.is_built());
    EXPECT_TRUE(projector_set.empty());
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 0u);
}
