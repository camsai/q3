#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set_store.hpp"
#include "utils/counting_projector_set_builder.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/silicon_projector_system.hpp"

using namespace q3;
using test_support::CountingProjectorSetBuilder;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::SiliconProjectorSystem;

namespace {

/// A store over the silicon system's two k-points whose builds are counted.
struct CountedStore {
    SiliconProjectorSystem silicon;
    int number_of_builds = 0;
    KleinmanBylanderProjectorSetStore store{
        silicon.crystal_structure,
        silicon.kpoint_grid,
        silicon.g_vector_indices_by_kpoint,
        std::make_unique<CountingProjectorSetBuilder>(silicon.crystal_structure,
                                                      silicon.planewave_basis,
                                                      silicon.pseudopotential_basis(),
                                                      number_of_builds)};
};

} // namespace

// ─── Building on demand ─────────────────────────────────────────────────────

TEST(KleinmanBylanderProjectorSetStore, ANewStoreHasOneUnbuiltSetPerKpoint) {
    CountedStore counted;

    EXPECT_EQ(counted.store.number_of_kpoints(), 2);
    EXPECT_FALSE(counted.store.is_built(0));
    EXPECT_FALSE(counted.store.is_built(1));
    EXPECT_EQ(counted.number_of_builds, 0);
}

TEST(KleinmanBylanderProjectorSetStore, ASetIsBuiltWhenFirstAskedForAndNotAgain) {
    CountedStore counted;

    const KleinmanBylanderProjectorSet& first_request = counted.store.at(1);
    const KleinmanBylanderProjectorSet& second_request = counted.store.at(1);

    EXPECT_EQ(&first_request, &second_request);
    EXPECT_EQ(counted.number_of_builds, 1);
    EXPECT_TRUE(counted.store.is_built(1));
    EXPECT_FALSE(counted.store.is_built(0));
    EXPECT_EQ(first_request.number_of_plane_waves(), counted.silicon.number_of_plane_waves(1));
    EXPECT_TRUE(first_request.is_built_from(counted.silicon.kpoint_grid[1].coordinates,
                                            counted.silicon.crystal_structure,
                                            counted.silicon.number_of_plane_waves(1)));
}

TEST(KleinmanBylanderProjectorSetStore, ANullBuilderIsRefused) {
    SiliconProjectorSystem silicon;

    EXPECT_THROW(KleinmanBylanderProjectorSetStore(silicon.crystal_structure,
                                                   silicon.kpoint_grid,
                                                   silicon.g_vector_indices_by_kpoint,
                                                   nullptr),
                 std::invalid_argument);
}

// ─── What it keeps ──────────────────────────────────────────────────────────

TEST(KleinmanBylanderProjectorSetStore, EverySetIsKeptUnlessToldOtherwise) {
    CountedStore counted;

    (void)counted.store.at(0);
    (void)counted.store.at(1);
    (void)counted.store.at(0);

    EXPECT_TRUE(counted.store.is_built(0));
    EXPECT_TRUE(counted.store.is_built(1));
    EXPECT_EQ(counted.number_of_builds, 2);
}

TEST(KleinmanBylanderProjectorSetStore, KeepingTheLatestSetOnlyFreesTheOthersWhenOneIsBuilt) {
    CountedStore counted;
    counted.store.keep_latest_set_only();

    (void)counted.store.at(0);
    (void)counted.store.at(1);

    EXPECT_FALSE(counted.store.is_built(0));
    EXPECT_TRUE(counted.store.is_built(1));

    // Walking back rebuilds: that is the price of holding one set at a time.
    (void)counted.store.at(0);
    EXPECT_TRUE(counted.store.is_built(0));
    EXPECT_FALSE(counted.store.is_built(1));
    EXPECT_EQ(counted.number_of_builds, 3);
}

TEST(KleinmanBylanderProjectorSetStore, TheLatestSetIsNotRebuiltWhenAskedForAgain) {
    CountedStore counted;
    counted.store.keep_latest_set_only();

    const KleinmanBylanderProjectorSet& projector_set = counted.store.at(1);
    const Complex* projectors = projector_set.host();
    (void)counted.store.at(1);

    EXPECT_EQ(counted.number_of_builds, 1);
    EXPECT_EQ(counted.store.at(1).host(), projectors);
}

TEST(KleinmanBylanderProjectorSetStore, KeepingAllSetsAgainStopsTheFreeing) {
    CountedStore counted;
    counted.store.keep_latest_set_only();
    (void)counted.store.at(0);

    counted.store.keep_all_sets();
    (void)counted.store.at(1);

    EXPECT_TRUE(counted.store.is_built(0));
    EXPECT_TRUE(counted.store.is_built(1));
}

TEST(KleinmanBylanderProjectorSetStore, ReleasingFreesEverySetAndTheNextRequestRebuilds) {
    CountedStore counted;
    (void)counted.store.at(0);
    (void)counted.store.at(1);

    counted.store.release();

    EXPECT_FALSE(counted.store.is_built(0));
    EXPECT_FALSE(counted.store.is_built(1));
    EXPECT_FALSE(counted.store.at(0).empty());
    EXPECT_EQ(counted.number_of_builds, 3);
}

// ─── Knowing when a set is out of date ──────────────────────────────────────

TEST(KleinmanBylanderProjectorSetStore, AMovedAtomRebuildsTheSetInPlace) {
    CountedStore counted;
    const KleinmanBylanderProjectorSet& projector_set = counted.store.at(1);
    const int projectors_per_atom = projector_set.number_of_projectors() / 2;
    const Complex second_atom_value_before = projector_set.projector(projectors_per_atom)[1];
    const Complex first_atom_value_before = projector_set.projector(0)[1];

    counted.silicon.displace_second_atom();
    const KleinmanBylanderProjectorSet& projector_set_after = counted.store.at(1);

    EXPECT_EQ(&projector_set_after, &projector_set);
    EXPECT_EQ(counted.number_of_builds, 2);
    // The atom that stayed keeps its projectors; the one that moved gets a new phase.
    EXPECT_EQ(projector_set.projector(0)[1], first_atom_value_before);
    EXPECT_NE(projector_set.projector(projectors_per_atom)[1], second_atom_value_before);
}

TEST(KleinmanBylanderProjectorSetStore, AnUnchangedGeometryIsNotRebuilt) {
    CountedStore counted;
    (void)counted.store.at(1);

    // Setting the same positions again is not a change.
    counted.silicon.crystal_structure.set_atomic_positions(
        {Vector3D(0.0, 0.0, 0.0), Vector3D(0.25, 0.25, 0.25)});
    (void)counted.store.at(1);

    EXPECT_EQ(counted.number_of_builds, 1);
}

// ─── Device copies ──────────────────────────────────────────────────────────

TEST(KleinmanBylanderProjectorSetStore, EachSetIsUploadedOnceHoweverOftenItIsRead) {
    CountedStore counted;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();

    for (int pass = 0; pass < 3; ++pass) {
        for (int kpoint_index = 0; kpoint_index < 2; ++kpoint_index) {
            (void)counted.store.at(kpoint_index).device(device_backend);
        }
    }

    const std::size_t all_sets_bytes =
        static_cast<std::size_t>(counted.store.at(0).number_of_projectors()) *
        (counted.silicon.number_of_plane_waves(0) + counted.silicon.number_of_plane_waves(1)) *
        sizeof(Complex);
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 2u);
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_bytes, all_sets_bytes);
}

TEST(KleinmanBylanderProjectorSetStore, AMovedAtomIsUploadedAgainWithoutBeingWithdrawnFirst) {
    CountedStore counted;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    const int projectors_per_atom = counted.store.at(1).number_of_projectors() / 2;
    const int number_of_plane_waves = counted.silicon.number_of_plane_waves(1);
    const Complex stale_device_value =
        counted.store.at(1).device(device_backend)[projectors_per_atom * number_of_plane_waves + 1];

    counted.silicon.displace_second_atom();
    const KleinmanBylanderProjectorSet& projector_set = counted.store.at(1);

    /**
     * The reader does nothing but ask for the device copy again: the set
     * knows the rebuild made its upload stale.
     */
    EXPECT_FALSE(projector_set.device_is_current());
    const Complex* device_projectors = projector_set.device(device_backend);
    EXPECT_NE(device_projectors[projectors_per_atom * number_of_plane_waves + 1],
              stale_device_value);
    EXPECT_EQ(device_projectors[projectors_per_atom * number_of_plane_waves + 1],
              projector_set.projector(projectors_per_atom)[1]);
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 2u);
}

TEST(KleinmanBylanderProjectorSetStore, ReleasingFreesTheDeviceCopiesToo) {
    CountedStore counted;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    const Complex* device_projectors = counted.store.at(0).device(device_backend);

    counted.store.release();

    EXPECT_FALSE(device_backend->is_device_memory(device_projectors));
}

// ─── A budget for the device copies ─────────────────────────────────────────

namespace {

/// A store over four k-points of the silicon system, with its builds counted.
struct FourKpointStore {
    SiliconProjectorSystem silicon;
    KPointsGrid kpoint_grid;
    std::vector<std::vector<int>> g_vector_indices_by_kpoint;
    int number_of_builds = 0;
    std::unique_ptr<KleinmanBylanderProjectorSetStore> store;
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();

    FourKpointStore() {
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (int kpoint_index = 0; kpoint_index < 2; ++kpoint_index) {
                KPoint kpoint = silicon.kpoint_grid[kpoint_index];
                kpoint.coordinates.z += 0.125 * repeat;
                kpoint.weight = 0.25;
                kpoint_grid.add_kpoint(kpoint);
                g_vector_indices_by_kpoint.push_back(
                    silicon.g_vector_indices_by_kpoint[static_cast<std::size_t>(kpoint_index)]);
            }
        }
        store = std::make_unique<KleinmanBylanderProjectorSetStore>(
            silicon.crystal_structure,
            kpoint_grid,
            g_vector_indices_by_kpoint,
            std::make_unique<CountingProjectorSetBuilder>(silicon.crystal_structure,
                                                          silicon.planewave_basis,
                                                          silicon.pseudopotential_basis(),
                                                          number_of_builds));
    }

    [[nodiscard]] std::size_t set_bytes(int kpoint_index) const {
        return static_cast<std::size_t>(
                   silicon.pseudopotential_basis().total_number_of_projectors()) *
               g_vector_indices_by_kpoint[static_cast<std::size_t>(kpoint_index)].size() *
               sizeof(Complex);
    }

    /// Hand out a k-point's set and read it on the device, as a solve of that k-point does.
    void use_on_device(int kpoint_index) {
        (void)store->at(kpoint_index).device(device_backend);
    }
};

} // namespace

TEST(KleinmanBylanderProjectorSetStore, WithoutABudgetEveryDeviceCopyIsKept) {
    FourKpointStore four;

    for (int kpoint_index = 0; kpoint_index < 4; ++kpoint_index) {
        four.use_on_device(kpoint_index);
    }

    EXPECT_EQ(four.store->device_memory_budget(), 0u);
    EXPECT_EQ(four.store->device_bytes(),
              four.set_bytes(0) + four.set_bytes(1) + four.set_bytes(2) + four.set_bytes(3));
}

TEST(KleinmanBylanderProjectorSetStore, OverBudgetTheSetUsedLongestAgoGivesUpItsDeviceCopy) {
    FourKpointStore four;
    // Room for the sets of two k-points, whichever two.
    four.store->set_device_memory_budget(2 * std::max(four.set_bytes(0), four.set_bytes(1)));

    four.use_on_device(0);
    four.use_on_device(1);
    four.use_on_device(0); // k-point 1 is now the one used longest ago
    four.use_on_device(2);

    EXPECT_GT(four.store->device_bytes(0), 0u);
    EXPECT_EQ(four.store->device_bytes(1), 0u);
    EXPECT_GT(four.store->device_bytes(2), 0u);
    EXPECT_LE(four.store->device_bytes(), four.store->device_memory_budget());
}

TEST(KleinmanBylanderProjectorSetStore, AnEvictedHostBuiltSetIsUploadedAgainAndNotRebuilt) {
    FourKpointStore four;
    four.store->set_device_memory_budget(four.set_bytes(0));

    four.use_on_device(0);
    four.use_on_device(1);
    four.use_on_device(0);

    // Built once each; k-point 0 went up twice, k-point 1 once.
    EXPECT_EQ(four.number_of_builds, 2);
    EXPECT_EQ(four.device_backend->transfer_totals().host_to_device_copies, 3u);
    EXPECT_TRUE(four.store->is_built(0));
    EXPECT_TRUE(four.store->is_built(1));
}

TEST(KleinmanBylanderProjectorSetStore, ManyPassesOverMoreKpointsThanFitStayUnderTheBudget) {
    FourKpointStore four;
    const std::size_t budget = 2 * std::max(four.set_bytes(0), four.set_bytes(1));
    four.store->set_device_memory_budget(budget);
    std::vector<Complex> first_pass_values;

    for (int pass = 0; pass < 3; ++pass) {
        for (int kpoint_index = 0; kpoint_index < 4; ++kpoint_index) {
            const KleinmanBylanderProjectorSet& projector_set = four.store->at(kpoint_index);
            const Complex value = projector_set.device(four.device_backend)[7];
            EXPECT_LE(four.store->device_bytes(), budget);
            if (pass == 0) {
                first_pass_values.push_back(value);
            } else {
                // What comes back after an eviction is what was there before it.
                EXPECT_EQ(value, first_pass_values[static_cast<std::size_t>(kpoint_index)]);
            }
        }
    }
    EXPECT_EQ(four.number_of_builds, 4);
}

TEST(KleinmanBylanderProjectorSetStore, ASetLargerThanTheBudgetIsStillHandedOut) {
    FourKpointStore four;
    four.store->set_device_memory_budget(1);

    four.use_on_device(0);
    four.use_on_device(1);

    EXPECT_EQ(four.store->device_bytes(0), 0u);
    EXPECT_GT(four.store->device_bytes(1), 0u);
}
