#include <gtest/gtest.h>

#include <memory>

#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set_store.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/projector_set_provider.hpp"
#include "utils/counting_projector_set_builder.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/silicon_projector_system.hpp"

using namespace q3;
using test_support::CountingProjectorSetBuilder;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::SiliconProjectorSystem;

namespace {

/// The provider's request for one k-point of the silicon system, as a solver makes it.
const KleinmanBylanderProjectorSet&
request(ProjectorSetProvider& provider, const SiliconProjectorSystem& silicon, int kpoint_index) {
    return provider.projector_set(kpoint_index,
                                  silicon.crystal_structure,
                                  silicon.kpoint_grid,
                                  silicon.planewave_basis,
                                  silicon.pseudopotential_basis(),
                                  silicon.g_vector_indices_by_kpoint[kpoint_index],
                                  silicon.number_of_plane_waves(kpoint_index));
}

} // namespace

TEST(ProjectorSetProvider, WithoutAStoreItBuildsASetOfItsOwnForThePlaneWavesGiven) {
    SiliconProjectorSystem silicon;
    ProjectorSetProvider provider;

    const KleinmanBylanderProjectorSet& projector_set = request(provider, silicon, 1);

    EXPECT_EQ(provider.store(), nullptr);
    EXPECT_TRUE(projector_set.is_built_from(silicon.kpoint_grid[1].coordinates,
                                            silicon.crystal_structure,
                                            silicon.number_of_plane_waves(1)));
    EXPECT_EQ(projector_set.number_of_projectors(),
              silicon.pseudopotential_basis().total_number_of_projectors());
}

TEST(ProjectorSetProvider, ItsOwnSetIsReplacedByTheNextRequest) {
    SiliconProjectorSystem silicon;
    ProjectorSetProvider provider;
    const KleinmanBylanderProjectorSet& at_first_kpoint = request(provider, silicon, 0);

    const KleinmanBylanderProjectorSet& at_second_kpoint = request(provider, silicon, 1);

    EXPECT_EQ(&at_first_kpoint, &at_second_kpoint);
    EXPECT_EQ(at_second_kpoint.number_of_plane_waves(), silicon.number_of_plane_waves(1));
}

TEST(ProjectorSetProvider, WithAStoreItHandsOutTheStoresSetBuiltOnce) {
    SiliconProjectorSystem silicon;
    int number_of_builds = 0;
    KleinmanBylanderProjectorSetStore store(
        silicon.crystal_structure,
        silicon.kpoint_grid,
        silicon.g_vector_indices_by_kpoint,
        std::make_unique<CountingProjectorSetBuilder>(silicon.crystal_structure,
                                                      silicon.planewave_basis,
                                                      silicon.pseudopotential_basis(),
                                                      number_of_builds));
    ProjectorSetProvider provider;
    provider.set_store(&store);

    const KleinmanBylanderProjectorSet& first_request = request(provider, silicon, 1);
    const KleinmanBylanderProjectorSet& second_request = request(provider, silicon, 1);

    EXPECT_EQ(provider.store(), &store);
    EXPECT_EQ(&first_request, &store.at(1));
    EXPECT_EQ(&first_request, &second_request);
    EXPECT_EQ(number_of_builds, 1);
}

TEST(ProjectorSetProvider, TakingTheStoreBackReturnsToItsOwnSet) {
    SiliconProjectorSystem silicon;
    int number_of_builds = 0;
    KleinmanBylanderProjectorSetStore store(
        silicon.crystal_structure,
        silicon.kpoint_grid,
        silicon.g_vector_indices_by_kpoint,
        std::make_unique<CountingProjectorSetBuilder>(silicon.crystal_structure,
                                                      silicon.planewave_basis,
                                                      silicon.pseudopotential_basis(),
                                                      number_of_builds));
    ProjectorSetProvider provider;
    provider.set_store(&store);
    provider.set_store(nullptr);

    const KleinmanBylanderProjectorSet& projector_set = request(provider, silicon, 0);

    EXPECT_NE(&projector_set, &store.at(0));
    EXPECT_EQ(number_of_builds, 1); // the store's own at(0) just above, not the provider's
}

TEST(ProjectorSetProvider, ASetItHandsOutCanBeReadOnADevice) {
    SiliconProjectorSystem silicon;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    ProjectorSetProvider provider;

    const KleinmanBylanderProjectorSet& projector_set = request(provider, silicon, 0);

    EXPECT_TRUE(device_backend->is_device_memory(projector_set.device(device_backend)));
}
