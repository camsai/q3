#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder_serial.hpp"
#include "utils/silicon_projector_system.hpp"

using namespace q3;
using test_support::SiliconProjectorSystem;

TEST(SerialProjectorSetBuilder, BuildsEveryProjectorOfEveryAtomOverTheKpointsPlaneWaves) {
    SiliconProjectorSystem silicon;
    SerialProjectorSetBuilder builder(
        silicon.crystal_structure, silicon.planewave_basis, silicon.pseudopotential_basis());
    KleinmanBylanderProjectorSet projector_set;

    builder.build(silicon.kpoint_grid[1].coordinates,
                  silicon.g_vector_indices_by_kpoint[1],
                  silicon.number_of_plane_waves(1),
                  projector_set);

    EXPECT_TRUE(projector_set.is_built());
    EXPECT_EQ(projector_set.number_of_projectors(), 8);
    EXPECT_EQ(projector_set.number_of_projectors(),
              silicon.pseudopotential_basis().total_number_of_projectors());
    EXPECT_EQ(projector_set.number_of_plane_waves(), silicon.number_of_plane_waves(1));
    EXPECT_TRUE(projector_set.is_built_from(silicon.kpoint_grid[1].coordinates,
                                            silicon.crystal_structure,
                                            silicon.number_of_plane_waves(1)));
}

TEST(SerialProjectorSetBuilder, TheSetHoldsWhatTheProjectorFormulaGives) {
    SiliconProjectorSystem silicon;
    SerialProjectorSetBuilder builder(
        silicon.crystal_structure, silicon.planewave_basis, silicon.pseudopotential_basis());
    KleinmanBylanderProjectorSet projector_set;
    const int number_of_plane_waves = silicon.number_of_plane_waves(1);
    const int number_of_projectors = silicon.pseudopotential_basis().total_number_of_projectors();

    builder.build(silicon.kpoint_grid[1].coordinates,
                  silicon.g_vector_indices_by_kpoint[1],
                  number_of_plane_waves,
                  projector_set);

    std::vector<Complex> expected(
        static_cast<std::size_t>(number_of_projectors) * number_of_plane_waves, Complex(0.0, 0.0));
    compute_kleinman_bylander_projectors(silicon.kpoint_grid[1].coordinates,
                                         silicon.crystal_structure,
                                         silicon.planewave_basis,
                                         silicon.pseudopotential_basis().pseudopotential_set(),
                                         silicon.pseudopotential_basis().beta_function_table(),
                                         silicon.g_vector_indices_by_kpoint[1],
                                         number_of_plane_waves,
                                         expected.data());
    Real largest_magnitude = 0.0;
    for (std::size_t element = 0; element < expected.size(); ++element) {
        EXPECT_EQ(projector_set.host()[element], expected[element]);
        largest_magnitude = std::max(largest_magnitude, std::abs(expected[element]));
    }
    EXPECT_GT(largest_magnitude, 0.0);
}

TEST(SerialProjectorSetBuilder, TheTwoAtomsProjectorsDifferByTheirStructureFactorAlone) {
    SiliconProjectorSystem silicon;
    SerialProjectorSetBuilder builder(
        silicon.crystal_structure, silicon.planewave_basis, silicon.pseudopotential_basis());
    KleinmanBylanderProjectorSet projector_set;
    const int number_of_plane_waves = silicon.number_of_plane_waves(1);

    builder.build(silicon.kpoint_grid[1].coordinates,
                  silicon.g_vector_indices_by_kpoint[1],
                  number_of_plane_waves,
                  projector_set);

    // Same species, same beta function, same plane wave: equal magnitude, different phase.
    const int projectors_per_atom = projector_set.number_of_projectors() / 2;
    for (int channel = 0; channel < projectors_per_atom; ++channel) {
        const Complex* first_atom = projector_set.projector(channel);
        const Complex* second_atom = projector_set.projector(projectors_per_atom + channel);
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            EXPECT_NEAR(std::abs(first_atom[g_index]), std::abs(second_atom[g_index]), 1.0e-12);
        }
    }
}

TEST(SerialProjectorSetBuilder, BuildingForAnotherKpointReplacesTheSet) {
    SiliconProjectorSystem silicon;
    SerialProjectorSetBuilder builder(
        silicon.crystal_structure, silicon.planewave_basis, silicon.pseudopotential_basis());
    KleinmanBylanderProjectorSet projector_set;
    builder.build(silicon.kpoint_grid[0].coordinates,
                  silicon.g_vector_indices_by_kpoint[0],
                  silicon.number_of_plane_waves(0),
                  projector_set);

    builder.build(silicon.kpoint_grid[1].coordinates,
                  silicon.g_vector_indices_by_kpoint[1],
                  silicon.number_of_plane_waves(1),
                  projector_set);

    EXPECT_EQ(projector_set.number_of_plane_waves(), silicon.number_of_plane_waves(1));
    EXPECT_FALSE(projector_set.is_built_from(silicon.kpoint_grid[0].coordinates,
                                             silicon.crystal_structure,
                                             silicon.number_of_plane_waves(0)));
    EXPECT_TRUE(projector_set.is_built_from(silicon.kpoint_grid[1].coordinates,
                                            silicon.crystal_structure,
                                            silicon.number_of_plane_waves(1)));
}
