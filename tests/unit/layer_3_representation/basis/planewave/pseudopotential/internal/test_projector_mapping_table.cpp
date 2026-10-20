#include <gtest/gtest.h>

#include <memory>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/atom.hpp"
#include "layer_1_structure/crystal/atom_species.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/projector_mapping_table.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

using namespace q3;

/**
 * Helper to build a simple Crystal with a given number of atoms all of one species.
 */
static Crystal build_simple_crystal(int number_of_atoms) {
    Crystal crystal;
    crystal.set_from_lattice_type(1, 10.0);
    AtomSpecies species{"Si", 28.08, 4.0, ""};
    crystal.add_species(species);
    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        Real offset = static_cast<Real>(atom_index) * 0.25;
        crystal.add_atom({0, {offset, 0.0, 0.0}});
    }
    crystal.compute_derived_quantities();
    return crystal;
}

/**
 * Helper to build a PseudopotentialSet with one species having given projectors.
 * Each projector is specified by its angular momentum.
 */
static PseudopotentialSet
build_pseudopotential_set_with_projectors(const std::vector<int>& angular_momenta) {
    auto pseudopotential = std::make_unique<Pseudopotential>();
    pseudopotential->element = "Si";
    pseudopotential->number_of_projectors = static_cast<int>(angular_momenta.size());
    pseudopotential->projectors.resize(angular_momenta.size());
    for (size_t index = 0; index < angular_momenta.size(); index++) {
        pseudopotential->projectors[index].angular_momentum = angular_momenta[index];
        pseudopotential->projectors[index].number_of_radial_cutoff_points = 100;
    }
    PseudopotentialSet set;
    set.push_back(std::move(pseudopotential));
    return set;
}

/**
 * Default-constructed ProjectorMappingTable should have zero total projectors.
 */
TEST(ProjectorMappingTable, DefaultConstruction) {
    ProjectorMappingTable table;
    EXPECT_EQ(table.total_number_of_projectors(), 0);
}

/**
 * Test building with a single s-orbital projector (l=0).
 * l=0 has 2l+1 = 1 channel. With 1 atom, total projectors = 1.
 */
TEST(ProjectorMappingTable, SingleSProjector) {
    Crystal crystal = build_simple_crystal(1);
    PseudopotentialSet pseudopotential_set = build_pseudopotential_set_with_projectors({0});
    int number_of_species_types = 1;

    ProjectorMappingTable table;
    table.build(crystal, pseudopotential_set, number_of_species_types);

    EXPECT_EQ(table.total_number_of_projectors(), 1);
    EXPECT_EQ(table.number_of_channels(0), 1);
    EXPECT_EQ(table.to_beta_index(0, 0), 0);
    EXPECT_EQ(table.angular_momentum(0, 0), 0);
    EXPECT_EQ(table.magnetic_quantum_number(0, 0), 0);
}

/**
 * Test building with a single p-orbital projector (l=1).
 * l=1 has 2l+1 = 3 channels (m = 0, 1, -1). With 1 atom, total = 3.
 */
TEST(ProjectorMappingTable, SinglePProjector) {
    Crystal crystal = build_simple_crystal(1);
    PseudopotentialSet pseudopotential_set = build_pseudopotential_set_with_projectors({1});
    int number_of_species_types = 1;

    ProjectorMappingTable table;
    table.build(crystal, pseudopotential_set, number_of_species_types);

    EXPECT_EQ(table.total_number_of_projectors(), 3);
    EXPECT_EQ(table.number_of_channels(0), 3);

    // All 3 channels should map back to beta index 0
    for (int channel_index = 0; channel_index < 3; channel_index++) {
        EXPECT_EQ(table.to_beta_index(0, channel_index), 0);
        EXPECT_EQ(table.angular_momentum(0, channel_index), 1);
    }

    // Check magnetic quantum numbers follow alternating-sign convention: 0, 1, -1
    EXPECT_EQ(table.magnetic_quantum_number(0, 0), 0);
    EXPECT_EQ(table.magnetic_quantum_number(0, 1), 1);
    EXPECT_EQ(table.magnetic_quantum_number(0, 2), -1);
}

/**
 * Test building with both s (l=0) and p (l=1) projectors.
 * Channels: 1 (s) + 3 (p) = 4 total channels per atom.
 */
TEST(ProjectorMappingTable, MixedSAndPProjectors) {
    Crystal crystal = build_simple_crystal(2);
    PseudopotentialSet pseudopotential_set = build_pseudopotential_set_with_projectors({0, 1});
    int number_of_species_types = 1;

    ProjectorMappingTable table;
    table.build(crystal, pseudopotential_set, number_of_species_types);

    // 4 channels per type, 2 atoms of this type → 8 total
    EXPECT_EQ(table.number_of_channels(0), 4);
    EXPECT_EQ(table.total_number_of_projectors(), 8);

    // Channel 0: from s projector (beta_index=0, l=0, m=0)
    EXPECT_EQ(table.to_beta_index(0, 0), 0);
    EXPECT_EQ(table.angular_momentum(0, 0), 0);
    EXPECT_EQ(table.magnetic_quantum_number(0, 0), 0);

    // Channels 1-3: from p projector (beta_index=1, l=1)
    EXPECT_EQ(table.to_beta_index(0, 1), 1);
    EXPECT_EQ(table.angular_momentum(0, 1), 1);
    EXPECT_EQ(table.magnetic_quantum_number(0, 1), 0);

    EXPECT_EQ(table.to_beta_index(0, 2), 1);
    EXPECT_EQ(table.angular_momentum(0, 2), 1);
    EXPECT_EQ(table.magnetic_quantum_number(0, 2), 1);

    EXPECT_EQ(table.to_beta_index(0, 3), 1);
    EXPECT_EQ(table.angular_momentum(0, 3), 1);
    EXPECT_EQ(table.magnetic_quantum_number(0, 3), -1);
}

/**
 * Test with a d-orbital projector (l=2). Should produce 5 channels.
 */
TEST(ProjectorMappingTable, DProjectorFiveChannels) {
    Crystal crystal = build_simple_crystal(1);
    PseudopotentialSet pseudopotential_set = build_pseudopotential_set_with_projectors({2});
    int number_of_species_types = 1;

    ProjectorMappingTable table;
    table.build(crystal, pseudopotential_set, number_of_species_types);

    EXPECT_EQ(table.total_number_of_projectors(), 5);
    EXPECT_EQ(table.number_of_channels(0), 5);

    // All channels should have l=2
    for (int channel_index = 0; channel_index < 5; channel_index++) {
        EXPECT_EQ(table.angular_momentum(0, channel_index), 2);
    }

    // Magnetic quantum numbers: 0, 1, -1, 2, -2
    EXPECT_EQ(table.magnetic_quantum_number(0, 0), 0);
    EXPECT_EQ(table.magnetic_quantum_number(0, 1), 1);
    EXPECT_EQ(table.magnetic_quantum_number(0, 2), -1);
    EXPECT_EQ(table.magnetic_quantum_number(0, 3), 2);
    EXPECT_EQ(table.magnetic_quantum_number(0, 4), -2);
}

/**
 * Test the raw table accessors return non-empty vectors after build.
 */
TEST(ProjectorMappingTable, RawTableAccessors) {
    Crystal crystal = build_simple_crystal(1);
    PseudopotentialSet pseudopotential_set = build_pseudopotential_set_with_projectors({0, 1});
    int number_of_species_types = 1;

    ProjectorMappingTable table;
    table.build(crystal, pseudopotential_set, number_of_species_types);

    EXPECT_EQ(static_cast<int>(table.channels_per_type_table().size()), 1);
    EXPECT_EQ(table.channels_per_type_table()[0], 4);
    EXPECT_EQ(static_cast<int>(table.to_beta_index_table().size()), 1);
    EXPECT_EQ(static_cast<int>(table.to_beta_index_table()[0].size()), 4);
    EXPECT_EQ(static_cast<int>(table.combined_lm_index_table().size()), 1);
    EXPECT_EQ(static_cast<int>(table.angular_momentum_table().size()), 1);
}

/**
 * Test combined_lm_index values for l=0 and l=1 projectors.
 */
TEST(ProjectorMappingTable, CombinedLmIndex) {
    Crystal crystal = build_simple_crystal(1);
    PseudopotentialSet pseudopotential_set = build_pseudopotential_set_with_projectors({0, 1});
    int number_of_species_types = 1;

    ProjectorMappingTable table;
    table.build(crystal, pseudopotential_set, number_of_species_types);

    // l=0, m=0: combined = 0² = 0
    EXPECT_EQ(table.combined_lm_index(0, 0), 0);

    // l=1, m=0: combined = 1² = 1
    EXPECT_EQ(table.combined_lm_index(0, 1), 1);

    // l=1, m=1: combined = 1² + 2*1 - 1 = 2
    EXPECT_EQ(table.combined_lm_index(0, 2), 2);

    // l=1, m=-1: combined = 1² + 2*1 = 3
    EXPECT_EQ(table.combined_lm_index(0, 3), 3);
}
