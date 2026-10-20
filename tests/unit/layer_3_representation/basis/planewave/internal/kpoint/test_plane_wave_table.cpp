#include <gtest/gtest.h>

#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/kpoint/plane_wave_table.hpp"

using namespace q3;

/**
 * Helper to build an FCC silicon crystal.
 */
static Crystal make_silicon_crystal() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263);
    AtomSpecies silicon{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/**
 * Helper to build and initialize a PlaneWaveTable for a 2x2x2 k-grid.
 */
static PlaneWaveTable make_initialized_table() {
    Crystal silicon = make_silicon_crystal();
    GVectorsGrid g_vectors_grid;
    g_vectors_grid.generate(silicon, 64.0, 25, 25, 25);

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 0, 0, 0);

    PlaneWaveTable table;
    table.initialize(g_vectors_grid, silicon, kpoint_grid, 16.0);
    return table;
}

// ── Initialized table has correct number of k-points ──
TEST(PlaneWaveTable, HasCorrectNumberOfKpoints) {
    PlaneWaveTable table = make_initialized_table();

    EXPECT_EQ(static_cast<int>(table.kpoint_number_of_plane_waves().size()), 8);
    EXPECT_EQ(static_cast<int>(table.kpoint_g_vector_indices().size()), 8);
    EXPECT_EQ(static_cast<int>(table.kpoint_kinetic_energy_squared().size()), 8);
}

// ── Each k-point has positive number of plane waves ──
TEST(PlaneWaveTable, EachKpointHasPositivePlaneWaveCount) {
    PlaneWaveTable table = make_initialized_table();

    for (int kpoint_index = 0; kpoint_index < 8; kpoint_index++) {
        EXPECT_GT(table.number_of_plane_waves_by_kpoint_index(kpoint_index), 0)
            << "k-point " << kpoint_index << " has zero plane waves";
    }
}

// ── Per-k-point accessors are consistent with bulk accessors ──
TEST(PlaneWaveTable, PerKpointAccessorsConsistentWithBulk) {
    PlaneWaveTable table = make_initialized_table();
    const auto& all_indices = table.kpoint_g_vector_indices();
    const auto& all_counts = table.kpoint_number_of_plane_waves();
    const auto& all_kinetic_energies = table.kpoint_kinetic_energy_squared();

    for (int kpoint_index = 0; kpoint_index < 8; kpoint_index++) {
        EXPECT_EQ(table.g_vector_indices_by_kpoint_index(kpoint_index), all_indices[kpoint_index]);
        EXPECT_EQ(table.number_of_plane_waves_by_kpoint_index(kpoint_index),
                  all_counts[kpoint_index]);
        EXPECT_EQ(table.kinetic_energy_squared_by_kpoint_index(kpoint_index),
                  all_kinetic_energies[kpoint_index]);
    }
}

// ── Number of plane waves matches g_vector_indices size ──
TEST(PlaneWaveTable, PlaneWaveCountMatchesIndicesSize) {
    PlaneWaveTable table = make_initialized_table();

    for (int kpoint_index = 0; kpoint_index < 8; kpoint_index++) {
        EXPECT_EQ(table.number_of_plane_waves_by_kpoint_index(kpoint_index),
                  static_cast<int>(table.g_vector_indices_by_kpoint_index(kpoint_index).size()));
    }
}

// ── Number of plane waves matches kinetic_energy_squared size ──
TEST(PlaneWaveTable, PlaneWaveCountMatchesKineticEnergySize) {
    PlaneWaveTable table = make_initialized_table();

    for (int kpoint_index = 0; kpoint_index < 8; kpoint_index++) {
        EXPECT_EQ(
            static_cast<int>(table.kinetic_energy_squared_by_kpoint_index(kpoint_index).size()),
            table.number_of_plane_waves_by_kpoint_index(kpoint_index));
    }
}
