#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/kpoint/plane_wave_selector.hpp"

using namespace q3;

/**
 * Helper to build an FCC silicon crystal and G-vector grid for selector tests.
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

static GVectorsGrid make_g_vectors_grid(const Crystal& crystal_structure) {
    GVectorsGrid g_vectors_grid;
    // Use large enough FFT dimensions for the cutoff
    g_vectors_grid.generate(crystal_structure, 64.0, 25, 25, 25);
    return g_vectors_grid;
}

// ── Select at Gamma point returns at least one plane wave ──
TEST(PlaneWaveSelector, SelectAtGammaReturnsPlaneWaves) {
    Crystal silicon = make_silicon_crystal();
    GVectorsGrid g_vectors_grid = make_g_vectors_grid(silicon);

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    PlaneWaveSelector::select(g_vectors_grid,
                              Vector3D(0.0, 0.0, 0.0),
                              silicon,
                              16.0,
                              g_vector_indices,
                              kinetic_energy_squared);

    EXPECT_GT(g_vector_indices.size(), 0u);
    EXPECT_EQ(g_vector_indices.size(), kinetic_energy_squared.size());
}

// ── At Gamma, the first kinetic energy should be zero (G=0) ──
TEST(PlaneWaveSelector, FirstKineticEnergyAtGammaIsZero) {
    Crystal silicon = make_silicon_crystal();
    GVectorsGrid g_vectors_grid = make_g_vectors_grid(silicon);

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    PlaneWaveSelector::select(g_vectors_grid,
                              Vector3D(0.0, 0.0, 0.0),
                              silicon,
                              16.0,
                              g_vector_indices,
                              kinetic_energy_squared);

    ASSERT_FALSE(kinetic_energy_squared.empty());
    EXPECT_NEAR(kinetic_energy_squared[0], 0.0, 1e-12);
}

// ── Kinetic energies are sorted in non-decreasing order ──
TEST(PlaneWaveSelector, KineticEnergiesAreSortedNonDecreasing) {
    Crystal silicon = make_silicon_crystal();
    GVectorsGrid g_vectors_grid = make_g_vectors_grid(silicon);

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    PlaneWaveSelector::select(g_vectors_grid,
                              Vector3D(0.0, 0.0, 0.0),
                              silicon,
                              16.0,
                              g_vector_indices,
                              kinetic_energy_squared);

    ASSERT_GT(kinetic_energy_squared.size(), 1u);
    EXPECT_TRUE(std::is_sorted(kinetic_energy_squared.begin(), kinetic_energy_squared.end()));
}

// ── All kinetic energies are within the wavefunction cutoff ──
TEST(PlaneWaveSelector, AllKineticEnergiesWithinCutoff) {
    Crystal silicon = make_silicon_crystal();
    GVectorsGrid g_vectors_grid = make_g_vectors_grid(silicon);

    Real energy_cutoff_wavefunction = 16.0;
    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    PlaneWaveSelector::select(g_vectors_grid,
                              Vector3D(0.0, 0.0, 0.0),
                              silicon,
                              energy_cutoff_wavefunction,
                              g_vector_indices,
                              kinetic_energy_squared);

    for (size_t index = 0; index < kinetic_energy_squared.size(); index++) {
        EXPECT_LE(kinetic_energy_squared[index], energy_cutoff_wavefunction + 1e-6)
            << "Kinetic energy at index " << index << " exceeds cutoff";
    }
}

// ── Higher cutoff yields more or equal plane waves ──
TEST(PlaneWaveSelector, HigherCutoffYieldsMorePlaneWaves) {
    Crystal silicon = make_silicon_crystal();
    GVectorsGrid g_vectors_grid = make_g_vectors_grid(silicon);

    std::vector<int> indices_low, indices_high;
    std::vector<Real> kinetic_energies_low, kinetic_energies_high;
    PlaneWaveSelector::select(
        g_vectors_grid, Vector3D(0.0, 0.0, 0.0), silicon, 8.0, indices_low, kinetic_energies_low);
    PlaneWaveSelector::select(g_vectors_grid,
                              Vector3D(0.0, 0.0, 0.0),
                              silicon,
                              16.0,
                              indices_high,
                              kinetic_energies_high);

    EXPECT_GE(indices_high.size(), indices_low.size());
}

// ── Non-Gamma k-point produces non-zero first kinetic energy ──
TEST(PlaneWaveSelector, NonGammaKPointProducesNonZeroFirstKineticEnergy) {
    Crystal silicon = make_silicon_crystal();
    GVectorsGrid g_vectors_grid = make_g_vectors_grid(silicon);

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    PlaneWaveSelector::select(g_vectors_grid,
                              Vector3D(0.25, 0.25, 0.25),
                              silicon,
                              16.0,
                              g_vector_indices,
                              kinetic_energy_squared);

    ASSERT_FALSE(kinetic_energy_squared.empty());
    EXPECT_GT(kinetic_energy_squared[0], 1e-10);
}
