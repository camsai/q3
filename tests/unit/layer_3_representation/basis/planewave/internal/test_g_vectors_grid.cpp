#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include <gtest/gtest.h>

using namespace q3;

static execution::PlanewaveSerialExecutionPolicy serial_policy;

static Crystal make_silicon_structure() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263); // FCC Silicon
    AtomSpecies silicon{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

TEST(PlanewaveBasisTest, GenerationAndAccessors) {
    Crystal silicon = make_silicon_structure();
    PlanewaveBasis basis;
    // Wavefunction cutoff: 16 Ry, Charge density cutoff: 64 Ry
    basis.generate(serial_policy, silicon, 16.0, 64.0);

    EXPECT_DOUBLE_EQ(basis.wavefunction_energy_cutoff(), 16.0);
    EXPECT_DOUBLE_EQ(basis.charge_density_energy_cutoff(), 64.0);
    EXPECT_GT(basis.number_of_g_vectors(), 0);
    EXPECT_GT(basis.maximum_plane_waves_per_kpoint(), 0);
}

TEST(PlanewaveBasisTest, GStartChecks) {
    Crystal silicon = make_silicon_structure();
    PlanewaveBasis basis;
    basis.generate(serial_policy, silicon, 16.0, 64.0);
    // G=0 should exist and be at index 0 of g_vectors
    int gstart = basis.first_nonzero_g_index();
    EXPECT_EQ(gstart, 1);
    EXPECT_NEAR(basis.g_vectors()[0].magnitude_squared, 0.0, 1e-12);
}

TEST(PlanewaveBasisTest, ComputeGK) {
    Crystal silicon = make_silicon_structure();
    PlanewaveBasis basis;
    basis.generate(serial_policy, silicon, 16.0, 64.0);

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    // Compute k+G at Gamma (0,0,0)
    basis.compute_gk(Vector3D(0.0, 0.0, 0.0), silicon, g_vector_indices, kinetic_energy_squared);

    EXPECT_GT(g_vector_indices.size(), 0u);
    EXPECT_EQ(g_vector_indices.size(), kinetic_energy_squared.size());
    // Minimum kinetic energy at Gamma should be 0.0
    EXPECT_NEAR(kinetic_energy_squared[0], 0.0, 1e-12);
}
