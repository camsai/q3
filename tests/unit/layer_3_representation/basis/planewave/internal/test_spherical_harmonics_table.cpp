#include <gtest/gtest.h>

#include <cmath>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/spherical_harmonics_table.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

static execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Helper to build a silicon crystal and planewave basis for harmonics tests.
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

static PlanewaveBasis make_planewave_basis(const Crystal& crystal_structure) {
    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 16.0, 64.0);
    return basis;
}

// ── Default construction gives zero angular momentum and components ──
TEST(SphericalHarmonicsTable, DefaultConstructionHasZeroAngularMomentum) {
    SphericalHarmonicsTable table;

    EXPECT_EQ(table.maximum_angular_momentum(), 0);
    EXPECT_EQ(table.number_of_lm_components(), 1);
}

// ── After init with l_max=0, the table has 1 component ──
TEST(SphericalHarmonicsTable, InitWithAngularMomentumZeroHasOneComponent) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveBasis basis = make_planewave_basis(silicon);
    SphericalHarmonicsTable table;
    table.init(silicon, basis, 0);

    EXPECT_EQ(table.maximum_angular_momentum(), 0);
    EXPECT_EQ(table.number_of_lm_components(), 1);
}

// ── After init with l_max=2, the table has 9 components ──
TEST(SphericalHarmonicsTable, InitWithAngularMomentumTwoHasNineComponents) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveBasis basis = make_planewave_basis(silicon);
    SphericalHarmonicsTable table;
    table.init(silicon, basis, 2);

    EXPECT_EQ(table.maximum_angular_momentum(), 2);
    EXPECT_EQ(table.number_of_lm_components(), 9);
}

// ── Y_00 at any G-vector = 1/sqrt(4π) ──
TEST(SphericalHarmonicsTable, Y00IsConstant) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveBasis basis = make_planewave_basis(silicon);
    SphericalHarmonicsTable table;
    table.init(silicon, basis, 0);

    Real expected_y00 = 1.0 / std::sqrt(constants::four_pi);
    int number_of_g_vectors = basis.number_of_g_vectors();
    ASSERT_GT(number_of_g_vectors, 0);

    for (int g_index = 0; g_index < std::min(number_of_g_vectors, 20); g_index++) {
        EXPECT_NEAR(table.value(g_index, 0), expected_y00, 1e-10)
            << "Y_00 differs at g_index " << g_index;
    }
}

// ── Accessing out-of-range lm index returns zero ──
TEST(SphericalHarmonicsTable, OutOfRangeLmIndexReturnsZero) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveBasis basis = make_planewave_basis(silicon);
    SphericalHarmonicsTable table;
    table.init(silicon, basis, 1);

    // Table has (1+1)^2=4 components. Requesting index 10 should return 0.
    EXPECT_DOUBLE_EQ(table.value(0, 10), 0.0);
}

// ── Number of lm components formula: (l_max+1)^2 ──
TEST(SphericalHarmonicsTable, NumberOfLmComponentsFormula) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveBasis basis = make_planewave_basis(silicon);

    for (int angular_momentum = 0; angular_momentum <= 4; angular_momentum++) {
        SphericalHarmonicsTable table;
        table.init(silicon, basis, angular_momentum);
        int expected_components = (angular_momentum + 1) * (angular_momentum + 1);
        EXPECT_EQ(table.number_of_lm_components(), expected_components)
            << "l_max = " << angular_momentum;
    }
}
