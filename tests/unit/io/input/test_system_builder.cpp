/**
 * build_global_kpoint_grid_from_input: which meshes get reduced to the
 * irreducible wedge, and which are left alone.
 *
 * The reduction used to be gated on the mesh's *first* dimension being greater
 * than one, so a 1x4x4 mesh -- a slab or a wire, sampled across the two
 * periodic directions -- kept all sixteen points and every symmetry-equivalent
 * duplicate was solved again. Correct, and up to a point group slower.
 */

#include "io/input/system_builder.hpp"

#include <gtest/gtest.h>

using namespace q3;

namespace {

/// FCC silicon: the full 48-operation point group, so a wedge is much smaller.
Crystal make_silicon() {
    Crystal crystal;
    crystal.set_from_lattice_type(2, 10.263);
    crystal.add_species(AtomSpecies{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"});
    crystal.add_atom({0, {0.0, 0.0, 0.0}});
    crystal.add_atom({0, {0.25, 0.25, 0.25}});
    crystal.compute_derived_quantities();
    return crystal;
}

InputData make_input(int dimension_1, int dimension_2, int dimension_3, int shift = 0) {
    InputData input;
    input.kpoint_grid_dimension_1 = dimension_1;
    input.kpoint_grid_dimension_2 = dimension_2;
    input.kpoint_grid_dimension_3 = dimension_3;
    input.kpoint_grid_shift_1 = shift;
    input.kpoint_grid_shift_2 = shift;
    input.kpoint_grid_shift_3 = shift;
    return input;
}

Real total_weight(const KPointsGrid& grid) {
    Real sum = 0.0;
    for (const auto& kpoint : grid.kpoints()) {
        sum += kpoint.weight;
    }
    return sum;
}

} // namespace

TEST(SystemBuilderKPointGridTest, ReducesAMeshThatIsFlatInTheFirstDimension) {
    const Crystal silicon = make_silicon();
    const KPointsGrid grid = build_global_kpoint_grid_from_input(make_input(1, 4, 4), silicon);

    EXPECT_LT(grid.number_of_kpoints(), 16) << "a 1x4x4 mesh has a wedge like any other";
    EXPECT_GT(grid.number_of_kpoints(), 0);
    EXPECT_NEAR(total_weight(grid), 2.0, 1e-12) << "reduction moves weight, never loses it";
}

TEST(SystemBuilderKPointGridTest, StillReducesAMeshDenseInEveryDimension) {
    const Crystal silicon = make_silicon();
    const KPointsGrid grid = build_global_kpoint_grid_from_input(make_input(4, 4, 4), silicon);

    EXPECT_LT(grid.number_of_kpoints(), 64);
    EXPECT_NEAR(total_weight(grid), 2.0, 1e-12);
}

TEST(SystemBuilderKPointGridTest, LeavesASinglePointAlone) {
    const Crystal silicon = make_silicon();
    /**
     * One point is its own wedge, shifted or not, so the reduction has nothing
     * to do and is not called.
     */
    for (int shift : {0, 1}) {
        const KPointsGrid grid =
            build_global_kpoint_grid_from_input(make_input(1, 1, 1, shift), silicon);
        EXPECT_EQ(grid.number_of_kpoints(), 1) << "shift " << shift;
        EXPECT_NEAR(total_weight(grid), 2.0, 1e-12) << "shift " << shift;
    }
}

TEST(SystemBuilderKPointGridTest, DisableSymmetryKeepsEveryPointOfAFlatMesh) {
    const Crystal silicon = make_silicon();
    InputData input = make_input(1, 4, 4);
    input.disable_symmetry = true;

    const KPointsGrid grid = build_global_kpoint_grid_from_input(input, silicon);
    EXPECT_EQ(grid.number_of_kpoints(), 16);
    EXPECT_NEAR(total_weight(grid), 2.0, 1e-12);
}

TEST(SystemBuilderKPointGridTest, AnExplicitKpointListIsNeverReduced) {
    const Crystal silicon = make_silicon();
    InputData input = make_input(1, 4, 4);
    input.kpoints_list = {
        {{0.0, 0.0, 0.0}, 1.0},
        {{0.25, 0.25, 0.25}, 1.0},
        {{0.5, 0.5, 0.5}, 1.0},
    };

    const KPointsGrid grid = build_global_kpoint_grid_from_input(input, silicon);
    EXPECT_EQ(grid.number_of_kpoints(), 3) << "a band path is the caller's list, not a mesh";
}

namespace {
/// Two same-species atoms (a CsCl-type cell) with per-atom moments.
InputData make_two_site_input(std::vector<Real> moments, int spin_components = 2) {
    InputData input;
    input.number_of_spin_components = spin_components;
    input.species = {AtomSpecies{"Fe", 55.845, 8.0, "Fe.UPF"}};
    input.atoms = {Atom{0, {0.0, 0.0, 0.0}}, Atom{0, {0.5, 0.5, 0.5}}};
    input.atomic_starting_magnetization = std::move(moments);
    return input;
}
} // namespace

TEST(SystemBuilderSiteSymmetryTest, OpposedMomentsOnOneSpeciesSplitIt) {
    EXPECT_EQ(site_symmetry_classes_from_input(make_two_site_input({0.5, -0.5})),
              (std::vector<int>{0, 1}));
}

TEST(SystemBuilderSiteSymmetryTest, UniformMomentsKeepTheSpeciesPartition) {
    /**
     * An empty result leaves the crystal untouched, which is what keeps
     * per-atom input that is uniform per species bit-identical to per-species
     * input.
     */
    EXPECT_TRUE(site_symmetry_classes_from_input(make_two_site_input({0.5, 0.5})).empty());
    EXPECT_TRUE(site_symmetry_classes_from_input(make_two_site_input({})).empty());
    EXPECT_TRUE(site_symmetry_classes_from_input(make_two_site_input({0.5, -0.5}, 1)).empty())
        << "an unpolarized run has no moments to keep apart";
}

TEST(SystemBuilderSiteSymmetryTest, TheBuiltCrystalCarriesThePartition) {
    InputData input = make_two_site_input({0.5, -0.5});
    input.bravais_lattice_index = 1;
    input.cell_dimension_1 = 5.42;
    const Crystal crystal = build_crystal_from_input(input, input.species);
    EXPECT_EQ(crystal.site_symmetry_class(0), 0);
    EXPECT_EQ(crystal.site_symmetry_class(1), 1);
    EXPECT_EQ(SymmetryAnalyzer::number_of_symmetry_operations(crystal), 48);
}
