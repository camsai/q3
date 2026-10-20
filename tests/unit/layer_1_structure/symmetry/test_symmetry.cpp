#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

using namespace q3;

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

TEST(SymmetryTest, SpacegroupDetection) {
    Crystal silicon = make_silicon_structure();
    std::string spacegroup = SymmetryAnalyzer::find_spacegroup(silicon);
    // Silicon space group is Fd-3m (No. 227)
    EXPECT_TRUE(spacegroup.find("Fd-3m") != std::string::npos ||
                spacegroup.find("227") != std::string::npos);
}

TEST(SymmetryTest, NumberOfSymmetryOperations) {
    Crystal silicon = make_silicon_structure();
    int number_of_operations = SymmetryAnalyzer::number_of_symmetry_operations(silicon);
    // FCC silicon should have 48 symmetry operations
    EXPECT_EQ(number_of_operations, 48);
}

TEST(SymmetryTest, KPointsReduction) {
    Crystal silicon = make_silicon_structure();
    KPointsGrid full_grid;
    full_grid.generate(2, 2, 2, 0, 0, 0);
    KPointsGrid kpoints = SymmetryAnalyzer::reduce_kpoints(silicon, full_grid);
    // Monkhorst-Pack 2x2x2 grid should reduce to a smaller number of k-points in IBZ
    EXPECT_GT(kpoints.number_of_kpoints(), 0);
    EXPECT_LE(kpoints.number_of_kpoints(), 8);
}

TEST(SymmetryTest, ReducedGeneratedGridsPreserveDimensionsShiftsWeightsAndReduction) {
    Crystal silicon = make_silicon_structure();
    const std::array<int, 3> dimensions = {4, 4, 4};
    const std::array<std::array<int, 3>, 2> shift_cases = {{{0, 0, 0}, {1, 1, 1}}};

    for (bool use_internal_reduction : {false, true}) {
        for (const auto& shifts : shift_cases) {
            SCOPED_TRACE(::testing::Message() << "internal=" << use_internal_reduction
                                              << " shifts=" << shifts[0] << shifts[1] << shifts[2]);
            KPointsGrid full_grid;
            full_grid.generate(
                dimensions[0], dimensions[1], dimensions[2], shifts[0], shifts[1], shifts[2]);

            KPointsGrid reduced_grid =
                use_internal_reduction
                    ? SymmetryAnalyzer::reduce_kpoints_internal(silicon, full_grid)
                    : SymmetryAnalyzer::reduce_kpoints(silicon, full_grid);

            EXPECT_EQ(reduced_grid.grid_dimensions(), dimensions);
            EXPECT_EQ(reduced_grid.grid_shifts(), shifts);
            EXPECT_LT(reduced_grid.number_of_kpoints(), full_grid.number_of_kpoints());

            Real total_weight = 0.0;
            for (const auto& kpoint : reduced_grid.kpoints()) {
                total_weight += kpoint.weight;
            }
            EXPECT_NEAR(total_weight, 2.0, 1e-12);
        }
    }
}

TEST(SymmetryTest, VectorSymmetrization) {
    Crystal silicon = make_silicon_structure();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(silicon);
    std::vector<Vector3D> forces(silicon.number_of_atoms(), Vector3D(1.0, 1.0, 1.0));
    SymmetryAnalyzer::symmetrize_vector(silicon, operations, forces);
    // For a highly symmetric structure, arbitrary forces get symmetrized out (averaged to 0)
    for (const auto& force : forces) {
        EXPECT_NEAR(force.x, 0.0, 1e-5);
        EXPECT_NEAR(force.y, 0.0, 1e-5);
        EXPECT_NEAR(force.z, 0.0, 1e-5);
    }
}

/**
 * Displacing one atom of the 2-atom diamond-structure cell along a single
 * Cartesian axis reduces the space group from Fd-3m (48 operations) to a
 * strictly smaller subgroup, while still leaving the crystal genuinely
 * symmetric (just under fewer operations) -- not asymmetric. This is the
 * regression case for a bug (found 2026-07-26 while validating the
 * Gamma-phonon workflow, see TODOs.md) where symmetrize_vector used the
 * rotation matrix un-transposed. Symmetrizing a vector field like forces
 * applies the TRANSPOSE of the position-rotation matrix, paired with the image
 * atom: the group average needs the covariant transform of the INVERSE
 * operation, and that is exactly R^T (see symmetrize_vector's own derivation).
 * The existing VectorSymmetrization
 * test above cannot catch this: an isotropic (1,1,1) input under the FULL
 * 48-operation cubic group averages to zero regardless of whether the rotation
 * is transposed, since the group has no invariant polar direction either way.
 */
static Crystal make_displaced_silicon_structure() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263); // FCC Silicon, same cell as above
    AtomSpecies silicon{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(silicon);
    /**
     * Same fractional coordinates as examples/data/materials/silicon.json
     * (atom 0 at the equilibrium position, then displaced by exactly
     * -0.01 Angstrom along CARTESIAN x -- NOT fractional x, which for the
     * ibrav=2 cell points along the (-1,0,1) Cartesian direction instead)
     * used to independently confirm this exact fixed point against the
     * reference QE run documented in the project's tracked follow-up
     * items. Atom 1 untouched.
     */
    crystal_structure.add_atom({0, {0.8768285781478513, 0.8731714218521487, 0.8768285781478513}});
    crystal_structure.add_atom({0, {0.125, 0.125, 0.125}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

TEST(SymmetryTest, VectorSymmetrizationUnderReducedSymmetryPreservesCompatibleField) {
    Crystal displaced_silicon = make_displaced_silicon_structure();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(displaced_silicon);
    /**
     * A genuine reduced-symmetry case: strictly fewer than the 48 operations
     * of the undisplaced crystal, but still a real (non-trivial) point group.
     */
    ASSERT_GT(operations.size(), 1u);
    ASSERT_LT(operations.size(), 48u);

    /**
     * A force field that already obeys Newton's third law AND the crystal's
     * residual symmetry (pure x-component, equal and opposite) must be an
     * exact fixed point of symmetrization -- symmetrizing an already-
     * symmetric field must not change it.
     */
    std::vector<Vector3D> compatible_forces = {{0.00684513, 0.0, 0.0}, {-0.00684513, 0.0, 0.0}};
    std::vector<Vector3D> expected = compatible_forces;
    SymmetryAnalyzer::symmetrize_vector(displaced_silicon, operations, compatible_forces);
    for (std::size_t atom_index = 0; atom_index < compatible_forces.size(); atom_index++) {
        EXPECT_NEAR(compatible_forces[atom_index].x, expected[atom_index].x, 1e-9);
        EXPECT_NEAR(compatible_forces[atom_index].y, expected[atom_index].y, 1e-9);
        EXPECT_NEAR(compatible_forces[atom_index].z, expected[atom_index].z, 1e-9);
    }
}

TEST(SymmetryTest, VectorSymmetrizationUnderReducedSymmetryProjectsOutIncompatibleComponents) {
    Crystal displaced_silicon = make_displaced_silicon_structure();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(displaced_silicon);
    ASSERT_GT(operations.size(), 1u);
    ASSERT_LT(operations.size(), 48u);

    /**
     * An isotropic field is NOT compatible with this crystal's residual
     * symmetry (only the x-component survives); the y and z components must
     * be projected out to exactly zero, not leaked into spurious nonzero
     * values the way the pre-fix (untransposed rotation) code did.
     */
    std::vector<Vector3D> forces = {{1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0}};
    SymmetryAnalyzer::symmetrize_vector(displaced_silicon, operations, forces);
    for (const auto& force : forces) {
        EXPECT_NEAR(force.y, 0.0, 1e-9);
        EXPECT_NEAR(force.z, 0.0, 1e-9);
    }
    /**
     * The x-component is not projected away (it is compatible with the
     * residual symmetry), so it must remain nonzero.
     */
    EXPECT_GT(std::abs(forces[0].x), 1e-3);
}

/**
 * reduce_kpoints_internal() (the default automatic Monkhorst-Pack + IBZ
 * reduction path) used to build k-point equivalence classes from
 * compute_bravais_lattice_rotations() -- the Bravais LATTICE's holohedral
 * point group, which depends only on the lattice vectors and ignores the
 * atomic basis entirely. For a structure whose basis reduces the true
 * symmetry below the lattice's (any finite-difference-displaced structure,
 * like the one below), that group is too large: it silently treats
 * k-points as equivalent under rotations that are not actual symmetries of
 * the decorated crystal, corrupting the k-point-summed forces and stress
 * downstream (found 2026-07-27 while following up on the IBZ vector
 * reconstruction issue flagged in TODOs.md while building the Gamma-phonon
 * workflow). Fixed by reusing get_symmetry_operations() -- the same
 * basis-aware, spglib-backed detection already used for density
 * symmetrization -- instead of re-deriving the Bravais-only group by hand.
 *
 * This is a cross-check regression test: reduce_kpoints_internal must now
 * agree with the independent reduce_kpoints() (spglib's own
 * spg_get_ir_reciprocal_mesh, which already accounts for the atomic basis)
 * on the number of irreducible k-points for a reduced-symmetry structure.
 * Before the fix, reduce_kpoints_internal over-reduced using the full
 * 48-operation cubic group instead of this structure's actual 8-operation
 * point group.
 */
/**
 * The reduction had no gate on WHICH k-points it returns: the existing cases
 * compare counts and total weight, both of which survive a renumbering that
 * emits different representatives. That matters because the per-representative
 * force and stress reconstruction starts from whichever star member is
 * returned, so a silent change of representative is a silent change of the
 * quantity being reconstructed.
 *
 * These two pin the properties a renumbering would break, stated as invariants
 * rather than as a table of expected numbers so they hold for any crystal and
 * any grid.
 */
TEST(SymmetryTest, ReducedKpointsAreUnmodifiedMembersOfTheInputGrid) {
    /**
     * Every returned k-point must be one of the input grid's own points, to the
     * bit. A representative that has been folded, re-derived, or reconstructed
     * on the way out will drift -- most visibly at the zone boundary, where
     * +0.5 and -0.5 name the same plane but are different numbers.
     */
    Crystal silicon = make_silicon_structure();

    for (const auto& grid_shape : std::vector<std::array<int, 6>>{
             {4, 4, 4, 0, 0, 0}, {3, 3, 3, 0, 0, 0}, {2, 2, 2, 1, 1, 1}, {4, 4, 4, 1, 1, 1}}) {
        KPointsGrid full_grid;
        full_grid.generate(grid_shape[0],
                           grid_shape[1],
                           grid_shape[2],
                           grid_shape[3],
                           grid_shape[4],
                           grid_shape[5]);
        KPointsGrid reduced = SymmetryAnalyzer::reduce_kpoints_internal(silicon, full_grid);

        for (int reduced_index = 0; reduced_index < reduced.number_of_kpoints(); reduced_index++) {
            const Vector3D& candidate = reduced[reduced_index].coordinates;
            bool found_in_input = false;
            for (int grid_index = 0; grid_index < full_grid.number_of_kpoints(); grid_index++) {
                const Vector3D& grid_point = full_grid[grid_index].coordinates;
                if (std::abs(candidate.x - grid_point.x) < 1e-12 &&
                    std::abs(candidate.y - grid_point.y) < 1e-12 &&
                    std::abs(candidate.z - grid_point.z) < 1e-12) {
                    found_in_input = true;
                    break;
                }
            }
            EXPECT_TRUE(found_in_input)
                << "reduced k-point " << reduced_index << " (" << candidate.x << ", " << candidate.y
                << ", " << candidate.z << ") is not a point of the " << grid_shape[0] << "x"
                << grid_shape[1] << "x" << grid_shape[2] << " grid";
        }
    }
}

TEST(SymmetryTest, ReducedKpointsKeepTheInputGridOrdering) {
    /**
     * Representatives are emitted in increasing grid order, so the reduced list
     * is a subsequence of the input grid rather than a reordering of it. A
     * reduction that walked the mesh differently would still return the right
     * count and the right total weight, and this is what would notice.
     */
    Crystal silicon = make_silicon_structure();
    KPointsGrid full_grid;
    full_grid.generate(4, 4, 4, 0, 0, 0);
    KPointsGrid reduced = SymmetryAnalyzer::reduce_kpoints_internal(silicon, full_grid);

    int previous_grid_position = -1;
    for (int reduced_index = 0; reduced_index < reduced.number_of_kpoints(); reduced_index++) {
        const Vector3D& candidate = reduced[reduced_index].coordinates;
        int grid_position = -1;
        for (int grid_index = 0; grid_index < full_grid.number_of_kpoints(); grid_index++) {
            const Vector3D& grid_point = full_grid[grid_index].coordinates;
            if (std::abs(candidate.x - grid_point.x) < 1e-12 &&
                std::abs(candidate.y - grid_point.y) < 1e-12 &&
                std::abs(candidate.z - grid_point.z) < 1e-12) {
                grid_position = grid_index;
                break;
            }
        }
        ASSERT_GE(grid_position, 0) << "reduced k-point " << reduced_index << " is off-grid";
        EXPECT_GT(grid_position, previous_grid_position)
            << "representatives are not in increasing grid order at " << reduced_index;
        previous_grid_position = grid_position;
    }
}

TEST(SymmetryTest, ReduceKpointsInternalRejectsAnUngeneratedGrid) {
    /**
     * The reduction reads the caller's k-point list instead of rebuilding the
     * mesh, so a grid that carries dimensions but no points is a programming
     * error rather than an empty reduction.
     */
    Crystal silicon = make_silicon_structure();
    KPointsGrid metadata_only;
    metadata_only.set_grid_metadata({4, 4, 4}, {0, 0, 0});
    EXPECT_THROW(SymmetryAnalyzer::reduce_kpoints_internal(silicon, metadata_only),
                 std::runtime_error);
}

TEST(SymmetryTest, ReduceKpointsInternalUsesActualCrystalSymmetryNotBravaisLattice) {
    Crystal displaced_silicon = make_displaced_silicon_structure();

    KPointsGrid full_grid;
    full_grid.generate(6, 6, 6, 1, 1, 1);

    KPointsGrid internal_reduction =
        SymmetryAnalyzer::reduce_kpoints_internal(displaced_silicon, full_grid);
    KPointsGrid spglib_reduction = SymmetryAnalyzer::reduce_kpoints(displaced_silicon, full_grid);

    EXPECT_EQ(internal_reduction.number_of_kpoints(), spglib_reduction.number_of_kpoints());

    /**
     * Both reductions must conserve the total weight (2.0, the spin degeneracy
     * factor KPointsGrid::generate() itself bakes into an unreduced grid).
     */
    auto total_weight = [](const KPointsGrid& grid) {
        Real sum = 0.0;
        for (int kpoint_index = 0; kpoint_index < grid.number_of_kpoints(); kpoint_index++) {
            sum += grid[kpoint_index].weight;
        }
        return sum;
    };
    EXPECT_NEAR(total_weight(internal_reduction), 2.0, 1e-9);
    EXPECT_NEAR(total_weight(spglib_reduction), 2.0, 1e-9);
}

/**
 * Two iron atoms at the corner and body centre of a simple-cubic cell: as one
 * class they are a BCC lattice in a doubled cell (the centring translation
 * doubles the 48 point operations); split into two classes they are the CsCl
 * arrangement of an antiferromagnet, which loses the translation and every
 * operation that swaps the two sites.
 */
TEST(SymmetryTest, SiteSymmetryClassesRemoveOperationsThatSwapInequivalentSites) {
    Crystal iron;
    iron.set_from_lattice_type(1, 5.42);
    iron.add_species({"Fe", 55.845, 8.0, "Fe.UPF"});
    iron.add_atom({0, {0.0, 0.0, 0.0}});
    iron.add_atom({0, {0.5, 0.5, 0.5}});
    iron.compute_derived_quantities();

    EXPECT_EQ(SymmetryAnalyzer::number_of_symmetry_operations(iron), 96);

    iron.set_site_symmetry_classes({0, 1});
    EXPECT_EQ(SymmetryAnalyzer::number_of_symmetry_operations(iron), 48);
    for (const auto& operation : SymmetryAnalyzer::get_symmetry_operations(iron)) {
        EXPECT_EQ(operation.atom_map[0], 0);
        EXPECT_EQ(operation.atom_map[1], 1);
    }
}
