#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"

using namespace q3;

namespace {

/**
 * Reconstructs sum_b w_b * b (x) b from a computed shell set and compares
 * it against the identity matrix -- the completeness condition every
 * shell/weight solution must satisfy, independent of how it was found.
 */
void expect_completeness_condition_satisfied(const KPointNeighborShells& shells, Real tolerance) {
    Matrix3x3 reconstructed;
    for (const KPointNeighborDirection& direction : shells.neighbor_directions()) {
        const Vector3D& b = direction.reciprocal_lattice_translation;
        const Real weight = direction.weight;
        reconstructed[0].x += weight * b.x * b.x;
        reconstructed[0].y += weight * b.x * b.y;
        reconstructed[0].z += weight * b.x * b.z;
        reconstructed[1].x += weight * b.y * b.x;
        reconstructed[1].y += weight * b.y * b.y;
        reconstructed[1].z += weight * b.y * b.z;
        reconstructed[2].x += weight * b.z * b.x;
        reconstructed[2].y += weight * b.z * b.y;
        reconstructed[2].z += weight * b.z * b.z;
    }

    EXPECT_NEAR(reconstructed[0].x, 1.0, tolerance);
    EXPECT_NEAR(reconstructed[1].y, 1.0, tolerance);
    EXPECT_NEAR(reconstructed[2].z, 1.0, tolerance);
    EXPECT_NEAR(reconstructed[0].y, 0.0, tolerance);
    EXPECT_NEAR(reconstructed[0].z, 0.0, tolerance);
    EXPECT_NEAR(reconstructed[1].z, 0.0, tolerance);
}

} // namespace

// ── Simple cubic: hand-derivable exact weight ──────────────────────

TEST(KPointNeighborShells, SimpleCubicSingleShellWithHandDerivedWeight) {
    /**
     * Simple cubic reciprocal lattice (b1,b2,b3 = orthonormal), grid 4x4x4.
     * The 6 nearest neighbors are at b = +-(1/4,0,0), +-(0,1/4,0),
     * +-(0,0,1/4), each with |b|^2 = (1/4)^2. Their shell tensor is
     * diag(2/16, 2/16, 2/16), so completeness (w * tensor = I) gives
     * w = 16/2 = 8 = n^2/2 for grid dimension n = 4.
     */
    Matrix3x3 reciprocal_lattice_vectors({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0});

    KPointNeighborShells shells;
    shells.compute(reciprocal_lattice_vectors, 4, 4, 4);

    EXPECT_EQ(shells.number_of_shells(), 1);
    ASSERT_EQ(shells.neighbor_directions().size(), 6u);

    for (const KPointNeighborDirection& direction : shells.neighbor_directions()) {
        EXPECT_NEAR(direction.weight, 8.0, 1e-10);
        EXPECT_NEAR(direction.reciprocal_lattice_translation.norm2(), 0.0625, 1e-12);
    }

    expect_completeness_condition_satisfied(shells, 1e-8);
}

TEST(KPointNeighborShells, SimpleCubicWeightScalesWithGridDimensionSquared) {
    Matrix3x3 reciprocal_lattice_vectors({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0});

    KPointNeighborShells shells;
    shells.compute(reciprocal_lattice_vectors, 6, 6, 6);

    EXPECT_EQ(shells.number_of_shells(), 1);
    for (const KPointNeighborDirection& direction : shells.neighbor_directions()) {
        EXPECT_NEAR(direction.weight, 18.0, 1e-9); // 6^2 / 2
    }
}

TEST(KPointNeighborShells, EveryDirectionHasAnOppositePartnerWithEqualWeight) {
    Matrix3x3 reciprocal_lattice_vectors({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0});

    KPointNeighborShells shells;
    shells.compute(reciprocal_lattice_vectors, 4, 4, 4);

    for (const KPointNeighborDirection& direction : shells.neighbor_directions()) {
        bool found_opposite = false;
        for (const KPointNeighborDirection& other : shells.neighbor_directions()) {
            if (other.grid_step.x == -direction.grid_step.x &&
                other.grid_step.y == -direction.grid_step.y &&
                other.grid_step.z == -direction.grid_step.z) {
                found_opposite = true;
                EXPECT_NEAR(other.weight, direction.weight, 1e-12);
            }
        }
        EXPECT_TRUE(found_opposite);
    }
}

// ── Hexagonal (graphite-like) and triclinic: self-consistency ─────

TEST(KPointNeighborShells, HexagonalTallCellSatisfiesCompleteness) {
    /**
     * a1=(1,0,0), a2=(-1/2,sqrt3/2,0), a3=(0,0,3) -- graphite-shaped
     * (in-plane hexagonal, tall out-of-plane axis), grid 9x9x3.
     */
    Matrix3x3 direct_lattice_vectors(
        {1.0, 0.0, 0.0}, {-0.5, std::sqrt(3.0) / 2.0, 0.0}, {0.0, 0.0, 3.0});
    Matrix3x3 reciprocal_lattice_vectors = direct_lattice_vectors.inverse().transpose();

    KPointNeighborShells shells;
    shells.compute(reciprocal_lattice_vectors, 9, 9, 3);

    EXPECT_GT(shells.number_of_shells(), 1);
    expect_completeness_condition_satisfied(shells, 1e-8);
}

TEST(KPointNeighborShells, TriclinicCellSatisfiesCompletenessWithMultipleShells) {
    // A deliberately low-symmetry, fully skewed lattice.
    Matrix3x3 direct_lattice_vectors({1.0, 0.0, 0.0}, {0.3, 1.0, 0.0}, {0.2, 0.15, 1.1});
    Matrix3x3 reciprocal_lattice_vectors = direct_lattice_vectors.inverse().transpose();

    KPointNeighborShells shells;
    shells.compute(reciprocal_lattice_vectors, 4, 4, 4);

    /**
     * A generic skewed lattice's single nearest-neighbor shell is not
     * proportional to the identity, so completeness genuinely needs more
     * than one shell -- the plan's own worked example calls for >= 4.
     */
    EXPECT_GE(shells.number_of_shells(), 4);
    expect_completeness_condition_satisfied(shells, 1e-8);
}

// ── Search-range exhaustion is a typed error, not a silent miss ───

TEST(KPointNeighborShells, InsufficientSearchRangeThrows) {
    Matrix3x3 reciprocal_lattice_vectors({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0});
    KPointNeighborShells shells;

    EXPECT_THROW(
        shells.compute(
            reciprocal_lattice_vectors, 4, 4, 4, /*search_range=*/0, /*maximum_shell_count=*/12),
        std::runtime_error);
}

TEST(KPointNeighborShells, ShellCountCapThrowsBeforeCompletenessOnATriclinicCell) {
    Matrix3x3 direct_lattice_vectors({1.0, 0.0, 0.0}, {0.3, 1.0, 0.0}, {0.2, 0.15, 1.1});
    Matrix3x3 reciprocal_lattice_vectors = direct_lattice_vectors.inverse().transpose();
    KPointNeighborShells shells;

    EXPECT_THROW(
        shells.compute(
            reciprocal_lattice_vectors, 4, 4, 4, /*search_range=*/3, /*maximum_shell_count=*/1),
        std::runtime_error);
}

// ── resolve_neighbor: integer grid-index arithmetic + BZ wrap ─────

TEST(KPointNeighborShellsResolveNeighbor, InteriorKPointNeedsNoPeriodicImageShift) {
    KPointsGrid grid;
    grid.generate(4, 4, 4, 0, 0, 0);
    const std::array<int, 3> grid_dimensions = {4, 4, 4};

    /**
     * Unshifted n=4 grid, x-axis raw (unfolded) coordinates are
     * 0, 1/4, 2/4, 3/4; KPointsGrid folds anything >= 1/2 down by 1, so
     * folded coordinates for index 0,1,2,3 are 0, 1/4, -1/2, -1/4. Index
     * (0,1,1) -> (1,1,1) stays entirely below the 1/2 fold threshold (raw
     * 0 -> 1/4), so no periodic image shift is needed.
     */
    const int starting_index = 0 * 4 * 4 + 1 * 4 + 1;
    IntegerVector3D grid_step(1, 0, 0);

    ResolvedKPointNeighbor resolved = KPointNeighborShells::resolve_neighbor(
        starting_index, grid_step, grid_dimensions, grid.kpoints());

    EXPECT_EQ(resolved.neighbor_kpoint_index, 1 * 4 * 4 + 1 * 4 + 1);
    EXPECT_EQ(resolved.periodic_image_shift.x, 0);
    EXPECT_EQ(resolved.periodic_image_shift.y, 0);
    EXPECT_EQ(resolved.periodic_image_shift.z, 0);
}

TEST(KPointNeighborShellsResolveNeighbor, FoldBoundaryCrossingNeedsNonZeroPeriodicImageShift) {
    KPointsGrid grid;
    grid.generate(4, 4, 4, 0, 0, 0);
    const std::array<int, 3> grid_dimensions = {4, 4, 4};

    /**
     * Stepping from index x=1 (raw/folded coordinate 1/4) to x=2 crosses
     * the raw coordinate 1/2, which KPointsGrid folds down to -1/2 --
     * that discontinuity, not the array-index wraparound at x=3->0, is
     * what forces a nonzero periodic image shift here.
     */
    const int starting_index = 1 * 4 * 4 + 0 * 4 + 0;
    IntegerVector3D grid_step(1, 0, 0);

    ResolvedKPointNeighbor resolved = KPointNeighborShells::resolve_neighbor(
        starting_index, grid_step, grid_dimensions, grid.kpoints());

    EXPECT_EQ(resolved.neighbor_kpoint_index, 2 * 4 * 4 + 0 * 4 + 0);
    EXPECT_EQ(resolved.periodic_image_shift.x, 1);
}

TEST(KPointNeighborShellsResolveNeighbor,
     ArrayIndexWraparoundNeedsNoShiftWhenCoordinatesAlreadyAlign) {
    KPointsGrid grid;
    grid.generate(4, 4, 4, 0, 0, 0);
    const std::array<int, 3> grid_dimensions = {4, 4, 4};

    /**
     * Index x=3 (folded coordinate -1/4) stepping to the array-wraparound
     * neighbor x=0 (folded coordinate 0): -1/4 + 1/4 = 0 exactly, so this
     * wraparound needs no shift -- the array-index and coordinate-fold
     * boundaries do not coincide for an unshifted even-n grid.
     */
    const int starting_index = 3 * 4 * 4 + 0 * 4 + 0;
    IntegerVector3D grid_step(1, 0, 0);

    ResolvedKPointNeighbor resolved = KPointNeighborShells::resolve_neighbor(
        starting_index, grid_step, grid_dimensions, grid.kpoints());

    EXPECT_EQ(resolved.neighbor_kpoint_index, 0);
    EXPECT_EQ(resolved.periodic_image_shift.x, 0);
}

TEST(KPointNeighborShellsResolveNeighbor, ResolvedShiftReconstructsExactTargetCoordinate) {
    /**
     * The defining physical invariant: starting_coordinate + step/n must
     * equal neighbor_coordinate + periodic_image_shift, component-wise,
     * to floating precision -- for every grid index and every shell
     * direction on a real grid, not just the two hand-picked cases above.
     */
    KPointsGrid grid;
    grid.generate(4, 4, 4, 1, 1, 1);
    const std::array<int, 3> grid_dimensions = {4, 4, 4};

    Matrix3x3 reciprocal_lattice_vectors({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0});
    KPointNeighborShells shells;
    shells.compute(reciprocal_lattice_vectors, 4, 4, 4);

    for (int starting_index = 0; starting_index < grid.number_of_kpoints(); starting_index++) {
        for (const KPointNeighborDirection& direction : shells.neighbor_directions()) {
            ResolvedKPointNeighbor resolved = KPointNeighborShells::resolve_neighbor(
                starting_index, direction.grid_step, grid_dimensions, grid.kpoints());

            const Vector3D& starting_coordinates = grid[starting_index].coordinates;
            const Vector3D& neighbor_coordinates = grid[resolved.neighbor_kpoint_index].coordinates;

            for (int component = 0; component < 3; component++) {
                const Real target = starting_coordinates[component] +
                                    static_cast<Real>(direction.grid_step[component]) / 4.0;
                const Real reconstructed =
                    neighbor_coordinates[component] +
                    static_cast<Real>(resolved.periodic_image_shift[component]);
                EXPECT_NEAR(reconstructed, target, 1e-12);
            }
        }
    }
}

TEST(KPointNeighborShellsResolveNeighbor, NonGridKPointListThrows) {
    /**
     * A single k-point cannot be a full 4x4x4 grid: resolve_neighbor must
     * refuse rather than silently return a wrong shift.
     */
    std::vector<KPoint> not_a_full_grid(64);
    for (KPoint& kpoint : not_a_full_grid) {
        kpoint.coordinates = {0.0, 0.0, 0.0};
        kpoint.weight = 2.0 / 64.0;
    }
    const std::array<int, 3> grid_dimensions = {4, 4, 4};

    EXPECT_THROW(KPointNeighborShells::resolve_neighbor(
                     0, IntegerVector3D(1, 0, 0), grid_dimensions, not_a_full_grid),
                 std::runtime_error);
}
