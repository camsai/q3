#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"

using namespace q3;

// ── Gamma-only: 1×1×1 mesh with no offset ──────────────────────────
TEST(KPoints, GammaOnly) {
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    EXPECT_EQ(kpoint_grid.number_of_kpoints(), 1);
    EXPECT_NEAR(kpoint_grid[0].coordinates.x, 0.0, 1e-12);
    EXPECT_NEAR(kpoint_grid[0].coordinates.y, 0.0, 1e-12);
    EXPECT_NEAR(kpoint_grid[0].coordinates.z, 0.0, 1e-12);
    EXPECT_NEAR(kpoint_grid[0].weight, 2.0, 1e-12); // spin degeneracy
}

// ── 3×3×3 mesh with offset produces expected k-point count ──────────
TEST(KPoints, Mesh_3x3x3_Count) {
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(3, 3, 3, 1, 1, 1);

    // Full 3×3×3 = 27 k-points (no symmetry reduction)
    EXPECT_EQ(kpoint_grid.number_of_kpoints(), 27);
}

TEST(KPoints, ExplicitListRetainsMissingGeneratedGridMetadata) {
    KPointsGrid kpoint_grid;
    KPoint kpoint;
    kpoint.coordinates = {0.125, 0.25, 0.375};
    kpoint.weight = 2.0;
    kpoint_grid.add_kpoint(kpoint);

    const std::array<int, 3> missing_metadata = {0, 0, 0};
    EXPECT_EQ(kpoint_grid.grid_dimensions(), missing_metadata);
    EXPECT_EQ(kpoint_grid.grid_shifts(), missing_metadata);
}

// ── Weights sum to 2.0 (spin degeneracy) ────────────────────────────
TEST(KPoints, WeightsSumTo2) {
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(3, 3, 3, 1, 1, 1);

    Real total_weight = 0.0;
    for (int kpoint_index = 0; kpoint_index < kpoint_grid.number_of_kpoints(); kpoint_index++) {
        total_weight += kpoint_grid[kpoint_index].weight;
    }
    EXPECT_NEAR(total_weight, 2.0, 1e-10);
}

/**
 * ── scale_weights(0.5): spin-polarized (LSDA) halving ──────────────
 * Generated/reduced grids embed spin degeneracy (weights sum to 2.0 —
 * see WeightsSumTo2 above). Spin-polarized calculations enumerate both
 * spin channels as explicit k-point states, so callers must halve the
 * weights once via scale_weights(0.5) (build_global_kpoint_grid_from_input
 * does this for number_of_spin_components == 2). A caller that hand-builds
 * a KPointsGrid and skips this step silently double-counts every
 * k-point-summed quantity (band/Hartree/XC energy) in a spin-polarized run.
 */
TEST(KPoints, ScaleWeightsHalvesForSpinPolarized) {
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(3, 3, 3, 1, 1, 1);

    std::vector<Real> weights_before;
    for (int kpoint_index = 0; kpoint_index < kpoint_grid.number_of_kpoints(); kpoint_index++)
        weights_before.push_back(kpoint_grid[kpoint_index].weight);

    kpoint_grid.scale_weights(0.5);

    Real total_weight = 0.0;
    for (int kpoint_index = 0; kpoint_index < kpoint_grid.number_of_kpoints(); kpoint_index++) {
        EXPECT_NEAR(kpoint_grid[kpoint_index].weight, 0.5 * weights_before[kpoint_index], 1e-12)
            << "k-point " << kpoint_index << " not scaled correctly";
        total_weight += kpoint_grid[kpoint_index].weight;
    }
    EXPECT_NEAR(total_weight, 1.0, 1e-10);
}

// ── 2×2×2 mesh with no offset: all k-points in [-0.5, 0.5) ─────────
TEST(KPoints, Mesh_2x2x2_InBZ) {
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 0, 0, 0);

    for (int kpoint_index = 0; kpoint_index < kpoint_grid.number_of_kpoints(); kpoint_index++) {
        EXPECT_GE(kpoint_grid[kpoint_index].coordinates.x, -0.5 - 1e-10)
            << "k-point " << kpoint_index << " out of BZ";
        EXPECT_LT(kpoint_grid[kpoint_index].coordinates.x, 0.5 + 1e-10)
            << "k-point " << kpoint_index << " out of BZ";
        EXPECT_GE(kpoint_grid[kpoint_index].coordinates.y, -0.5 - 1e-10);
        EXPECT_LT(kpoint_grid[kpoint_index].coordinates.y, 0.5 + 1e-10);
        EXPECT_GE(kpoint_grid[kpoint_index].coordinates.z, -0.5 - 1e-10);
        EXPECT_LT(kpoint_grid[kpoint_index].coordinates.z, 0.5 + 1e-10);
    }
}

// ── Equal weights for uniform mesh ──────────────────────────────────
TEST(KPoints, UniformWeights) {
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(4, 4, 4, 0, 0, 0);

    Real expected_w = 2.0 / kpoint_grid.number_of_kpoints();
    for (int kpoint_index = 0; kpoint_index < kpoint_grid.number_of_kpoints(); kpoint_index++) {
        EXPECT_NEAR(kpoint_grid[kpoint_index].weight, expected_w, 1e-12)
            << "k-point " << kpoint_index << " has unexpected weight";
    }
}
