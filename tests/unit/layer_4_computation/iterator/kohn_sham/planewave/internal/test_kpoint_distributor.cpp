/**
 * Unit tests for KPointDistributor (serial baseline).
 *
 * No MPI — these run under the standard ctest / q3_tests executable.
 */

#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/internal/kpoint_distributor.hpp"
#include <array>
#include <gtest/gtest.h>
#include <numeric>

namespace q3 {
namespace parallel {
namespace {

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Build a simple flat k-point grid with number_of_kpoints equally-weighted points.
 */
static KPointsGrid make_uniform_grid(int number_of_kpoints) {
    KPointsGrid grid;
    const Real weight = 2.0 / number_of_kpoints;
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
        KPoint kpoint;
        kpoint.coordinates.x = static_cast<Real>(kpoint_index) / number_of_kpoints;
        kpoint.coordinates.y = 0.0;
        kpoint.coordinates.z = 0.0;
        kpoint.weight = weight;
        grid.add_kpoint(kpoint);
    }
    return grid;
}

// ── Invalid arguments ─────────────────────────────────────────────────────────

TEST(KPointDistributorTest, ZeroPoolsThrows) {
    KPointsGrid grid = make_uniform_grid(4);
    EXPECT_THROW(KPointDistributor(grid, 0, 0), std::invalid_argument);
}

TEST(KPointDistributorTest, NegativePoolIndexThrows) {
    KPointsGrid grid = make_uniform_grid(4);
    EXPECT_THROW(KPointDistributor(grid, -1, 2), std::invalid_argument);
}

TEST(KPointDistributorTest, PoolIndexOutOfRangeThrows) {
    KPointsGrid grid = make_uniform_grid(4);
    EXPECT_THROW(KPointDistributor(grid, 2, 2), std::invalid_argument);
}

// ── Single pool (identity) ────────────────────────────────────────────────────

TEST(KPointDistributorTest, SinglePoolOwnsAllKpoints) {
    KPointsGrid grid = make_uniform_grid(6);
    KPointDistributor distributor(grid, 0, 1);
    EXPECT_EQ(distributor.number_of_local_kpoints(), 6);
    for (int local_index = 0; local_index < 6; ++local_index) {
        EXPECT_EQ(distributor.local_to_global_kpoint_map()[local_index], local_index);
    }
}

// ── Two pools, 6 k-points ─────────────────────────────────────────────────────

TEST(KPointDistributorTest, TwoPoolsSixKpointsPool0GetsEvenIndices) {
    KPointsGrid grid = make_uniform_grid(6);
    KPointDistributor pool0(grid, 0, 2);
    EXPECT_EQ(pool0.number_of_local_kpoints(), 3);
    EXPECT_EQ(pool0.local_to_global_kpoint_map()[0], 0);
    EXPECT_EQ(pool0.local_to_global_kpoint_map()[1], 2);
    EXPECT_EQ(pool0.local_to_global_kpoint_map()[2], 4);
}

TEST(KPointDistributorTest, TwoPoolsSixKpointsPool1GetsOddIndices) {
    KPointsGrid grid = make_uniform_grid(6);
    KPointDistributor pool1(grid, 1, 2);
    EXPECT_EQ(pool1.number_of_local_kpoints(), 3);
    EXPECT_EQ(pool1.local_to_global_kpoint_map()[0], 1);
    EXPECT_EQ(pool1.local_to_global_kpoint_map()[1], 3);
    EXPECT_EQ(pool1.local_to_global_kpoint_map()[2], 5);
}

// ── Four pools, 6 k-points (uneven split) ────────────────────────────────────

TEST(KPointDistributorTest, FourPoolsSixKpointsFirstTwoPoolsGetTwoEach) {
    KPointsGrid grid = make_uniform_grid(6);
    KPointDistributor pool0(grid, 0, 4);
    KPointDistributor pool1(grid, 1, 4);
    KPointDistributor pool2(grid, 2, 4);
    KPointDistributor pool3(grid, 3, 4);
    EXPECT_EQ(pool0.number_of_local_kpoints(), 2); // {0, 4}
    EXPECT_EQ(pool1.number_of_local_kpoints(), 2); // {1, 5}
    EXPECT_EQ(pool2.number_of_local_kpoints(), 1); // {2}
    EXPECT_EQ(pool3.number_of_local_kpoints(), 1); // {3}
}

// ── Weight preservation ───────────────────────────────────────────────────────

TEST(KPointDistributorTest, WeightsSumToGlobalWeightAcrossAllPools) {
    const int number_of_kpoints = 6;
    const int number_of_pools = 3;
    KPointsGrid grid = make_uniform_grid(number_of_kpoints);

    Real global_weight_sum = 0.0;
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
        global_weight_sum += grid[kpoint_index].weight;
    }

    Real pooled_weight_sum = 0.0;
    for (int pool_index = 0; pool_index < number_of_pools; ++pool_index) {
        KPointDistributor distributor(grid, pool_index, number_of_pools);
        for (int local_index = 0; local_index < distributor.number_of_local_kpoints();
             ++local_index) {
            pooled_weight_sum += distributor.local_kpoint_grid()[local_index].weight;
        }
    }

    EXPECT_NEAR(pooled_weight_sum, global_weight_sum, 1e-12);
}

TEST(KPointDistributorTest, GeneratedGridMetadataSurvivesDistribution) {
    KPointsGrid grid;
    grid.generate(4, 2, 3, 1, 0, 1);
    KPointDistributor distributor(grid, 1, 3);

    const std::array<int, 3> expected_dimensions = {4, 2, 3};
    const std::array<int, 3> expected_shifts = {1, 0, 1};
    EXPECT_EQ(distributor.local_kpoint_grid().grid_dimensions(), expected_dimensions);
    EXPECT_EQ(distributor.local_kpoint_grid().grid_shifts(), expected_shifts);
}

// ── Local grid coordinates match global grid ──────────────────────────────────

TEST(KPointDistributorTest, LocalGridCoordinatesMatchGlobalGrid) {
    KPointsGrid grid = make_uniform_grid(8);
    const int number_of_pools = 4;

    for (int pool_index = 0; pool_index < number_of_pools; ++pool_index) {
        KPointDistributor distributor(grid, pool_index, number_of_pools);
        for (int local_index = 0; local_index < distributor.number_of_local_kpoints();
             ++local_index) {
            const int global_index = distributor.local_to_global_kpoint_map()[local_index];
            EXPECT_NEAR(distributor.local_kpoint_grid()[local_index].coordinates.x,
                        grid[global_index].coordinates.x,
                        1e-14);
        }
    }
}

// ── Coverage: all k-points are assigned exactly once ─────────────────────────

TEST(KPointDistributorTest, AllKpointsAssignedExactlyOnce) {
    const int number_of_kpoints = 7;
    const int number_of_pools = 3;
    KPointsGrid grid = make_uniform_grid(number_of_kpoints);

    std::vector<int> assignment_count(number_of_kpoints, 0);
    for (int pool_index = 0; pool_index < number_of_pools; ++pool_index) {
        KPointDistributor distributor(grid, pool_index, number_of_pools);
        for (int global_index : distributor.local_to_global_kpoint_map()) {
            assignment_count[global_index]++;
        }
    }
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
        EXPECT_EQ(assignment_count[kpoint_index], 1) << "k-point " << kpoint_index << " assigned "
                                                     << assignment_count[kpoint_index] << " times";
    }
}

} // namespace
} // namespace parallel
} // namespace q3
