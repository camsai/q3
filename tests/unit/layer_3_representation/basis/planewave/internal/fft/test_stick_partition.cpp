#include "layer_3_representation/basis/planewave/internal/fft/stick_partition.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <set>

using namespace q3;

namespace {

/**
 * Stick weights for a sphere of the given radius inscribed in the grid: the
 * number of third-dimension points inside the cutoff for each first/second
 * pair. This is the shape the real cutoff produces -- heavy near the origin,
 * empty at the corners -- without needing a lattice.
 */
std::vector<int>
spherical_weights(int dimension_1, int dimension_2, int dimension_3, double radius) {
    std::vector<int> weights(static_cast<std::size_t>(dimension_1) * dimension_2, 0);
    auto signed_index = [](int wrapped, int dimension) {
        return wrapped <= dimension / 2 ? wrapped : wrapped - dimension;
    };
    for (int first = 0; first < dimension_1; ++first) {
        for (int second = 0; second < dimension_2; ++second) {
            double h = signed_index(first, dimension_1);
            double k = signed_index(second, dimension_2);
            int count = 0;
            for (int third = 0; third < dimension_3; ++third) {
                double l = signed_index(third, dimension_3);
                if (h * h + k * k + l * l <= radius * radius) {
                    ++count;
                }
            }
            weights[static_cast<std::size_t>(first) * dimension_2 + second] = count;
        }
    }
    return weights;
}

} // namespace

// ─── Mirror pairing ─────────────────────────────────────────────────────────

/**
 * The property the whole decomposition exists for: wherever a rank holds the
 * stick at (h, k) it also holds the stick at (-h, -k), so the gamma-only
 * reconstruction never has to leave the rank.
 */
TEST(FFTStickPartition, MirrorSticksShareAnOwner) {
    const int d1 = 27, d2 = 25, d3 = 24;
    auto weights = spherical_weights(d1, d2, d3, 9.0);
    for (int ranks : {1, 2, 3, 4, 8}) {
        FFTStickPartition partition;
        partition.build(d1, d2, weights, ranks);
        for (int first = 0; first < d1; ++first) {
            for (int second = 0; second < d2; ++second) {
                int mirror_first = (d1 - first) % d1;
                int mirror_second = (d2 - second) % d2;
                EXPECT_EQ(partition.owner_of(first, second),
                          partition.owner_of(mirror_first, mirror_second))
                    << "ranks=" << ranks << " stick=(" << first << "," << second << ")";
            }
        }
    }
}

/**
 * Even grids have self-mirrored sticks off the origin -- the Nyquist rows --
 * which the pairing must treat as single rather than pairing with themselves
 * twice and double-counting their weight.
 */
TEST(FFTStickPartition, EvenGridsHandleSelfMirroredSticks) {
    const int d1 = 16, d2 = 16, d3 = 16;
    auto weights = spherical_weights(d1, d2, d3, 6.0);
    FFTStickPartition partition;
    partition.build(d1, d2, weights, 4);

    long long assigned = 0;
    for (const auto& rank_weight : partition.weight_per_rank()) {
        assigned += rank_weight;
    }
    long long expected = 0;
    for (int weight : weights) {
        expected += weight;
    }
    EXPECT_EQ(assigned, expected);
}

// ─── Coverage ───────────────────────────────────────────────────────────────

TEST(FFTStickPartition, EveryOccupiedStickHasAnOwnerAndEmptyOnesDoNot) {
    const int d1 = 20, d2 = 18, d3 = 16;
    auto weights = spherical_weights(d1, d2, d3, 5.0);
    FFTStickPartition partition;
    partition.build(d1, d2, weights, 3);
    for (std::size_t index = 0; index < weights.size(); ++index) {
        if (weights[index] > 0) {
            EXPECT_GE(partition.owner()[index], 0) << "occupied stick " << index;
            EXPECT_LT(partition.owner()[index], 3);
        } else {
            EXPECT_EQ(partition.owner()[index], FFTStickPartition::unassigned);
        }
    }
}

// ─── Balance ────────────────────────────────────────────────────────────────

/**
 * The point of dealing sticks rather than planes. The slab split on the
 * 54-atom silicon cell was measured uneven by 3.8x across four ranks; this
 * gate is set at 1.2x, and pairing is what makes it achievable at all since
 * pairs are the indivisible unit.
 */
TEST(FFTStickPartition, SplitIsBalancedAcrossRanks) {
    const int d1 = 72, d2 = 72, d3 = 72;
    auto weights = spherical_weights(d1, d2, d3, 24.0);
    for (int ranks : {2, 4, 8, 16}) {
        FFTStickPartition partition;
        partition.build(d1, d2, weights, ranks);
        EXPECT_LT(partition.imbalance(), 1.2) << "ranks=" << ranks;
    }
}

TEST(FFTStickPartition, SingleRankOwnsEverythingOccupied) {
    const int d1 = 12, d2 = 12, d3 = 12;
    auto weights = spherical_weights(d1, d2, d3, 4.0);
    FFTStickPartition partition;
    partition.build(d1, d2, weights, 1);
    EXPECT_DOUBLE_EQ(partition.imbalance(), 1.0);
    for (std::size_t index = 0; index < weights.size(); ++index) {
        EXPECT_EQ(partition.owner()[index], weights[index] > 0 ? 0 : FFTStickPartition::unassigned);
    }
}

// ─── Agreement across ranks ─────────────────────────────────────────────────

/**
 * Every rank builds this independently and they must agree exactly, so the
 * deal has to be a pure function of its inputs -- no iteration order over an
 * unordered container, no ties broken by address.
 */
TEST(FFTStickPartition, AssignmentIsDeterministic) {
    const int d1 = 30, d2 = 27, d3 = 24;
    auto weights = spherical_weights(d1, d2, d3, 10.0);
    FFTStickPartition first_build;
    FFTStickPartition second_build;
    first_build.build(d1, d2, weights, 6);
    second_build.build(d1, d2, weights, 6);
    EXPECT_EQ(first_build.owner(), second_build.owner());
    EXPECT_EQ(first_build.weight_per_rank(), second_build.weight_per_rank());
}

// ─── Miller lookup ──────────────────────────────────────────────────────────

TEST(FFTStickPartition, NegativeMillerIndicesWrapOntoTheirStick) {
    const int d1 = 15, d2 = 15, d3 = 12;
    auto weights = spherical_weights(d1, d2, d3, 5.0);
    FFTStickPartition partition;
    partition.build(d1, d2, weights, 4);
    EXPECT_EQ(partition.owner_of_miller(-1, -2), partition.owner_of(d1 - 1, d2 - 2));
    EXPECT_EQ(partition.owner_of_miller(1, 2), partition.owner_of_miller(-1, -2));
}
