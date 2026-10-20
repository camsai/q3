#include <gtest/gtest.h>

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vector.hpp"
#include "layer_3_representation/basis/planewave/internal/miller_index_lookup.hpp"

using namespace q3;

namespace {

GVector make_g_vector(int h, int k, int l) {
    GVector g_vector;
    g_vector.miller = {h, k, l};
    return g_vector;
}

} // namespace

TEST(MillerIndexLookup, FindsEveryBuiltIndexExactly) {
    std::vector<GVector> g_vectors = {make_g_vector(0, 0, 0),
                                      make_g_vector(1, 0, 0),
                                      make_g_vector(-1, 0, 0),
                                      make_g_vector(0, 2, -3),
                                      make_g_vector(-5, 5, -5)};

    MillerIndexLookup lookup;
    lookup.build(g_vectors);

    EXPECT_EQ(lookup.find(0, 0, 0), 0);
    EXPECT_EQ(lookup.find(1, 0, 0), 1);
    EXPECT_EQ(lookup.find(-1, 0, 0), 2);
    EXPECT_EQ(lookup.find(0, 2, -3), 3);
    EXPECT_EQ(lookup.find(-5, 5, -5), 4);
    EXPECT_EQ(lookup.size(), 5);
}

TEST(MillerIndexLookup, MissingTripleReturnsMinusOne) {
    std::vector<GVector> g_vectors = {make_g_vector(0, 0, 0), make_g_vector(1, 1, 1)};

    MillerIndexLookup lookup;
    lookup.build(g_vectors);

    EXPECT_EQ(lookup.find(2, 2, 2), -1);
    EXPECT_EQ(lookup.find(-1, 0, 0), -1);
}

TEST(MillerIndexLookup, TripleOutsideTheBuiltRangeReturnsMinusOneRatherThanCollide) {
    /**
     * Only small indices were built; a much larger triple must not alias
     * onto a real entry through the packed-key arithmetic.
     */
    std::vector<GVector> g_vectors = {make_g_vector(1, 1, 1)};

    MillerIndexLookup lookup;
    lookup.build(g_vectors);

    EXPECT_EQ(lookup.find(1, 1, 1), 0);
    EXPECT_EQ(lookup.find(1000, 1000, 1000), -1);
    EXPECT_EQ(lookup.find(-1000, -1000, -1000), -1);
}

TEST(MillerIndexLookup, HandlesMillerIndicesWiderThanTheOldHardcodedBound) {
    /**
     * The precedent this class generalizes used a fixed offset of 1000,
     * silently colliding above that range. Build with indices past 1000
     * on every axis and confirm every one still resolves uniquely --
     * the offset must be sized from the actual data, not assumed.
     */
    std::vector<GVector> g_vectors = {
        make_g_vector(1500, -1200, 900), make_g_vector(-1500, 1200, -900), make_g_vector(0, 0, 0)};

    MillerIndexLookup lookup;
    lookup.build(g_vectors);

    EXPECT_EQ(lookup.find(1500, -1200, 900), 0);
    EXPECT_EQ(lookup.find(-1500, 1200, -900), 1);
    EXPECT_EQ(lookup.find(0, 0, 0), 2);
}

TEST(MillerIndexLookup, IntegerVector3DOverloadMatchesComponentOverload) {
    std::vector<GVector> g_vectors = {make_g_vector(3, -2, 1)};

    MillerIndexLookup lookup;
    lookup.build(g_vectors);

    IntegerVector3D miller(3, -2, 1);
    EXPECT_EQ(lookup.find(miller), lookup.find(3, -2, 1));
}

TEST(MillerIndexLookup, RebuildingReplacesThePreviousTable) {
    MillerIndexLookup lookup;
    lookup.build({make_g_vector(0, 0, 0), make_g_vector(1, 0, 0)});
    EXPECT_EQ(lookup.size(), 2);

    lookup.build({make_g_vector(5, 5, 5)});
    EXPECT_EQ(lookup.size(), 1);
    EXPECT_EQ(lookup.find(0, 0, 0), -1);
    EXPECT_EQ(lookup.find(5, 5, 5), 0);
}
