#include "layer_1_structure/reciprocal_space/kpoint.hpp"
#include "layer_1_structure/reciprocal_space/reciprocal_space_vector.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vector.hpp"
#include <gtest/gtest.h>

using namespace q3;

// ─── ReciprocalSpaceVector ──────────────────────────────────────────────────

TEST(ReciprocalSpaceVector, DefaultConstructionHasZeroCoordinates) {
    ReciprocalSpaceVector vector;
    EXPECT_DOUBLE_EQ(vector.coordinates.x, 0.0);
    EXPECT_DOUBLE_EQ(vector.coordinates.y, 0.0);
    EXPECT_DOUBLE_EQ(vector.coordinates.z, 0.0);
}

// ─── KPoint ─────────────────────────────────────────────────────────────────

TEST(KPoint, DefaultWeightIsOne) {
    KPoint kpoint;
    EXPECT_DOUBLE_EQ(kpoint.weight, 1.0);
}

TEST(KPoint, DefaultCoordinatesAreZero) {
    KPoint kpoint;
    EXPECT_DOUBLE_EQ(kpoint.coordinates.x, 0.0);
    EXPECT_DOUBLE_EQ(kpoint.coordinates.y, 0.0);
    EXPECT_DOUBLE_EQ(kpoint.coordinates.z, 0.0);
}

TEST(KPoint, CoordinatesCanBeSet) {
    KPoint kpoint;
    kpoint.coordinates = {0.5, 0.5, 0.5};
    EXPECT_DOUBLE_EQ(kpoint.coordinates.x, 0.5);
    EXPECT_DOUBLE_EQ(kpoint.coordinates.y, 0.5);
    EXPECT_DOUBLE_EQ(kpoint.coordinates.z, 0.5);
}

TEST(KPoint, WeightCanBeSet) {
    KPoint kpoint;
    kpoint.weight = 0.25;
    EXPECT_DOUBLE_EQ(kpoint.weight, 0.25);
}

// ─── GVector ────────────────────────────────────────────────────────────────

TEST(GVector, DefaultConstructionHasZeroMagnitudeSquared) {
    GVector g_vector;
    EXPECT_DOUBLE_EQ(g_vector.magnitude_squared, 0.0);
}

TEST(GVector, DefaultMillerIndicesAreZero) {
    GVector g_vector;
    EXPECT_EQ(g_vector.miller.x, 0);
    EXPECT_EQ(g_vector.miller.y, 0);
    EXPECT_EQ(g_vector.miller.z, 0);
}

TEST(GVector, MillerIndicesCanBeSetAndRead) {
    GVector g_vector;
    g_vector.miller = {1, -2, 3};
    EXPECT_EQ(g_vector.miller.x, 1);
    EXPECT_EQ(g_vector.miller.y, -2);
    EXPECT_EQ(g_vector.miller.z, 3);
}

TEST(GVector, MagnitudeSquaredCanBeSet) {
    GVector g_vector;
    g_vector.magnitude_squared = 12.5;
    EXPECT_DOUBLE_EQ(g_vector.magnitude_squared, 12.5);
}

TEST(GVector, CartesianCoordinatesDefaultToZero) {
    GVector g_vector;
    EXPECT_DOUBLE_EQ(g_vector.g_cart.x, 0.0);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.y, 0.0);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.z, 0.0);
}

TEST(GVector, InheritsReciprocalSpaceVectorCoordinates) {
    GVector g_vector;
    g_vector.coordinates = {1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(g_vector.coordinates.x, 1.0);
    EXPECT_DOUBLE_EQ(g_vector.coordinates.y, 2.0);
    EXPECT_DOUBLE_EQ(g_vector.coordinates.z, 3.0);
}
