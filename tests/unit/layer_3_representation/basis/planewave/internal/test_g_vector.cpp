#include <gtest/gtest.h>

#include "layer_3_representation/basis/planewave/internal/g_vector.hpp"

using namespace q3;

// ── Default construction gives zeroed fields ──
TEST(GVector, DefaultConstructionProducesZeroValues) {
    GVector g_vector;

    EXPECT_EQ(g_vector.miller.x, 0);
    EXPECT_EQ(g_vector.miller.y, 0);
    EXPECT_EQ(g_vector.miller.z, 0);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.x, 0.0);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.y, 0.0);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.z, 0.0);
    EXPECT_DOUBLE_EQ(g_vector.magnitude_squared, 0.0);
}

// ── Fields can be set and read back ──
TEST(GVector, FieldAssignmentAndReadback) {
    GVector g_vector;
    g_vector.miller = {1, -2, 3};
    g_vector.g_cart = {0.5, -1.0, 1.5};
    g_vector.magnitude_squared = 3.5;

    EXPECT_EQ(g_vector.miller.x, 1);
    EXPECT_EQ(g_vector.miller.y, -2);
    EXPECT_EQ(g_vector.miller.z, 3);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.x, 0.5);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.y, -1.0);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.z, 1.5);
    EXPECT_DOUBLE_EQ(g_vector.magnitude_squared, 3.5);
}

// ── GVector inherits from ReciprocalSpaceVector ──
TEST(GVector, InheritsFromReciprocalSpaceVector) {
    GVector g_vector;
    g_vector.coordinates = {1.0, 2.0, 3.0};
    // Access via base class reference
    const ReciprocalSpaceVector& base_reference = g_vector;
    EXPECT_DOUBLE_EQ(base_reference.coordinates.x, 1.0);
    EXPECT_DOUBLE_EQ(base_reference.coordinates.y, 2.0);
    EXPECT_DOUBLE_EQ(base_reference.coordinates.z, 3.0);
}

// ── Magnitude squared is independent from g_cart (manually set) ──
TEST(GVector, MagnitudeSquaredIsIndependentFromCartesianCoordinates) {
    GVector g_vector;
    g_vector.g_cart = {1.0, 1.0, 1.0};
    g_vector.magnitude_squared = 99.0;
    // magnitude_squared is a stored field, not computed from g_cart
    EXPECT_DOUBLE_EQ(g_vector.magnitude_squared, 99.0);
    EXPECT_DOUBLE_EQ(g_vector.g_cart.norm2(), 3.0);
}

// ── Miller indices support negative values ──
TEST(GVector, MillerIndicesSupportNegativeValues) {
    GVector g_vector;
    g_vector.miller = {-5, -10, -15};
    EXPECT_EQ(g_vector.miller[0], -5);
    EXPECT_EQ(g_vector.miller[1], -10);
    EXPECT_EQ(g_vector.miller[2], -15);
}
