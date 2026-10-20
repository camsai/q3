#include <gtest/gtest.h>

#include "layer_0_core/types/integer_vector_3d.hpp"

using namespace q3;

TEST(IntegerVector3D, DefaultConstruction) {
    IntegerVector3D vector;
    EXPECT_EQ(vector.x, 0);
    EXPECT_EQ(vector.y, 0);
    EXPECT_EQ(vector.z, 0);
}

TEST(IntegerVector3D, ParameterizedConstruction) {
    IntegerVector3D vector(1, -2, 3);
    EXPECT_EQ(vector.x, 1);
    EXPECT_EQ(vector.y, -2);
    EXPECT_EQ(vector.z, 3);
}

TEST(IntegerVector3D, ConstIndexAccess) {
    const IntegerVector3D vector(10, 20, 30);
    EXPECT_EQ(vector[0], 10);
    EXPECT_EQ(vector[1], 20);
    EXPECT_EQ(vector[2], 30);
}

TEST(IntegerVector3D, MutableIndexAccess) {
    IntegerVector3D vector;
    vector[0] = 5;
    vector[1] = 6;
    vector[2] = 7;
    EXPECT_EQ(vector.x, 5);
    EXPECT_EQ(vector.y, 6);
    EXPECT_EQ(vector.z, 7);
}

TEST(IntegerVector3D, EqualityTrue) {
    IntegerVector3D first(1, 2, 3);
    IntegerVector3D second(1, 2, 3);
    EXPECT_TRUE(first == second);
}

TEST(IntegerVector3D, EqualityFalse) {
    IntegerVector3D first(1, 2, 3);
    IntegerVector3D second(1, 2, 4);
    EXPECT_FALSE(first == second);
}
