#include <gtest/gtest.h>

#include "layer_0_core/types/vector_3d.hpp"

using namespace q3;

TEST(Vector3D, DefaultConstruction) {
    Vector3D vector;
    EXPECT_DOUBLE_EQ(vector.x, 0.0);
    EXPECT_DOUBLE_EQ(vector.y, 0.0);
    EXPECT_DOUBLE_EQ(vector.z, 0.0);
}

TEST(Vector3D, ParameterizedConstruction) {
    Vector3D vector(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(vector.x, 1.0);
    EXPECT_DOUBLE_EQ(vector.y, 2.0);
    EXPECT_DOUBLE_EQ(vector.z, 3.0);
}

TEST(Vector3D, ConstIndexAccess) {
    const Vector3D vector(4.0, 5.0, 6.0);
    EXPECT_DOUBLE_EQ(vector[0], 4.0);
    EXPECT_DOUBLE_EQ(vector[1], 5.0);
    EXPECT_DOUBLE_EQ(vector[2], 6.0);
}

TEST(Vector3D, MutableIndexAccess) {
    Vector3D vector;
    vector[0] = 7.0;
    vector[1] = 8.0;
    vector[2] = 9.0;
    EXPECT_DOUBLE_EQ(vector.x, 7.0);
    EXPECT_DOUBLE_EQ(vector.y, 8.0);
    EXPECT_DOUBLE_EQ(vector.z, 9.0);
}

TEST(Vector3D, Addition) {
    Vector3D first(1.0, 2.0, 3.0);
    Vector3D second(4.0, 5.0, 6.0);
    Vector3D result = first + second;
    EXPECT_DOUBLE_EQ(result.x, 5.0);
    EXPECT_DOUBLE_EQ(result.y, 7.0);
    EXPECT_DOUBLE_EQ(result.z, 9.0);
}

TEST(Vector3D, Subtraction) {
    Vector3D first(10.0, 20.0, 30.0);
    Vector3D second(1.0, 2.0, 3.0);
    Vector3D result = first - second;
    EXPECT_DOUBLE_EQ(result.x, 9.0);
    EXPECT_DOUBLE_EQ(result.y, 18.0);
    EXPECT_DOUBLE_EQ(result.z, 27.0);
}

TEST(Vector3D, ScalarMultiplication) {
    Vector3D vector(1.0, 2.0, 3.0);
    Vector3D result = vector * 3.0;
    EXPECT_DOUBLE_EQ(result.x, 3.0);
    EXPECT_DOUBLE_EQ(result.y, 6.0);
    EXPECT_DOUBLE_EQ(result.z, 9.0);
}

TEST(Vector3D, ScalarDivision) {
    Vector3D vector(6.0, 9.0, 12.0);
    Vector3D result = vector / 3.0;
    EXPECT_DOUBLE_EQ(result.x, 2.0);
    EXPECT_DOUBLE_EQ(result.y, 3.0);
    EXPECT_DOUBLE_EQ(result.z, 4.0);
}

TEST(Vector3D, UnaryNegation) {
    Vector3D vector(1.0, -2.0, 3.0);
    Vector3D result = -vector;
    EXPECT_DOUBLE_EQ(result.x, -1.0);
    EXPECT_DOUBLE_EQ(result.y, 2.0);
    EXPECT_DOUBLE_EQ(result.z, -3.0);
}

TEST(Vector3D, PlusEqualsOperator) {
    Vector3D vector(1.0, 2.0, 3.0);
    Vector3D addend(10.0, 20.0, 30.0);
    vector += addend;
    EXPECT_DOUBLE_EQ(vector.x, 11.0);
    EXPECT_DOUBLE_EQ(vector.y, 22.0);
    EXPECT_DOUBLE_EQ(vector.z, 33.0);
}

TEST(Vector3D, DotProduct) {
    Vector3D first(1.0, 2.0, 3.0);
    Vector3D second(4.0, 5.0, 6.0);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_DOUBLE_EQ(first.dot(second), 32.0);
}

TEST(Vector3D, DotProductOrthogonalVectors) {
    Vector3D x_axis(1.0, 0.0, 0.0);
    Vector3D y_axis(0.0, 1.0, 0.0);
    EXPECT_DOUBLE_EQ(x_axis.dot(y_axis), 0.0);
}

TEST(Vector3D, CrossProduct) {
    Vector3D x_axis(1.0, 0.0, 0.0);
    Vector3D y_axis(0.0, 1.0, 0.0);
    Vector3D result = x_axis.cross(y_axis);
    EXPECT_DOUBLE_EQ(result.x, 0.0);
    EXPECT_DOUBLE_EQ(result.y, 0.0);
    EXPECT_DOUBLE_EQ(result.z, 1.0);
}

TEST(Vector3D, CrossProductAnticommutative) {
    Vector3D first(1.0, 2.0, 3.0);
    Vector3D second(4.0, 5.0, 6.0);
    Vector3D cross_ab = first.cross(second);
    Vector3D cross_ba = second.cross(first);
    EXPECT_DOUBLE_EQ(cross_ab.x, -cross_ba.x);
    EXPECT_DOUBLE_EQ(cross_ab.y, -cross_ba.y);
    EXPECT_DOUBLE_EQ(cross_ab.z, -cross_ba.z);
}

TEST(Vector3D, NormSquared) {
    Vector3D vector(3.0, 4.0, 0.0);
    EXPECT_DOUBLE_EQ(vector.norm2(), 25.0);
}

TEST(Vector3D, Norm) {
    Vector3D vector(3.0, 4.0, 0.0);
    EXPECT_DOUBLE_EQ(vector.norm(), 5.0);
}

TEST(Vector3D, NormOfUnitVector) {
    Vector3D unit_z(0.0, 0.0, 1.0);
    EXPECT_DOUBLE_EQ(unit_z.norm(), 1.0);
}

TEST(Vector3D, FreeScalarTimesVector) {
    Vector3D vector(1.0, 2.0, 3.0);
    Vector3D result = 2.5 * vector;
    EXPECT_DOUBLE_EQ(result.x, 2.5);
    EXPECT_DOUBLE_EQ(result.y, 5.0);
    EXPECT_DOUBLE_EQ(result.z, 7.5);
}

TEST(Vector3D, FreeScalarTimesVectorMatchesMemberVersion) {
    Vector3D vector(1.5, -2.5, 3.5);
    Real scalar = 4.0;
    Vector3D left_product = scalar * vector;
    Vector3D right_product = vector * scalar;
    EXPECT_DOUBLE_EQ(left_product.x, right_product.x);
    EXPECT_DOUBLE_EQ(left_product.y, right_product.y);
    EXPECT_DOUBLE_EQ(left_product.z, right_product.z);
}
