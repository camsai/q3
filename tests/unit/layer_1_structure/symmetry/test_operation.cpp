#include <gtest/gtest.h>

#include "layer_1_structure/symmetry/operation.hpp"

using namespace q3;

TEST(SymmetryOperation, DefaultConstruction) {
    SymmetryOperation operation{};
    // Default-initialized arrays should be zero
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            EXPECT_EQ(operation.rotation[row][column], 0);
            EXPECT_EQ(operation.reciprocal_rotation[row][column], 0);
        }
    }
    EXPECT_DOUBLE_EQ(operation.fractional_translation.x, 0.0);
    EXPECT_DOUBLE_EQ(operation.fractional_translation.y, 0.0);
    EXPECT_DOUBLE_EQ(operation.fractional_translation.z, 0.0);
    EXPECT_TRUE(operation.atom_map.empty());
}

TEST(SymmetryOperation, IdentityRotation) {
    SymmetryOperation identity;
    identity.rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    identity.reciprocal_rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    identity.fractional_translation = Vector3D(0.0, 0.0, 0.0);
    for (int index = 0; index < 3; index++) {
        EXPECT_EQ(identity.rotation[index][index], 1);
        EXPECT_EQ(identity.reciprocal_rotation[index][index], 1);
    }
    EXPECT_DOUBLE_EQ(identity.fractional_translation.x, 0.0);
}

TEST(SymmetryOperation, InversionRotation) {
    SymmetryOperation inversion;
    inversion.rotation = {{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}};
    inversion.reciprocal_rotation = {{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}};
    inversion.fractional_translation = Vector3D(0.0, 0.0, 0.0);
    for (int index = 0; index < 3; index++) {
        EXPECT_EQ(inversion.rotation[index][index], -1);
    }
    // Off-diagonal elements should be zero
    EXPECT_EQ(inversion.rotation[0][1], 0);
    EXPECT_EQ(inversion.rotation[0][2], 0);
    EXPECT_EQ(inversion.rotation[1][0], 0);
}

TEST(SymmetryOperation, NonSymmorphicTranslation) {
    SymmetryOperation operation;
    operation.rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    operation.reciprocal_rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    operation.fractional_translation = Vector3D(0.5, 0.5, 0.0);
    EXPECT_DOUBLE_EQ(operation.fractional_translation.x, 0.5);
    EXPECT_DOUBLE_EQ(operation.fractional_translation.y, 0.5);
    EXPECT_DOUBLE_EQ(operation.fractional_translation.z, 0.0);
}

TEST(SymmetryOperation, AtomMap) {
    SymmetryOperation operation;
    operation.rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    operation.reciprocal_rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    operation.fractional_translation = Vector3D(0.0, 0.0, 0.0);
    operation.atom_map = {1, 0, 2};
    ASSERT_EQ(operation.atom_map.size(), 3u);
    EXPECT_EQ(operation.atom_map[0], 1);
    EXPECT_EQ(operation.atom_map[1], 0);
    EXPECT_EQ(operation.atom_map[2], 2);
}

TEST(SymmetryOperation, CopySemantics) {
    SymmetryOperation original;
    original.rotation = {{{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}}};
    original.reciprocal_rotation = {{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}};
    original.fractional_translation = Vector3D(0.0, 0.0, 0.0);
    original.atom_map = {0, 1};
    SymmetryOperation copy = original;
    EXPECT_EQ(copy.rotation[0][1], 1);
    EXPECT_EQ(copy.reciprocal_rotation[1][0], 1);
    ASSERT_EQ(copy.atom_map.size(), 2u);
    EXPECT_EQ(copy.atom_map[0], 0);
}
