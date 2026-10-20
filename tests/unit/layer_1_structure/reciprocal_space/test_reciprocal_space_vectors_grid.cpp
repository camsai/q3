#include <gtest/gtest.h>

#include "layer_1_structure/reciprocal_space/reciprocal_space_vectors_grid.hpp"

using namespace q3;

TEST(ReciprocalSpaceVectorsGrid, DefaultConstruction) {
    ReciprocalSpaceVectorsGrid grid;
    EXPECT_EQ(grid.number_of_vectors(), 0);
    EXPECT_TRUE(grid.vectors().empty());
}

TEST(ReciprocalSpaceVectorsGrid, InheritsFromPointsGrid) {
    ReciprocalSpaceVectorsGrid grid;
    // Verify the PointsGrid base class methods are accessible
    EXPECT_EQ(grid.number_of_points(), 0);
    EXPECT_TRUE(grid.empty());
    EXPECT_TRUE(grid.points().empty());
}

/**
 * Test subclass that exposes the protected vectors_ member for testing.
 */
class TestableReciprocalSpaceVectorsGrid : public ReciprocalSpaceVectorsGrid {
  public:
    void add_vector(const ReciprocalSpaceVector& vector) {
        vectors_.push_back(vector);
    }
};

TEST(ReciprocalSpaceVectorsGrid, NumberOfVectorsAfterAdding) {
    TestableReciprocalSpaceVectorsGrid grid;
    grid.add_vector({Vector3D(1.0, 0.0, 0.0)});
    grid.add_vector({Vector3D(0.0, 1.0, 0.0)});
    grid.add_vector({Vector3D(0.0, 0.0, 1.0)});
    EXPECT_EQ(grid.number_of_vectors(), 3);
}

TEST(ReciprocalSpaceVectorsGrid, VectorAccessByIndex) {
    TestableReciprocalSpaceVectorsGrid grid;
    grid.add_vector({Vector3D(1.5, 2.5, 3.5)});
    grid.add_vector({Vector3D(-1.0, -2.0, -3.0)});
    ASSERT_EQ(grid.number_of_vectors(), 2);
    EXPECT_DOUBLE_EQ(grid.vector(0).coordinates.x, 1.5);
    EXPECT_DOUBLE_EQ(grid.vector(0).coordinates.y, 2.5);
    EXPECT_DOUBLE_EQ(grid.vector(0).coordinates.z, 3.5);
    EXPECT_DOUBLE_EQ(grid.vector(1).coordinates.x, -1.0);
}

TEST(ReciprocalSpaceVectorsGrid, VectorsReturnsConstReference) {
    TestableReciprocalSpaceVectorsGrid grid;
    grid.add_vector({Vector3D(1.0, 2.0, 3.0)});
    const auto& vectors = grid.vectors();
    ASSERT_EQ(vectors.size(), 1u);
    EXPECT_DOUBLE_EQ(vectors[0].coordinates.x, 1.0);
}
