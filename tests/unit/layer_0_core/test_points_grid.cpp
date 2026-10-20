#include "layer_0_core/points_grid.hpp"
#include <gtest/gtest.h>

using namespace q3;

/**
 * Concrete test subclass that exposes the protected points_ vector
 * so we can add points and test the PointsGrid accessors.
 */
class TestPointsGrid : public PointsGrid {
  public:
    void add_point(const WeightedPoint3D& weighted_point) {
        points_.push_back(weighted_point);
    }
};

// ─── Point3D ────────────────────────────────────────────────────────────────

TEST(Point3D, DefaultConstructionHasZeroCoordinates) {
    Point3D point;
    EXPECT_DOUBLE_EQ(point.coordinates.x, 0.0);
    EXPECT_DOUBLE_EQ(point.coordinates.y, 0.0);
    EXPECT_DOUBLE_EQ(point.coordinates.z, 0.0);
}

// ─── WeightedPoint3D ────────────────────────────────────────────────────────

TEST(WeightedPoint3D, DefaultConstructionHasZeroCoordinatesAndUnitWeight) {
    WeightedPoint3D weighted_point;
    EXPECT_DOUBLE_EQ(weighted_point.weight, 1.0);
    EXPECT_DOUBLE_EQ(weighted_point.coordinates.x, 0.0);
    EXPECT_DOUBLE_EQ(weighted_point.coordinates.y, 0.0);
    EXPECT_DOUBLE_EQ(weighted_point.coordinates.z, 0.0);
}

TEST(WeightedPoint3D, CustomWeightAndCoordinatesArePreserved) {
    WeightedPoint3D weighted_point;
    weighted_point.coordinates = {0.1, 0.2, 0.3};
    weighted_point.weight = 0.42;
    EXPECT_DOUBLE_EQ(weighted_point.weight, 0.42);
    EXPECT_DOUBLE_EQ(weighted_point.coordinates.x, 0.1);
    EXPECT_DOUBLE_EQ(weighted_point.coordinates.y, 0.2);
    EXPECT_DOUBLE_EQ(weighted_point.coordinates.z, 0.3);
}

// ─── PointsGrid ─────────────────────────────────────────────────────────────

TEST(PointsGrid, DefaultConstructionIsEmpty) {
    PointsGrid grid;
    EXPECT_TRUE(grid.empty());
    EXPECT_EQ(grid.number_of_points(), 0);
    EXPECT_TRUE(grid.points().empty());
}

TEST(PointsGrid, AddingSinglePointMakesGridNonEmpty) {
    TestPointsGrid grid;
    WeightedPoint3D point;
    point.coordinates = {0.5, 0.5, 0.5};
    point.weight = 1.0;
    grid.add_point(point);

    EXPECT_FALSE(grid.empty());
    EXPECT_EQ(grid.number_of_points(), 1);
}

TEST(PointsGrid, AddingMultiplePointsReportsCorrectCount) {
    TestPointsGrid grid;

    WeightedPoint3D point_a;
    point_a.coordinates = {0.0, 0.0, 0.0};
    point_a.weight = 0.25;
    grid.add_point(point_a);

    WeightedPoint3D point_b;
    point_b.coordinates = {0.5, 0.5, 0.0};
    point_b.weight = 0.50;
    grid.add_point(point_b);

    WeightedPoint3D point_c;
    point_c.coordinates = {0.5, 0.0, 0.5};
    point_c.weight = 0.25;
    grid.add_point(point_c);

    EXPECT_EQ(grid.number_of_points(), 3);
    EXPECT_FALSE(grid.empty());
}

TEST(PointsGrid, PointAccessorReturnsCorrectCoordinatesAndWeight) {
    TestPointsGrid grid;

    WeightedPoint3D point_a;
    point_a.coordinates = {0.1, 0.2, 0.3};
    point_a.weight = 0.6;
    grid.add_point(point_a);

    WeightedPoint3D point_b;
    point_b.coordinates = {0.7, 0.8, 0.9};
    point_b.weight = 0.4;
    grid.add_point(point_b);

    EXPECT_DOUBLE_EQ(grid.point(0).coordinates.x, 0.1);
    EXPECT_DOUBLE_EQ(grid.point(0).coordinates.y, 0.2);
    EXPECT_DOUBLE_EQ(grid.point(0).coordinates.z, 0.3);
    EXPECT_DOUBLE_EQ(grid.point(0).weight, 0.6);

    EXPECT_DOUBLE_EQ(grid.point(1).coordinates.x, 0.7);
    EXPECT_DOUBLE_EQ(grid.point(1).coordinates.y, 0.8);
    EXPECT_DOUBLE_EQ(grid.point(1).coordinates.z, 0.9);
    EXPECT_DOUBLE_EQ(grid.point(1).weight, 0.4);
}

TEST(PointsGrid, PointsVectorMatchesAddedPoints) {
    TestPointsGrid grid;

    WeightedPoint3D point;
    point.coordinates = {0.25, 0.25, 0.25};
    point.weight = 0.5;
    grid.add_point(point);

    const auto& all_points = grid.points();
    ASSERT_EQ(static_cast<int>(all_points.size()), 1);
    EXPECT_DOUBLE_EQ(all_points[0].coordinates.x, 0.25);
    EXPECT_DOUBLE_EQ(all_points[0].coordinates.y, 0.25);
    EXPECT_DOUBLE_EQ(all_points[0].coordinates.z, 0.25);
    EXPECT_DOUBLE_EQ(all_points[0].weight, 0.5);
}
