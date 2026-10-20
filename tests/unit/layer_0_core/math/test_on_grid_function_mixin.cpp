#include <gtest/gtest.h>

#include <vector>

#include "layer_0_core/math/on_grid_function_mixin.hpp"
#include "layer_0_core/points_grid.hpp"

using namespace q3;

/**
 * Concrete subclass of OnGridFunctionMixin that exposes the protected
 * members for testing. Uses PointsGrid as the grid type and Real as
 * the value type.
 */
class TestableOnGridFunction : public OnGridFunctionMixin<PointsGrid, Real> {
  public:
    void set_values(const std::vector<Real>& new_values) {
        values_ = new_values;
    }
};

// ── Default state ───────────────────────────────────────────────────

TEST(OnGridFunctionMixin, DefaultHasZeroGridPoints) {
    TestableOnGridFunction function;
    EXPECT_EQ(function.number_of_grid_points(), 0);
}

TEST(OnGridFunctionMixin, DefaultValuesAreEmpty) {
    TestableOnGridFunction function;
    EXPECT_TRUE(function.values().empty());
}

// ── After setting values ────────────────────────────────────────────

TEST(OnGridFunctionMixin, NumberOfGridPointsMatchesValues) {
    TestableOnGridFunction function;
    function.set_values({1.0, 2.0, 3.0, 4.0, 5.0});
    EXPECT_EQ(function.number_of_grid_points(), 5);
}

TEST(OnGridFunctionMixin, ValuesReturnsConstReference) {
    TestableOnGridFunction function;
    function.set_values({10.0, 20.0});
    const auto& values = function.values();
    ASSERT_EQ(values.size(), 2u);
    EXPECT_DOUBLE_EQ(values[0], 10.0);
    EXPECT_DOUBLE_EQ(values[1], 20.0);
}

TEST(OnGridFunctionMixin, ValueAtIndexReturnsCorrectElement) {
    TestableOnGridFunction function;
    function.set_values({1.5, 2.5, 3.5});
    EXPECT_DOUBLE_EQ(function.value_at_index(0), 1.5);
    EXPECT_DOUBLE_EQ(function.value_at_index(1), 2.5);
    EXPECT_DOUBLE_EQ(function.value_at_index(2), 3.5);
}

TEST(OnGridFunctionMixin, MutableValueAtIndexAllowsModification) {
    TestableOnGridFunction function;
    function.set_values({1.0, 2.0, 3.0});
    function.value_at_index(1) = 99.0;
    EXPECT_DOUBLE_EQ(function.value_at_index(1), 99.0);
}

TEST(OnGridFunctionMixin, ConstValueAtIndexConsistentWithValues) {
    TestableOnGridFunction function;
    function.set_values({7.0, 8.0, 9.0});
    const auto& const_function = function;
    EXPECT_DOUBLE_EQ(const_function.value_at_index(0), const_function.values()[0]);
    EXPECT_DOUBLE_EQ(const_function.value_at_index(2), const_function.values()[2]);
}

// ── Grid accessor ───────────────────────────────────────────────────

TEST(OnGridFunctionMixin, GridReturnsDefaultGrid) {
    TestableOnGridFunction function;
    const PointsGrid& grid = function.grid();
    EXPECT_EQ(grid.number_of_points(), 0);
    EXPECT_TRUE(grid.empty());
}

/**
 * Concrete subclass with Complex values to test the template
 * works with different value types.
 */
class ComplexOnGridFunction : public OnGridFunctionMixin<PointsGrid, Complex> {
  public:
    void set_values(const std::vector<Complex>& new_values) {
        values_ = new_values;
    }
};

TEST(OnGridFunctionMixin, WorksWithComplexValueType) {
    ComplexOnGridFunction function;
    function.set_values({Complex(1.0, 2.0), Complex(3.0, 4.0)});
    EXPECT_EQ(function.number_of_grid_points(), 2);
    EXPECT_DOUBLE_EQ(function.value_at_index(0).real(), 1.0);
    EXPECT_DOUBLE_EQ(function.value_at_index(0).imag(), 2.0);
}
