#include <gtest/gtest.h>

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/mixins/ff_transformable.hpp"

using namespace q3;

/**
 * Concrete test subclass of FFTransformable with simple vector storage.
 * Implements the pure virtual transform methods as no-ops for unit testing
 * of the base class accessor functionality.
 */
class TestableFFTransformable : public FFTransformable<std::vector<Real>, std::vector<Complex>> {
  public:
    void to_reciprocal_space() override {
        transform_to_reciprocal_called_ = true;
    }

    void to_real_space() override {
        transform_to_real_called_ = true;
    }

    bool transform_to_reciprocal_called_ = false;
    bool transform_to_real_called_ = false;
};

// ── Default construction gives empty representations ──
TEST(FFTransformable, DefaultConstructionProducesEmptyRepresentations) {
    TestableFFTransformable field;

    EXPECT_TRUE(field.real_space_representation().empty());
    EXPECT_TRUE(field.reciprocal_space_representation().empty());
}

// ── Real space representation can be set and read via mutable accessor ──
TEST(FFTransformable, RealSpaceRepresentationCanBeSetAndRead) {
    TestableFFTransformable field;
    field.real_space_representation() = {1.0, 2.0, 3.0, 4.0};

    ASSERT_EQ(field.real_space_representation().size(), 4u);
    EXPECT_DOUBLE_EQ(field.real_space_representation()[0], 1.0);
    EXPECT_DOUBLE_EQ(field.real_space_representation()[3], 4.0);
}

// ── Reciprocal space representation can be set and read ──
TEST(FFTransformable, ReciprocalSpaceRepresentationCanBeSetAndRead) {
    TestableFFTransformable field;
    field.reciprocal_space_representation() = {Complex(1.0, 0.5), Complex(-1.0, 2.0)};

    ASSERT_EQ(field.reciprocal_space_representation().size(), 2u);
    EXPECT_DOUBLE_EQ(field.reciprocal_space_representation()[0].real(), 1.0);
    EXPECT_DOUBLE_EQ(field.reciprocal_space_representation()[0].imag(), 0.5);
    EXPECT_DOUBLE_EQ(field.reciprocal_space_representation()[1].real(), -1.0);
    EXPECT_DOUBLE_EQ(field.reciprocal_space_representation()[1].imag(), 2.0);
}

// ── Const accessors return the same data as mutable accessors ──
TEST(FFTransformable, ConstAccessorsReturnSameData) {
    TestableFFTransformable field;
    field.real_space_representation() = {10.0, 20.0};
    field.reciprocal_space_representation() = {Complex(5.0, 0.0)};

    const TestableFFTransformable& const_field = field;
    EXPECT_EQ(const_field.real_space_representation().size(), 2u);
    EXPECT_DOUBLE_EQ(const_field.real_space_representation()[0], 10.0);
    EXPECT_EQ(const_field.reciprocal_space_representation().size(), 1u);
    EXPECT_DOUBLE_EQ(const_field.reciprocal_space_representation()[0].real(), 5.0);
}

// ── to_reciprocal_space() can be called on subclass ──
TEST(FFTransformable, ToReciprocalSpaceCallsVirtualMethod) {
    TestableFFTransformable field;
    EXPECT_FALSE(field.transform_to_reciprocal_called_);

    field.to_reciprocal_space();
    EXPECT_TRUE(field.transform_to_reciprocal_called_);
}

// ── to_real_space() can be called on subclass ──
TEST(FFTransformable, ToRealSpaceCallsVirtualMethod) {
    TestableFFTransformable field;
    EXPECT_FALSE(field.transform_to_real_called_);

    field.to_real_space();
    EXPECT_TRUE(field.transform_to_real_called_);
}

// ── Virtual dispatch works through base pointer ──
TEST(FFTransformable, VirtualDispatchWorksThroughBasePointer) {
    TestableFFTransformable concrete_field;
    FFTransformable<std::vector<Real>, std::vector<Complex>>* base_pointer = &concrete_field;

    base_pointer->to_reciprocal_space();
    base_pointer->to_real_space();

    EXPECT_TRUE(concrete_field.transform_to_reciprocal_called_);
    EXPECT_TRUE(concrete_field.transform_to_real_called_);
}

// ── Mutable accessor allows in-place modification ──
TEST(FFTransformable, MutableAccessorAllowsInPlaceModification) {
    TestableFFTransformable field;
    field.real_space_representation().resize(100, 0.0);
    field.real_space_representation()[42] = 3.14;

    EXPECT_EQ(field.real_space_representation().size(), 100u);
    EXPECT_DOUBLE_EQ(field.real_space_representation()[42], 3.14);
}
