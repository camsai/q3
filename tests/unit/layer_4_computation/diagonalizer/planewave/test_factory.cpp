#include <gtest/gtest.h>

#include <stdexcept>

#include "layer_4_computation/diagonalizer/planewave/factory.hpp"

using namespace q3;

TEST(DiagonalizerFactory, DavidsonReturnsNonNull) {
    auto diagonalizer = create_planewave_diagonalizer("davidson");
    ASSERT_NE(diagonalizer, nullptr);
    EXPECT_EQ(diagonalizer->name(), "davidson");
}

TEST(DiagonalizerFactory, DirectReturnsNonNull) {
    auto diagonalizer = create_planewave_diagonalizer("direct");
    ASSERT_NE(diagonalizer, nullptr);
    EXPECT_EQ(diagonalizer->name(), "direct");
}

TEST(DiagonalizerFactory, UnknownNameThrowsRuntimeError) {
    EXPECT_THROW(create_planewave_diagonalizer("unknown"), std::runtime_error);
}

TEST(DiagonalizerFactory, EmptyNameThrowsRuntimeError) {
    EXPECT_THROW(create_planewave_diagonalizer(""), std::runtime_error);
}
