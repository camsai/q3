#include <gtest/gtest.h>

#include <stdexcept>

#include "layer_4_computation/iterator/internal/mixing/factory.hpp"

using namespace q3;

TEST(MixerFactory, BroydenReturnsNonNull) {
    auto mixer = MixerFactory::create("broyden");
    ASSERT_NE(mixer, nullptr);
    EXPECT_EQ(mixer->name(), "broyden");
}

TEST(MixerFactory, AndersonReturnsNonNull) {
    auto mixer = MixerFactory::create("anderson");
    ASSERT_NE(mixer, nullptr);
    EXPECT_EQ(mixer->name(), "anderson");
}

TEST(MixerFactory, LinearReturnsNonNull) {
    auto mixer = MixerFactory::create("linear");
    ASSERT_NE(mixer, nullptr);
    EXPECT_EQ(mixer->name(), "linear");
}

TEST(MixerFactory, BroydenParametersPassedCorrectly) {
    const Real mixing_parameter = 0.5;
    const int history_depth = 4;
    auto mixer = MixerFactory::create("broyden", mixing_parameter, history_depth);
    ASSERT_NE(mixer, nullptr);
    EXPECT_DOUBLE_EQ(mixer->mixing_parameter(), mixing_parameter);
    EXPECT_EQ(mixer->history_depth(), history_depth);
}

TEST(MixerFactory, UnknownModeThrowsRuntimeError) {
    EXPECT_THROW(MixerFactory::create("unknown"), std::runtime_error);
}
