/**
 * The PAW one-center convention registry and the factory that reads it: the
 * absolute convention core seeds, and a refusal for a convention the build
 * does not provide.
 */

#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_4_computation/paw/one_center/convention/factory.hpp"
#include "layer_4_computation/paw/one_center/convention/registry.hpp"
#include <gtest/gtest.h>

#include <stdexcept>

using namespace q3;
using namespace q3::paw_one_center;

TEST(OneCenterConventionRegistryTest, CoreSeedsTheAbsoluteConvention) {
    EXPECT_NE(one_center_convention_registry().find("absolute"), nullptr);
}

TEST(OneCenterConventionRegistryTest, ADatasetDefaultsToTheAbsoluteConvention) {
    const PawPseudopotential dataset;
    EXPECT_EQ(dataset.one_center_convention, "absolute");
    EXPECT_NE(create_one_center_convention(dataset), nullptr);
}

TEST(OneCenterConventionRegistryTest, AConventionTheBuildDoesNotProvideIsRefused) {
    PawPseudopotential dataset;
    dataset.one_center_convention = "registry_test_no_such_convention";
    EXPECT_THROW(create_one_center_convention(dataset), std::runtime_error);
}
