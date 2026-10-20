#include <gtest/gtest.h>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/core_charge_data.hpp"

using namespace q3;

/**
 * Default-constructed CoreChargeData should have no nonlinear core correction
 * and an empty reciprocal-space core charge density vector.
 */
TEST(CoreChargeData, DefaultConstruction) {
    CoreChargeData data;
    EXPECT_FALSE(data.has_nonlinear_core_correction());
    EXPECT_TRUE(data.core_charge_density_reciprocal().empty());
}

/**
 * Multiple default-constructed instances should all report no NLCC.
 */
TEST(CoreChargeData, MultipleDefaultInstancesNoNLCC) {
    CoreChargeData data_1;
    CoreChargeData data_2;
    EXPECT_FALSE(data_1.has_nonlinear_core_correction());
    EXPECT_FALSE(data_2.has_nonlinear_core_correction());
    EXPECT_TRUE(data_1.core_charge_density_reciprocal().empty());
    EXPECT_TRUE(data_2.core_charge_density_reciprocal().empty());
}
