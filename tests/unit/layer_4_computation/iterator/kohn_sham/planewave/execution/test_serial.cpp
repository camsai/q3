#include <gtest/gtest.h>

#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

/**
 * The checkpoint writes from the serial policy through the same calls a pooled
 * run makes, so the one process must count as owning its k-points whole, and
 * a k-point's local position must be its global one.
 */
TEST(PlanewaveSerialExecutionPolicyTest, TheOneProcessOwnsEveryKpointUnderItsOwnIndex) {
    execution::PlanewaveSerialExecutionPolicy policy;
    EXPECT_TRUE(policy.each_process_owns_whole_kpoints());
    for (int kpoint_index = 0; kpoint_index < 5; ++kpoint_index) {
        EXPECT_EQ(policy.global_kpoint_index(kpoint_index), kpoint_index);
    }
}
