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

/**
 * A serial run is one process in one pool with nothing distributed, and says
 * so in results.json the same way an MPI run states its topology.
 */
TEST(PlanewaveSerialExecutionPolicyTest, TheExecutionRecordIsOneUndistributedProcess) {
    execution::PlanewaveSerialExecutionPolicy policy;

    const generated::ResultsExecution record = policy.execution_record();

    EXPECT_EQ(record.number_of_processes, 1);
    EXPECT_EQ(record.number_of_kpoint_pools, 1);
    EXPECT_EQ(record.band_group_size, 1);
    EXPECT_FALSE(record.reciprocal_space_vectors_are_distributed);
    EXPECT_EQ(record.number_of_threads_per_process, execution::threading::maximum_threads());
    EXPECT_GE(record.peak_resident_memory_bytes, 0);
}
