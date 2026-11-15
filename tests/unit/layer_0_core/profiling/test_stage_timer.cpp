/**
 * StageTimer: each scope adds its wall time and one call to the named stage, so a results
 * file can say both how long a stage took and how many times it ran.
 */

#include "layer_0_core/profiling/stage_timer.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

using namespace q3;

TEST(StageTimer, RecordsTimeAndOneCallPerScope) {
    auto& counters = ProfilingCounters::instance();
    counters.stage_time_ms.clear();
    counters.stage_call_count.clear();
    {
        StageTimer timer("test.stage");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    { StageTimer timer("test.stage"); }
    ASSERT_EQ(counters.stage_call_count.count("test.stage"), 1u);
    EXPECT_EQ(counters.stage_call_count.at("test.stage"), 2);
    EXPECT_GE(counters.stage_time_ms.at("test.stage"), 2.0);
}

TEST(StageTimer, NestedStagesAreBothRecorded) {
    auto& counters = ProfilingCounters::instance();
    counters.stage_time_ms.clear();
    counters.stage_call_count.clear();
    {
        StageTimer outer("test.outer");
        StageTimer inner("test.inner");
    }
    EXPECT_EQ(counters.stage_call_count.at("test.outer"), 1);
    EXPECT_EQ(counters.stage_call_count.at("test.inner"), 1);
    EXPECT_GE(counters.stage_time_ms.at("test.outer"), counters.stage_time_ms.at("test.inner"));
}

TEST(StageTimer, ResetClearsTheStages) {
    auto& counters = ProfilingCounters::instance();
    { StageTimer timer("test.cleared"); }
    counters.reset();
    EXPECT_TRUE(counters.stage_time_ms.empty());
    EXPECT_TRUE(counters.stage_call_count.empty());
}
