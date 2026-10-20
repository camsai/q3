#include "layer_0_core/execution/threading.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace q3::execution::threading;

/**
 * ─── Thread-count policy ────────────────────────────────────────────────────
 * The contract these tests pin is what makes a one-thread run of an OpenMP
 * build indistinguishable from a build without OpenMP: every call site asks
 * thread_count_for() how large a team to request, and a returned 1 turns the
 * `if` clause on its pragma false, so no team is forked at all.
 */

TEST(Threading, MaximumThreadsIsAtLeastOne) {
    EXPECT_GE(maximum_threads(), 1);
}

TEST(Threading, OutsideAParallelRegionThereIsOneThreadWithIndexZero) {
    EXPECT_FALSE(is_inside_parallel_region());
    EXPECT_EQ(thread_index(), 0);
    EXPECT_EQ(team_size(), 1);
}

TEST(Threading, CompiledWithOpenmpAgreesWithThePreprocessor) {
#ifdef _OPENMP
    EXPECT_TRUE(is_compiled_with_openmp());
#else
    EXPECT_FALSE(is_compiled_with_openmp());
    EXPECT_EQ(maximum_threads(), 1);
#endif
}

TEST(Threading, WorkTooSmallForATeamRunsOnOneThread) {
    EXPECT_EQ(thread_count_for(1000, 4096), 1);
    EXPECT_EQ(thread_count_for(4095, 4096), 1);
    // Exactly one thread's worth is still one thread: a second would get nothing.
    EXPECT_EQ(thread_count_for(4096, 4096), 1);
}

TEST(Threading, DegenerateWorkCountsRunOnOneThread) {
    EXPECT_EQ(thread_count_for(0, 1), 1);
    EXPECT_EQ(thread_count_for(-5, 1), 1);
    EXPECT_EQ(thread_count_for(100, 0), 1);
    EXPECT_EQ(thread_count_for(100, -1), 1);
}

TEST(Threading, ThreadCountNeverExceedsWhatTheRuntimeOffers) {
    EXPECT_LE(thread_count_for(1000000, 1), maximum_threads());
    EXPECT_GE(thread_count_for(1000000, 1), 1);
}

TEST(Threading, GrainSizeCapsTheTeamBelowTheRuntimeMaximum) {
    /**
     * Two grains' worth of work may occupy two threads at most, however many
     * the runtime would otherwise hand out.
     */
    EXPECT_LE(thread_count_for(2 * 4096, 4096), 2);
}

/**
 * ─── Slice bounds ───────────────────────────────────────────────────────────
 * A threaded reduction only stays reproducible if each thread always sums the
 * same contiguous run of terms. These pin that the slices partition the range
 * exactly -- no gap silently drops work, no overlap silently doubles it.
 */

TEST(Threading, SlicesPartitionTheRangeExactly) {
    for (long long work_items : {0LL, 1LL, 7LL, 64LL, 1000LL}) {
        for (int number_of_slices = 1; number_of_slices <= 8; ++number_of_slices) {
            long long expected_next = 0;
            for (int slice_index = 0; slice_index < number_of_slices; ++slice_index) {
                long long first = -1;
                long long last = -1;
                slice_bounds(work_items, number_of_slices, slice_index, first, last);
                EXPECT_EQ(first, expected_next)
                    << "work_items=" << work_items << " slices=" << number_of_slices;
                EXPECT_GE(last, first);
                expected_next = last;
            }
            EXPECT_EQ(expected_next, work_items)
                << "work_items=" << work_items << " slices=" << number_of_slices;
        }
    }
}

TEST(Threading, SliceSizesDifferByAtMostOne) {
    constexpr long long work_items = 1000;
    constexpr int number_of_slices = 7;
    long long smallest = work_items;
    long long largest = 0;
    for (int slice_index = 0; slice_index < number_of_slices; ++slice_index) {
        long long first = 0;
        long long last = 0;
        slice_bounds(work_items, number_of_slices, slice_index, first, last);
        long long size = last - first;
        smallest = std::min(smallest, size);
        largest = std::max(largest, size);
    }
    EXPECT_LE(largest - smallest, 1);
}

TEST(Threading, OneSliceIsTheWholeRange) {
    long long first = -1;
    long long last = -1;
    slice_bounds(500, 1, 0, first, last);
    EXPECT_EQ(first, 0);
    EXPECT_EQ(last, 500);
}
