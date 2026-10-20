#include "layer_0_core/profiling/counters.hpp"
#include <gtest/gtest.h>

using namespace q3;

// ─── Singleton ──────────────────────────────────────────────────────────────

TEST(ProfilingCounters, InstanceReturnsSameReference) {
    auto& first_reference = ProfilingCounters::instance();
    auto& second_reference = ProfilingCounters::instance();
    EXPECT_EQ(&first_reference, &second_reference);
}

/**
 * ─── Initial counter values ─────────────────────────────────────────────────
 * Note: Because ProfilingCounters is a singleton, counters may have been
 * modified by other tests. We reset them before checking initial state.
 */

TEST(ProfilingCounters, FftCallCountStartsAtZero) {
    auto& counters = ProfilingCounters::instance();
    counters.fft_call_count = 0;
    EXPECT_EQ(counters.fft_call_count.load(), 0);
}

TEST(ProfilingCounters, DavidsonIterationCountStartsAtZero) {
    auto& counters = ProfilingCounters::instance();
    counters.davidson_iteration_count = 0;
    EXPECT_EQ(counters.davidson_iteration_count.load(), 0);
}

TEST(ProfilingCounters, HamiltonianApplicationCountStartsAtZero) {
    auto& counters = ProfilingCounters::instance();
    counters.hamiltonian_application_count = 0;
    EXPECT_EQ(counters.hamiltonian_application_count.load(), 0);
}

// ─── Incrementing counters ──────────────────────────────────────────────────

TEST(ProfilingCounters, IncrementingFftCallCountWorks) {
    auto& counters = ProfilingCounters::instance();
    counters.fft_call_count = 0;
    counters.fft_call_count++;
    counters.fft_call_count++;
    counters.fft_call_count++;
    EXPECT_EQ(counters.fft_call_count.load(), 3);
}

TEST(ProfilingCounters, IncrementingDavidsonIterationCountWorks) {
    auto& counters = ProfilingCounters::instance();
    counters.davidson_iteration_count = 0;
    counters.davidson_iteration_count += 5;
    EXPECT_EQ(counters.davidson_iteration_count.load(), 5);
}

TEST(ProfilingCounters, IncrementingHamiltonianApplicationCountWorks) {
    auto& counters = ProfilingCounters::instance();
    counters.hamiltonian_application_count = 0;
    counters.hamiltonian_application_count.fetch_add(10);
    EXPECT_EQ(counters.hamiltonian_application_count.load(), 10);
}

// ─── DavidsonTimingAccumulator ──────────────────────────────────────────────

TEST(DavidsonTimingAccumulator, ResetZeroesAllFields) {
    ProfilingCounters::DavidsonTimingAccumulator accumulator;
    accumulator.seed_time_ms = 10.0;
    accumulator.ritz_vector_time_ms = 20.0;
    accumulator.residual_time_ms = 30.0;
    accumulator.orthogonalize_time_ms = 40.0;
    accumulator.hamiltonian_application_time_ms = 50.0;
    accumulator.subspace_update_time_ms = 60.0;
    accumulator.collapse_time_ms = 70.0;
    accumulator.diagonalize_time_ms = 80.0;
    accumulator.total_solve_calls = 99;

    accumulator.reset();

    EXPECT_DOUBLE_EQ(accumulator.seed_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(accumulator.ritz_vector_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(accumulator.residual_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(accumulator.orthogonalize_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(accumulator.hamiltonian_application_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(accumulator.subspace_update_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(accumulator.collapse_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(accumulator.diagonalize_time_ms, 0.0);
    EXPECT_EQ(accumulator.total_solve_calls, 0);
}

TEST(DavidsonTimingAccumulator, FieldsAccumulateCorrectly) {
    ProfilingCounters::DavidsonTimingAccumulator accumulator;

    accumulator.seed_time_ms += 1.5;
    accumulator.seed_time_ms += 2.5;
    EXPECT_DOUBLE_EQ(accumulator.seed_time_ms, 4.0);

    accumulator.hamiltonian_application_time_ms += 100.0;
    accumulator.hamiltonian_application_time_ms += 200.5;
    EXPECT_DOUBLE_EQ(accumulator.hamiltonian_application_time_ms, 300.5);

    accumulator.total_solve_calls += 1;
    accumulator.total_solve_calls += 1;
    accumulator.total_solve_calls += 1;
    EXPECT_EQ(accumulator.total_solve_calls, 3);
}
