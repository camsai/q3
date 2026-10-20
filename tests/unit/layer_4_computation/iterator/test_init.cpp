#include <gtest/gtest.h>

#include "layer_4_computation/iterator/init.hpp"

using namespace q3;

/**
 * ── Iteration count after run() ─────────────────────────────────────
 *
 * Once run() returns, iteration() is what the run reports as the number of
 * iterations it performed: the SCF state, the checkpoint header and the results
 * file all read it there. The loop has two exits and the count must be right on
 * both -- including running out of budget, which is the exit every failed run
 * takes and the one whose count a reader actually needs.
 *
 * Driven through a stub iterator that counts its own iterations, so the value
 * under test is checked against an independent tally rather than against the
 * loop's arithmetic restated.
 */

namespace {

/// An SCF iterator that converges at a chosen iteration, or never (0).
class ScriptedConvergenceIterator : public SCFIterator {
  public:
    ScriptedConvergenceIterator(int maximum_iterations, int converge_at_iteration)
        : SCFIterator(make_config(maximum_iterations))
        , converge_at_iteration_(converge_at_iteration) {}

    Real total_energy() const override {
        return 0.0;
    }

    void perform_iteration() override {
        ++performed_iterations_;
    }

    bool is_iteration_converged() const override {
        return converge_at_iteration_ > 0 && iteration_ >= converge_at_iteration_;
    }

    /// Iterations performed since the last run() started, tallied independently of the loop.
    [[nodiscard]] int performed_iterations() const {
        return performed_iterations_;
    }

    void initialize() override {
        performed_iterations_ = 0;
        SCFIterator::initialize();
    }
    void finalize() override {}
    void report_iteration() override {}
    void report_non_convergence() override {}
    void report_convergence() override {}

  private:
    static SCFIteratorConfig make_config(int maximum_iterations) {
        SCFIteratorConfig config;
        config.maximum_iterations = maximum_iterations;
        return config;
    }

    int converge_at_iteration_;
    int performed_iterations_ = 0;
};

} // namespace

TEST(SCFIteratorIterationCount, IsTheIterationTheRunConvergedOn) {
    ScriptedConvergenceIterator iterator(/*maximum_iterations=*/10, /*converge_at_iteration=*/4);

    iterator.run();

    ASSERT_TRUE(iterator.is_converged());
    EXPECT_EQ(iterator.iteration(), 4);
    EXPECT_EQ(iterator.iteration(), iterator.performed_iterations());
}

TEST(SCFIteratorIterationCount, IsTheWholeBudgetWhenTheRunDoesNotConverge) {
    /**
     * The regression: the loop used to leave the counter one past the budget
     * here, so a run limited to 5 iterations reported 6.
     */
    ScriptedConvergenceIterator iterator(/*maximum_iterations=*/5, /*converge_at_iteration=*/0);

    iterator.run();

    ASSERT_FALSE(iterator.is_converged());
    EXPECT_EQ(iterator.performed_iterations(), 5);
    EXPECT_EQ(iterator.iteration(), 5);
}

TEST(SCFIteratorIterationCount, ConvergingOnTheLastIterationOfTheBudgetCountsThatIteration) {
    /**
     * Both exits land on the same count at this boundary; only the convergence
     * flag tells them apart, which is why the results report both.
     */
    ScriptedConvergenceIterator iterator(/*maximum_iterations=*/5, /*converge_at_iteration=*/5);

    iterator.run();

    ASSERT_TRUE(iterator.is_converged());
    EXPECT_EQ(iterator.iteration(), 5);
    EXPECT_EQ(iterator.iteration(), iterator.performed_iterations());
}

TEST(SCFIteratorIterationCount, ASecondRunCountsOnlyItsOwnIterations) {
    ScriptedConvergenceIterator iterator(/*maximum_iterations=*/5, /*converge_at_iteration=*/0);
    iterator.run();

    iterator.set_maximum_iterations(3);
    iterator.run();

    EXPECT_EQ(iterator.performed_iterations(), 3);
    EXPECT_EQ(iterator.iteration(), 3);
}
