#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "layer_0_core/math/convergence_window.hpp"

using namespace q3;

TEST(ConvergenceWindow, NeedsMoreValuesThanTheWindowBeforeItCanConverge) {
    ConvergenceWindow window(1.0e-6, 3);
    for (int step = 0; step < 3; step++) {
        window.record(1.0);
        EXPECT_FALSE(window.has_converged()) << "after " << step + 1 << " identical values";
    }
    window.record(1.0);
    EXPECT_TRUE(window.has_converged()) << "the fourth value gives three differences";
    EXPECT_EQ(window.number_of_steps(), 4);
}

TEST(ConvergenceWindow, EveryDifferenceInTheWindowMustBeInsideTheRelativeTolerance) {
    ConvergenceWindow window(1.0e-6, 2);
    window.record(1000.0);
    window.record(1000.0 + 5.0e-4);
    window.record(1000.0 + 6.0e-4);
    EXPECT_TRUE(window.has_converged()) << "differences of 1e-7 and 6e-7 relative";
    window.record(1000.0 + 3.0e-3);
    EXPECT_FALSE(window.has_converged()) << "one difference of 3e-6 relative breaks it";
    window.record(1000.0 + 3.0e-3);
    EXPECT_FALSE(window.has_converged()) << "the earlier step is still inside the window";
    window.record(1000.0 + 3.0e-3);
    EXPECT_TRUE(window.has_converged());
}

TEST(ConvergenceWindow, PrintsItsRecentHistoryOldestFirstWithSixDecimals) {
    ConvergenceWindow window(1.0e-10, 5);
    for (int step = 0; step < 7; step++) {
        window.record(32.869717 - 1.65e-5 * step);
    }
    EXPECT_EQ(window.recent_history_text(3), "32.869651, 32.869635, 32.869618");
    EXPECT_EQ(window.recent_history_text(10), window.recent_history_text(7));
    EXPECT_EQ(window.history().size(), 7u);
}

TEST(ConvergenceWindow, RefusesAnEmptyWindowOrANonPositiveTolerance) {
    EXPECT_THROW(ConvergenceWindow(0.0, 3), std::invalid_argument);
    EXPECT_THROW(ConvergenceWindow(1.0e-6, 0), std::invalid_argument);
}

/// What an error reports as numbers and as prose must be the same tail, in the same form.
TEST(ConvergenceWindow, TheRecentHistoryIsTheTailOldestFirstInTheProsesOwnForm) {
    ConvergenceWindow window(1.0e-6, 2);
    for (const Real value : {5.0, 4.0, 3.0, 2.0, 1.0, 0.5}) {
        window.record(value);
    }
    EXPECT_EQ(window.recent_history(3), (std::vector<Real>{2.0, 1.0, 0.5}));
    EXPECT_EQ(window.recent_history(10).size(), 6u);
    EXPECT_EQ(window.recent_history_text(3), "2.000000, 1.000000, 0.500000");
    EXPECT_EQ(ConvergenceWindow::history_text(window.recent_history(3)),
              window.recent_history_text(3));
    EXPECT_EQ(ConvergenceWindow::history_text({}), "");
}
