#include <gtest/gtest.h>

#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_4_computation/iterator/init.hpp"

using namespace q3;

/**
 * ── Per-iteration energy history ─────────────────────────────────────
 *
 * The iterate() loop records the total energy after every iteration. The
 * sequence is what distinguishes a run that converged from one that never
 * moved and one that oscillated -- all three can end on the same number, and
 * a restart-verification flavor has nothing to check without it.
 *
 * Driven through a stub iterator rather than a real SCF run: the behavior
 * under test belongs to the base loop, and a real run would test the physics
 * instead.
 */

namespace {

/// An SCF iterator that reports a scripted sequence of energies.
class ScriptedEnergyIterator : public SCFIterator {
  public:
    ScriptedEnergyIterator(SCFIteratorConfig config, std::vector<Real> scripted_energies_in_ry)
        : SCFIterator(config), scripted_energies_in_ry_(std::move(scripted_energies_in_ry)) {}

    Real total_energy() const override {
        return current_energy_in_ry_;
    }

    void perform_iteration() override {
        // iteration_ is 1-based; past the end of the script the energy holds.
        auto scripted_index = static_cast<size_t>(iteration_ - 1);
        if (scripted_index < scripted_energies_in_ry_.size()) {
            current_energy_in_ry_ = scripted_energies_in_ry_[scripted_index];
        }
    }

    bool is_iteration_converged() const override {
        return converge_at_iteration_ > 0 && iteration_ >= converge_at_iteration_;
    }

    void converge_at(int iteration) {
        converge_at_iteration_ = iteration;
    }

    void finalize() override {}
    void report_iteration() override {}
    void report_non_convergence() override {}
    void report_convergence() override {}

  private:
    std::vector<Real> scripted_energies_in_ry_;
    Real current_energy_in_ry_ = 0.0;
    int converge_at_iteration_ = 0;
};

SCFIteratorConfig make_config(int maximum_iterations) {
    SCFIteratorConfig config;
    config.maximum_iterations = maximum_iterations;
    return config;
}

} // namespace

TEST(SCFIteratorEnergyHistory, IsEmptyBeforeRunning) {
    ScriptedEnergyIterator iterator(make_config(3), {-1.0, -2.0, -3.0});

    EXPECT_TRUE(iterator.total_energy_history_in_ry().empty());
}

TEST(SCFIteratorEnergyHistory, RecordsOneEnergyPerIteration) {
    ScriptedEnergyIterator iterator(make_config(3), {-15.1, -15.7, -15.83});

    iterator.run();

    const auto& history = iterator.total_energy_history_in_ry();
    ASSERT_EQ(history.size(), 3u);
    EXPECT_DOUBLE_EQ(history[0], -15.1);
    EXPECT_DOUBLE_EQ(history[1], -15.7);
    EXPECT_DOUBLE_EQ(history[2], -15.83);
}

TEST(SCFIteratorEnergyHistory, RecordsTheEnergyOfTheConvergingIteration) {
    /**
     * The loop exits on convergence before report_iteration(), so the energy
     * has to be recorded first -- otherwise the trajectory is missing exactly
     * the iteration that mattered.
     */
    ScriptedEnergyIterator iterator(make_config(10), {-15.1, -15.7, -15.83});
    iterator.converge_at(3);

    iterator.run();

    ASSERT_EQ(iterator.total_energy_history_in_ry().size(), 3u);
    EXPECT_DOUBLE_EQ(iterator.total_energy_history_in_ry().back(), -15.83);
    EXPECT_TRUE(iterator.is_converged());
}

TEST(SCFIteratorEnergyHistory, AFlatTrajectoryIsDistinguishableFromADescendingOne) {
    /**
     * The restart-verification case: an already-converged density iterated
     * again produces the same energy every time. That is a different history
     * from a run that descended to the same final value, even though
     * total_energy() cannot tell them apart.
     */
    ScriptedEnergyIterator restarted(make_config(3), {-15.83, -15.83, -15.83});
    ScriptedEnergyIterator descending(make_config(3), {-15.1, -15.7, -15.83});

    restarted.run();
    descending.run();

    EXPECT_DOUBLE_EQ(restarted.total_energy(), descending.total_energy());
    EXPECT_NE(restarted.total_energy_history_in_ry(), descending.total_energy_history_in_ry());
    EXPECT_DOUBLE_EQ(restarted.total_energy_history_in_ry().front(),
                     restarted.total_energy_history_in_ry().back());
}
