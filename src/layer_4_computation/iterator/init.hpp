#pragma once

/**
 * SCFIterator — generic self-consistent field iteration control.
 *
 * Owns iteration limits, convergence state, and the template-method run() loop.
 * Representation-specific data and physics live in derived iterators.
 *
 * The quantum system snapshot is provided via SCFIteratorConfig at construction time.
 * run() decomposes into: initialize() → iterate() → report().
 */

#include "io/logger.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/iterator/config.hpp"

#include <chrono>
#include <optional>
#include <vector>

namespace q3 {

class SCFIterator {
  public:
    explicit SCFIterator(SCFIteratorConfig config)
        : crystal_structure_(config.crystal_structure)
        , convergence_threshold_(config.convergence_threshold)
        , maximum_iterations_(config.maximum_iterations)
        , number_of_spin_components_(config.number_of_spin_components) {}

    virtual ~SCFIterator() = default;

    /**
     * The iteration in progress while run() iterates, numbered from 1.
     *
     * After run() it is the number of iterations performed, on either exit:
     * the iteration the run converged on, or the whole budget when it did not.
     */
    [[nodiscard]] int iteration() const {
        return iteration_;
    }
    [[nodiscard]] bool is_converged() const {
        return is_converged_;
    }
    [[nodiscard]] int number_of_spin_components() const {
        return number_of_spin_components_;
    }

    /**
     * Total energy (Ry) after each completed iteration, oldest first.
     *
     * Recorded by the iterate() loop for every flavor, since the sequence is
     * what shows *how* a run converged rather than only where it landed: a
     * monotone descent, an oscillation, and a run that never moved all report
     * the same final energy. The last entry equals total_energy() unless
     * finalize() ran, which recomputes the energy components from the restored
     * density after convergence.
     *
     * Empty before run(); one entry per iteration performed thereafter.
     */
    [[nodiscard]] const std::vector<Real>& total_energy_history_in_ry() const {
        return total_energy_history_in_ry_;
    }

    virtual Real total_energy() const = 0;

    /// Set the SCF convergence threshold (in Ry).
    void set_convergence_threshold(Real threshold) {
        convergence_threshold_ = threshold;
    }
    /// Set the maximum number of SCF iterations.
    void set_maximum_iterations(int maximum_iterations) {
        maximum_iterations_ = maximum_iterations;
    }

    /// Transfer ownership of a Crystal into this iterator.
    void set_owned_crystal(Crystal crystal) {
        owned_crystal_ = std::move(crystal);
        crystal_structure_ = &owned_crystal_.value();
    }

    void run() {
        auto timer_start = std::chrono::steady_clock::now();
        initialize();
        auto timer_after_initialization = std::chrono::steady_clock::now();
        iterate();
        auto timer_after_iteration = std::chrono::steady_clock::now();
        report();
        auto timer_end = std::chrono::steady_clock::now();

        auto duration_milliseconds = [](auto start, auto end) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        };
        log::info("     Timing: init={}ms  iterate={}ms  report={}ms  total={}ms",
                  duration_milliseconds(timer_start, timer_after_initialization),
                  duration_milliseconds(timer_after_initialization, timer_after_iteration),
                  duration_milliseconds(timer_after_iteration, timer_end),
                  duration_milliseconds(timer_start, timer_end));
    }

  protected:
    // Configuration — set from SCFIteratorConfig at construction.
    Crystal* crystal_structure_;
    Real convergence_threshold_;
    int maximum_iterations_;

    int iteration_{0};
    bool is_converged_{false};

  private:
    int number_of_spin_components_;
    /// Appended by iterate(); read through total_energy_history_in_ry().
    std::vector<Real> total_energy_history_in_ry_;
    /**
     * Optional owned Crystal — used when the factory constructs the iterator
     * from InputData (the iterator owns the crystal rather than borrowing it).
     */
    std::optional<Crystal> owned_crystal_;

  protected:
    /// Base implementation prints the SCF banner. Children call SCFIterator::initialize() last.
    virtual void initialize() {
        iteration_ = 0;
        is_converged_ = false;
        /**
         * Run state, like the two above: a second run() reports its own
         * trajectory rather than one appended to the previous run's.
         */
        total_energy_history_in_ry_.clear();

        log::section("Self-consistent Calculation:");
    }

    /// Main iteration loop.
    virtual void iterate() {
        /**
         * iteration_ is assigned from the loop counter rather than being it, so
         * it never steps past the last iteration performed. Were it the counter,
         * a run that exhausted its budget would leave it one past that budget,
         * and the count read after run() -- by the SCF state, the checkpoint
         * header and the results file -- would name an iteration that never ran.
         */
        for (int iteration_number = 1; iteration_number <= maximum_iterations_;
             ++iteration_number) {
            iteration_ = iteration_number;
            perform_iteration();
            total_energy_history_in_ry_.push_back(total_energy());

            if (is_iteration_converged()) {
                is_converged_ = true;
                finalize();
                return;
            }

            report_iteration();
        }
    }

    /// Report convergence or non-convergence after iterate() finishes.
    virtual void report() {
        if (!is_converged_) {
            report_non_convergence();
        } else {
            report_convergence();
        }
    }

    virtual void perform_iteration() = 0;
    virtual bool is_iteration_converged() const = 0;
    virtual void finalize() = 0;
    virtual void report_iteration() = 0;
    virtual void report_non_convergence() = 0;
    virtual void report_convergence() = 0;

  public:
    /**
     * Write binary checkpoint files (charge density, wavefunctions) to the given directory.
     *
     * Each level of the hierarchy overrides this, calling the parent first,
     * then adding its own checkpoint data. Default no-op.
     */
    virtual void write_checkpoints(const std::string& output_directory,
                                   const std::string& prefix) const {
        (void)output_directory;
        (void)prefix;
    }
};

} // namespace q3
