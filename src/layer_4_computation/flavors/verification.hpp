#pragma once

/**
 * VerificationFlavor — restart-stability check of a converged charge density.
 *
 * Answers one question a third party cannot otherwise ask: is the density
 * shipped with a published calculation actually the converged one? A total
 * energy in a results file is an assertion. A density is checkable, because a
 * converged density is a fixed point of the SCF map -- restart from it, iterate
 * again, and the energy must not move. A density that was edited, taken from a
 * different structure, or captured before convergence moves immediately, and by
 * much more than the SCF threshold.
 *
 * Workflow:
 *   1. Refuse an iteration budget that could not answer the question.
 *   2. Load the charge density from <prefix>.save/effective-potential.q3cp.
 *   3. Construct an SCF iterator over the same input, with the convergence
 *      exit disabled so the run always spends its whole budget.
 *   4. Inject the density via load_charge_density(), so iteration 1 builds its
 *      Hamiltonian from the submitted density instead of a superposition of
 *      atomic densities.
 *   5. Return the per-iteration energies as scf_energies_in_ry.
 *
 * Two decisions distinguish this from NSCF, which loads the same file:
 *
 * **The charge density is loaded, not V_eff.** Injecting the effective
 * potential freezes it, which is exactly right for a band structure and
 * useless here: a frozen potential never re-derives anything from the density,
 * so a tampered density would produce a perfectly stable energy. Loading the
 * density puts the submitted quantity back into the loop that has to reproduce
 * it.
 *
 * **The convergence exit is disabled**, by setting the threshold to zero --
 * the residual check is `<`, so no residual, not even an exactly-zero one,
 * satisfies it. The run therefore always performs the full budget and always
 * reports the same number of energies. Stopping early on convergence would
 * make the length of the trajectory depend on the answer, and one energy is
 * not evidence of stability.
 *
 * The flavor reports the trajectory and does not judge it. What counts as
 * stable is a protocol parameter, versioned by the consumer that set it, and
 * an engine that hard-coded a tolerance would silently outvote it.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"
#include "layer_4_computation/flavors/base.hpp"
#include "layer_4_computation/flavors/result.hpp"
#include "layer_4_computation/iterator/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"

namespace q3 {

class VerificationFlavor : public Flavor {
  protected:
    FlavorResult run_calculation(const InputData& input) override {
        /**
         * ─── Step 1: Check the input before touching the disk ────────────────
         * The budget is whatever the input asks for, not a number chosen here.
         * A few iterations are enough to show drift and a verifier paying for
         * the compute is the one entitled to decide how many; a cap applied in
         * the engine would quietly overrule an input that meant what it said.
         * What is refused is a budget that could not answer the question.
         */
        const int iteration_budget = input.maximum_iterations;
        if (iteration_budget < 2) {
            throw std::runtime_error(
                "VerificationFlavor: 'max_iterations' is " + std::to_string(iteration_budget) +
                "; a restart needs at least 2 iterations, since a single energy cannot show "
                "whether the density moved.");
        }

        const std::string save_directory = input.output_directory + "/" + input.prefix + ".save";
        const std::string checkpoint_path = save_directory + "/effective-potential.q3cp";

        // ─── Step 2: Load the submitted charge density ───────────────────────
        KohnShamQuantumSystemSnapshot snapshot;
        Q3PlanewaveKohnShamQuantumSystemCheckpoint checkpoint;

        if (!checkpoint.load(checkpoint_path, snapshot)) {
            throw std::runtime_error("VerificationFlavor: failed to load checkpoint at '" +
                                     checkpoint_path +
                                     "'. Place the submitted checkpoint there before verifying.");
        }
        if (snapshot.charge_density_real.empty()) {
            throw std::runtime_error("VerificationFlavor: checkpoint at '" + checkpoint_path +
                                     "' contains no charge density; there is nothing to restart "
                                     "from.");
        }

        log::info("Verification: restarting from '{}' for {} iterations",
                  checkpoint_path,
                  iteration_budget);

        /**
         * ─── Step 3: Build the iterator with the exit disabled ───────────────
         * A threshold of zero can never be met -- is_iteration_converged() is
         * `density_residual_ < convergence_threshold_` -- so the loop runs the
         * full budget whatever the density turns out to be.
         */
        InputData verification_input = input;
        verification_input.maximum_iterations = iteration_budget;
        verification_input.convergence_threshold = 0.0;

        auto iterator = create_iterator(verification_input);

        auto* planewave_iterator =
            dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(iterator.get());
        if (!planewave_iterator) {
            throw std::runtime_error(
                "VerificationFlavor: expected a planewave pseudopotential iterator.");
        }

        // ─── Step 4: Inject the density and run ──────────────────────────────
        planewave_iterator->load_charge_density(snapshot.charge_density_real);
        iterator->run();

        // ─── Step 5: Report the trajectory ───────────────────────────────────
        FlavorResult flavor_result;
        flavor_result.flavor = "verification";
        /**
         * The run deliberately never satisfies its (zeroed) convergence test,
         * so the iterator's own verdict says nothing here. What the flavor
         * produced is a complete trajectory, and that is what "converged"
         * reports -- the caller decides whether the trajectory is flat enough.
         */
        flavor_result.converged =
            static_cast<int>(iterator->total_energy_history_in_ry().size()) == iteration_budget;
        flavor_result.scf_energies_in_ry = iterator->total_energy_history_in_ry();
        flavor_result.total_energy_in_ry = iterator->total_energy();
        flavor_result.fermi_energy_in_ry = planewave_iterator->quantum_system().fermi_energy();

        log::info("Verification: {} iterations, energy drift {:.3e} Ry",
                  flavor_result.scf_energies_in_ry.size(),
                  energy_drift_in_ry(flavor_result.scf_energies_in_ry));

        return flavor_result;
    }

  private:
    /// Largest departure from the first energy, which is the submitted one.
    [[nodiscard]] static Real energy_drift_in_ry(const std::vector<Real>& energies_in_ry) {
        Real largest_drift = 0.0;
        for (const auto& energy_in_ry : energies_in_ry) {
            largest_drift =
                std::max(largest_drift, std::abs(energy_in_ry - energies_in_ry.front()));
        }
        return largest_drift;
    }
};

} // namespace q3
