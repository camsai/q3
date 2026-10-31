#pragma once

/**
 * SelfConsistentFieldFlavor — SCF calculation flavor.
 *
 * Runs a full self-consistent field calculation: builds the iterator,
 * iterates to convergence, computes Hellmann-Feynman forces and stress tensor,
 * writes binary checkpoints when the execution topology owns complete state,
 * and returns its results as results.json's own struct.
 *
 * The Python layer reads results.json and is responsible for any further
 * per-property formatting or output.  No per-property JSON files are written
 * from C++ anymore.
 */

#include <chrono>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "generated/esse/properties/scalar/magnetization_absolute.hpp"
#include "generated/esse/properties/scalar/magnetization_total.hpp"
#include "generated/esse/properties/scalar/pressure.hpp"
#include "generated/esse/properties/workflow/convergence_electronic.hpp"
#include "io/mean_field_export/factory.hpp"
#include "io/mean_field_export/record_and_export.hpp"
#include "io/mean_field_record/from_scf_state.hpp"
#include "io/output/results.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_4_computation/flavors/base.hpp"
#include "layer_4_computation/iterator/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/kpoint_states.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"
#include "layer_5_properties/forces/planewave/init.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/factory.hpp"
#include "layer_5_properties/stress_tensor/planewave/init.hpp"

namespace q3 {

class SelfConsistentFieldFlavor : public Flavor {
  public:
    /**
     * `iterator_factory` carries the entry point's execution policy (see
     * IteratorFactory); empty runs serially on the CPU.
     */
    explicit SelfConsistentFieldFlavor(IteratorFactory iterator_factory = {})
        : iterator_factory_(std::move(iterator_factory)) {}

  protected:
    generated::Results run_calculation(const InputData& input) override {
        std::unique_ptr<PlanewaveKohnShamSCFIterator> owned_iterator =
            create_iterator(input, iterator_factory_);
        PlanewaveKohnShamSCFIterator* iterator = owned_iterator.get();
        auto* planewave_iterator =
            dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(iterator);
        if (!planewave_iterator) {
            throw std::runtime_error(
                "SelfConsistentFieldFlavor: expected a planewave pseudopotential iterator.");
        }
        if (input.start_from_saved_charge_density) {
            offer_saved_charge_density(*iterator, input);
            offer_saved_wavefunctions(*planewave_iterator, input);
        }
        iterator->run();

        const bool converged = iterator->is_converged();
        auto scf_state = iterator->result();

        // Write binary checkpoints.
        planewave_iterator->write_checkpoints(input.output_directory, input.prefix);

        /**
         * ─── Mean-field export (BerkeleyGW, etc.) ──────────────────────────────
         * Format selection lives in the factory ("none" resolves to the Null
         * object), so there is no branching on the format name here -- see
         * io/mean_field_export/init.hpp. BerkeleyGW normally consumes an NSCF
         * run with many empty (conduction) bands rather than the SCF run
         * itself (see non_self_consistent_field.hpp), but the export is wired
         * into both flavors so either can be used directly. Which rank (if
         * any) writes is decided by the execution topology -- see
         * export_and_record_mean_field_under_topology()'s comment. The exported
         * Vxc is the valence one, not the state's own, which includes any core
         * charge -- see valence_exchange_correlation_potential_g_of().
         */
        export_and_record_mean_field_under_topology(
            planewave_iterator->execution_policy(),
            [&scf_state, &input, planewave_iterator] {
                auto record = build_mean_field_record(scf_state);
                record.exchange_correlation_potential_g =
                    planewave_iterator->valence_exchange_correlation_potential_g_of(
                        scf_state.charge_density_real);
                if (input.mean_field_export_kleinman_bylander_projectors)
                    add_kleinman_bylander_projectors_to_mean_field_record(record, scf_state);
                return record;
            },
            input,
            input.output_directory);

        // ─── Results ──────────────────────────────────────────────────────────
        generated::Results results = new_results("self_consistent_field");
        results.runtime_log.converged = converged;
        results.runtime_log.self_consistent_field_iterations = iterator->iteration();
        /**
         * Called on every rank, not just the writing one: the memory reduction
         * inside is collective, and guarding it behind the root would hang the
         * others.
         */
        results.runtime_log.execution = iterator->execution_policy().execution_record();
        /**
         * Carry the timing breakdown into the result, so a measurement does not
         * have to be scraped back out of the log. Counts go alongside the
         * stage times: a stage that got slower because it ran more often is a
         * different finding from one that got slower per call, and the times
         * alone cannot tell them apart.
         */
        {
            const auto& counters = ProfilingCounters::instance();
            nlohmann::json timings = counters.davidson_timing.as_map();
            timings["fft_calls"] = static_cast<double>(counters.fft_call_count.load());
            timings["davidson_iterations"] =
                static_cast<double>(counters.davidson_iteration_count.load());
            timings["hamiltonian_applications"] =
                static_cast<double>(counters.hamiltonian_application_count.load());
            if (!timings.empty()) {
                results.runtime_log.timings_ms = timings;
            }
            /**
             * total_solve_calls is deliberately not emitted: it is declared in
             * DavidsonTimingAccumulator but never incremented anywhere, so it
             * would publish a constant zero that reads as "no solves happened".
             */
        }
        const auto& energies = scf_state.energies();
        set_energy_records(results, energies.total(), energies);
        set_fermi_energy(results, planewave_iterator->quantum_system().fermi_energy());
        record_convergence(results, iterator->scf_history());

        /**
         * Magnetization and the DFT+U occupation traces come from the iterator
         * rather than being re-derived here: both are integrated over state the
         * SCF releases a few lines below, and both are already printed, so a
         * second derivation would be a second definition of the same number.
         */
        record_magnetization(results, planewave_iterator->magnetization());
        record_hubbard_occupations(
            results, planewave_iterator->hubbard_occupation_traces(), scf_state.hubbard_manifolds);

        /**
         * ─── Isolated-system correction (if SCF converged) ─────────────────────
         * Scheme selection lives in the factory ("none" resolves to the Null
         * object), so there is no branching on the scheme name here -- see
         * isolated_system_correction/planewave/init.hpp.
         */
        if (converged) {
            auto correction_calculator =
                create_isolated_system_correction_calculator(input.isolated_system_correction);
            auto correction = correction_calculator->compute(scf_state);
            if (correction.applied) {
                generated::ResultsIsolatedSystemCorrection block;
                block.method = correction.method;
                for (const auto& [label, value_in_ry] : correction.terms_in_ry) {
                    block.terms[label] = energy(label, value_in_ry);
                }
                block.total = energy("total", correction.total_in_ry());
                block.corrected_total_energy =
                    energy("corrected_total_energy", energies.total() + correction.total_in_ry());
                results.properties.isolated_system_correction = block;
                log::info("{} isolated-system correction:", correction.method);
                for (const auto& [label, value] : correction.terms_in_ry) {
                    log::info("     {:<24} = {:.8f} Ry", label, value);
                }
                log::info("     {:<24} = {:.8f} Ry",
                          "corrected total energy",
                          energies.total() + correction.total_in_ry());
            }
        }

        /**
         * The SCF is over: drop its working basis, projector cache and mixing
         * history before the properties allocate their own, so that the run's
         * peak is the SCF's rather than the SCF's with the forces' on top.
         */
        planewave_iterator->release_scf_workspace();

        // ─── Forces (if SCF converged) ────────────────────────────────────────
        if (converged) {
            auto timer_forces_start = std::chrono::high_resolution_clock::now();
            auto [total_forces, forces_contributions] = ForceCalculator::compute_forces(
                scf_state, input.dispersion_correction, input.disable_symmetry);
            auto timer_forces_end = std::chrono::high_resolution_clock::now();

            record_forces(results, total_forces, forces_contributions);

            /**
             * Record the derivative terms Q3 knows it left out, in the result
             * rather than only in the log. A relaxation driven by forces with
             * the Hubbard term missing converges to the wrong geometry without
             * anything in its output saying so; the Python relax and vc-relax
             * workflows read these and stop.
             *
             * Atomic-projector and ortho-atomic force and stress include their complete
             * orbital, overlap-operator, and cell-wide Lowdin-transform derivatives for
             * every dataset family. Keep this metadata guard for future projector types.
             */
            if (!HubbardStressTensorCalculator::supports(scf_state)) {
                record_incomplete_stress(
                    results,
                    "no DFT+U stress term is implemented for Hubbard projector type '" +
                        scf_state.hubbard_projector_type + "'");
            }

            /**
             * ─── Stress tensor ────────────────────────────────────────────────
             * Stress computation is not yet G-vector-distribution-aware.
             * Skip when band_size > 1 to avoid out-of-bounds access on
             * distributed wavefunctions/projectors.
             */
            bool skip_stress = (iterator->execution_policy().band_size() > 1);
            auto timer_stress_start = std::chrono::high_resolution_clock::now();
            if (!skip_stress) {
                auto [stress_total, stress_contributions] =
                    compute_stress(scf_state,
                                   input.nonlocal_stress_derivative_method,
                                   input.dispersion_correction,
                                   input.disable_symmetry);

                record_stress(results, stress_total, stress_contributions);
            } else {
                log::warn(
                    "     Stress tensor skipped (not yet supported with G-vector distribution)");
                record_incomplete_stress(
                    results,
                    "stress is not implemented when wavefunctions are distributed across an "
                    "MPI band group, so the reported zero tensor is not a physical result");
            }
            auto timer_stress_end = std::chrono::high_resolution_clock::now();

            log::debug("     Force/Stress timing:");
            log::debug(
                "       forces:  {:.1f} ms",
                std::chrono::duration<double, std::milli>(timer_forces_end - timer_forces_start)
                    .count());
            log::debug(
                "       stress:  {:.1f} ms",
                std::chrono::duration<double, std::milli>(timer_stress_end - timer_stress_start)
                    .count());
        } else {
            log::warn(
                "SelfConsistentFieldFlavor: skipping forces and stress — SCF did not converge");
        }

        // ─── Eigenvalues ──────────────────────────────────────────────────────
        set_record(results, assemble_global_eigenvalues_at_kpoints(*planewave_iterator));

        return results;
    }

  public:
    /**
     * ─── The SCF's results, shared with the bindings' SCFSolver ──────────────
     *
     * The bindings run the same calculation without this flavor, so what an SCF
     * reports is filled here once, and both report it the same way.
     */

    /**
     * The total energy after each iteration, as ESSE's convergence_electronic
     * record (one row: this run's only ionic step), and the density residual
     * after each, as runtime_log.density_residuals.
     */
    template <typename History>
    static void record_convergence(generated::Results& results, const History& history) {
        if (history.empty()) {
            return;
        }
        std::vector<double> energies_in_ry;
        std::vector<double> density_residuals;
        for (const auto& iteration : history) {
            energies_in_ry.push_back(iteration.total_energy_in_ry);
            density_residuals.push_back(iteration.density_residual);
        }
        set_record(results,
                   generated::ConvergenceElectronic{
                       generated::ConvergenceElectronicNameEnum::CONVERGENCE_ELECTRONIC,
                       generated::ConvergenceElectronicUnitsEnum::RY,
                       {energies_in_ry}});
        results.runtime_log.density_residuals = density_residuals;
    }

    /**
     * Both of the numbers the convergence report prints, ∫(ρ↑−ρ↓) and
     * ∫|ρ↑−ρ↓|, as ESSE's magnetization_total and magnetization_absolute
     * records. The total alone cannot describe an antiferromagnet -- it is zero
     * both for a converged AFM state and for one that collapsed to
     * non-magnetic -- so both are listed, and neither for a run without them
     * (spin-unpolarized, or one that never reached the convergence block):
     * absent rather than a zero that reads as a collapsed moment.
     */
    static void record_magnetization(generated::Results& results,
                                     const std::optional<Magnetization>& magnetization) {
        if (!magnetization) {
            return;
        }
        set_record(results,
                   generated::MagnetizationTotal{
                       generated::MagnetizationTotalNameEnum::MAGNETIZATION_TOTAL,
                       generated::MagnetizationTotalUnitsEnum::UB,
                       magnetization->total_in_bohr_magneton});
        set_record(results,
                   generated::MagnetizationAbsolute{
                       generated::MagnetizationAbsoluteNameEnum::MAGNETIZATION_ABSOLUTE,
                       generated::MagnetizationAbsoluteUnitsEnum::UB,
                       magnetization->absolute_in_bohr_magneton});
    }

    /**
     * Tr[n] per Hubbard manifold, per spin channel: the number of electrons the
     * manifold's projectors see, and the first thing to look at when a DFT+U
     * result looks wrong. Nothing when the run has no Hubbard manifolds.
     * `atom_index` is 0-based, matching the orbital projections' labels.
     */
    template <typename Manifolds>
    static void record_hubbard_occupations(generated::Results& results,
                                           const std::vector<std::vector<Real>>& traces,
                                           const Manifolds& manifolds) {
        std::vector<generated::ResultsPropertiesHubbardOccupationsItem> occupations;
        for (std::size_t manifold_index = 0;
             manifold_index < traces.size() && manifold_index < manifolds.size();
             ++manifold_index) {
            occupations.push_back({manifolds[manifold_index].atom_index,
                                   std::vector<double>(traces[manifold_index].begin(),
                                                       traces[manifold_index].end())});
        }
        if (!occupations.empty()) {
            results.properties.hubbard_occupations = occupations;
        }
    }

    /// ESSE's atomic_forces record, and beside it the share of each mechanism, in Ry/bohr.
    static void record_forces(generated::Results& results,
                              const std::vector<Vector3D>& total_forces,
                              const AtomicForcesContributions& contributions) {
        set_record(results, atomic_forces(total_forces));
        results.properties.force_contributions = std::map<std::string, generated::AtomicForces>{
            {"local", atomic_forces(contributions.local)},
            {"ewald", atomic_forces(contributions.ewald)},
            {"nonlocal", atomic_forces(contributions.nonlocal)},
            {"core_correction", atomic_forces(contributions.core_correction)},
            {"augmentation", atomic_forces(contributions.augmentation)},
            {"dispersion", atomic_forces(contributions.dispersion)},
            {"hubbard", atomic_forces(contributions.hubbard)},
        };
    }

    /**
     * ESSE's stress_tensor and pressure records, and beside them the share of
     * each mechanism, in kbar. Listed only by a run that computed the stress,
     * so absent is never read as a measured zero.
     */
    static void record_stress(generated::Results& results,
                              const StressTensorTotal& total,
                              const StressTensorContributions& contributions) {
        set_record(results, stress_tensor(total.values));
        set_record(results,
                   generated::Pressure{generated::PressureNameEnum::PRESSURE,
                                       generated::PressureUnitsEnum::KBAR,
                                       total.pressure});
        results.properties.stress_contributions = std::map<std::string, generated::StressTensor>{
            {"kinetic", stress_tensor(contributions.kinetic)},
            {"hartree", stress_tensor(contributions.hartree)},
            {"exchange_correlation", stress_tensor(contributions.exchange_correlation)},
            {"ewald", stress_tensor(contributions.ewald)},
            {"local_pseudopotential", stress_tensor(contributions.local)},
            {"nonlocal_pseudopotential", stress_tensor(contributions.nonlocal)},
            {"core_correction", stress_tensor(contributions.core_correction)},
            {"augmentation", stress_tensor(contributions.augmentation)},
            {"hubbard", stress_tensor(contributions.hubbard)},
            {"dispersion", stress_tensor(contributions.dispersion)},
        };
    }

    /**
     * Name a term the stress leaves out, in runtime_log.incomplete, rather than
     * only in the log: a variable-cell relaxation driven by a stress with a
     * term missing converges to the wrong cell without anything in its output
     * saying so, and the Python vc-relax workflow reads this and stops.
     */
    static void record_incomplete_stress(generated::Results& results, const std::string& reason) {
        auto& incomplete = results.runtime_log.incomplete;
        if (!incomplete) {
            incomplete.emplace();
        }
        incomplete->stress = incomplete->stress ? *incomplete->stress + "; " + reason : reason;
    }

  private:
    /**
     * Offer the density a previous run saved under this output directory and
     * prefix (startingpot = 'file'), as each geometry of a relaxation starts
     * from the last one's. A missing or unreadable save starts from
     * overlapping atoms, as pw.x does.
     */
    static void offer_saved_charge_density(PlanewaveKohnShamSCFIterator& iterator,
                                           const InputData& input) {
        const std::string checkpoint_path =
            input.output_directory + "/" + input.prefix + ".save/effective-potential.q3cp";
        KohnShamQuantumSystemSnapshot snapshot;
        Q3PlanewaveKohnShamQuantumSystemCheckpoint checkpoint;
        if (!checkpoint.load(checkpoint_path, snapshot) || snapshot.charge_density_real.empty()) {
            log::info("     No saved density at '{}'; starting from overlapping atoms",
                      checkpoint_path);
            return;
        }
        log::info("     Starting from the density saved at '{}'", checkpoint_path);
        iterator.offer_starting_charge_density(std::move(snapshot.charge_density_real));
    }

    /**
     * Offer the bands the same previous run saved (pw.x keeps the last ionic
     * step's wavefunctions as the next one's guess). Without them every
     * geometry of a relaxation starts the eigensolver from the atomic +
     * random guess -- for a dataset with no atomic orbitals, from random
     * vectors alone. The iterator takes each k-point's file only if it fits.
     */
    static void offer_saved_wavefunctions(PlanewavePseudopotentialKohnShamSCFIterator& iterator,
                                          const InputData& input) {
        iterator.offer_starting_wavefunctions(input.output_directory + "/" + input.prefix +
                                              ".save");
    }

    IteratorFactory iterator_factory_;
};

} // namespace q3
