#pragma once

/**
 * SelfConsistentFieldFlavor — SCF calculation flavor.
 *
 * Runs a full self-consistent field calculation: builds the iterator,
 * iterates to convergence, computes Hellmann-Feynman forces and stress tensor,
 * writes binary checkpoints when the execution topology owns complete state,
 * and returns a FlavorResult with all quantities as raw types.
 *
 * The Python layer reads results.json and is responsible for any further
 * per-property formatting or output.  No per-property JSON files are written
 * from C++ anymore.
 */

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "io/mean_field_export/factory.hpp"
#include "io/mean_field_export/record_and_export.hpp"
#include "io/mean_field_record/from_scf_state.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_4_computation/flavors/base.hpp"
#include "layer_4_computation/flavors/result.hpp"
#include "layer_4_computation/iterator/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"
#include "layer_5_properties/forces/planewave/init.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/factory.hpp"
#include "layer_5_properties/stress_tensor/planewave/init.hpp"

namespace q3 {

class SelfConsistentFieldFlavor : public Flavor {
  public:
    SelfConsistentFieldFlavor(std::unique_ptr<PlanewaveKohnShamSCFIterator> iterator = nullptr)
        : iterator_(std::move(iterator)) {}

  protected:
    FlavorResult run_calculation(const InputData& input) override {
        /**
         * When the iterator was pre-set via the constructor, keep it alive in
         * iterator_ for the full lifetime of this flavor. Only take a temporary
         * unique_ptr when we create the iterator on demand (scratch path).
         */
        std::unique_ptr<PlanewaveKohnShamSCFIterator> temp_iterator;
        PlanewaveKohnShamSCFIterator* iterator = nullptr;
        if (iterator_) {
            iterator = iterator_.get(); // borrow — ownership stays in iterator_
        } else {
            temp_iterator = create_iterator(input);
            iterator = temp_iterator.get();
        }
        if (input.start_from_saved_charge_density) {
            offer_saved_charge_density(*iterator, input);
        }
        iterator->run();

        const bool converged = iterator->is_converged();
        auto scf_state = iterator->result();

        auto* planewave_iterator =
            dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(iterator);
        if (!planewave_iterator) {
            throw std::runtime_error(
                "SelfConsistentFieldFlavor: expected a planewave pseudopotential iterator.");
        }

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
         * export_and_record_mean_field_under_topology()'s comment.
         */
        export_and_record_mean_field_under_topology(
            planewave_iterator->execution_policy(),
            [&scf_state] { return build_mean_field_record(scf_state); },
            input,
            input.output_directory);

        // ─── Populate FlavorResult ────────────────────────────────────────────
        FlavorResult flavor_result;
        flavor_result.flavor = "self_consistent_field";
        flavor_result.converged = converged;
        flavor_result.number_of_self_consistent_field_iterations = iterator->iteration();
        flavor_result.number_of_processes = iterator->execution_policy().number_of_processes();
        flavor_result.number_of_kpoint_pools = iterator->execution_policy().number_of_pools();
        flavor_result.band_group_size =
            flavor_result.number_of_processes / flavor_result.number_of_kpoint_pools;
        flavor_result.reciprocal_space_vectors_are_distributed =
            iterator->execution_policy().band_size() > 1;
        /**
         * Called on every rank, not just the writing one: reduce_scalar_max is
         * collective, and guarding it behind the root would hang the others.
         */
        Real peak_resident_megabytes =
            static_cast<Real>(ProfilingCounters::instance().peak_resident_bytes_now()) / 1048576.0;
        iterator->execution_policy().reduce_scalar_max(peak_resident_megabytes);
        flavor_result.peak_resident_memory_bytes =
            static_cast<std::uint64_t>(peak_resident_megabytes * 1048576.0);
        /**
         * Carry the timing breakdown into the result, so a measurement does not
         * have to be scraped back out of the log. Counts go alongside the
         * stage times: a stage that got slower because it ran more often is a
         * different finding from one that got slower per call, and the times
         * alone cannot tell them apart.
         */
        {
            const auto& counters = ProfilingCounters::instance();
            flavor_result.timings = counters.davidson_timing.as_map();
            flavor_result.timings["fft_calls"] =
                static_cast<double>(counters.fft_call_count.load());
            flavor_result.timings["davidson_iterations"] =
                static_cast<double>(counters.davidson_iteration_count.load());
            flavor_result.timings["hamiltonian_applications"] =
                static_cast<double>(counters.hamiltonian_application_count.load());
            /**
             * total_solve_calls is deliberately not emitted: it is declared in
             * DavidsonTimingAccumulator but never incremented anywhere, so it
             * would publish a constant zero that reads as "no solves happened".
             */
        }
        for (const auto& record : iterator->scf_history())
            flavor_result.scf_iterations.push_back({record.iteration,
                                                    record.total_energy_in_ry,
                                                    record.energy_change_in_ry,
                                                    record.density_residual});

        const auto& energies = scf_state.energies();
        flavor_result.total_energy_in_ry = energies.total();
        flavor_result.band_energy_in_ry = energies.band_energy;
        flavor_result.hartree_energy_in_ry = energies.hartree_energy;
        flavor_result.exchange_correlation_energy_in_ry = energies.exchange_correlation_energy;
        flavor_result.ewald_energy_in_ry = energies.ewald_energy;
        flavor_result.double_counting_correction_in_ry = energies.double_counting_correction;
        flavor_result.smearing_entropy_contribution_in_ry = energies.smearing_entropy_contribution;
        flavor_result.hubbard_energy_in_ry = energies.hubbard_energy_contribution();
        flavor_result.dispersion_energy_in_ry = energies.dispersion_energy();
        flavor_result.one_center_energy_in_ry = energies.one_center_energy_contribution();

        /**
         * Magnetization and the DFT+U occupation traces come from the iterator
         * rather than being re-derived here: both are integrated over state the
         * SCF releases a few lines below, and both are already printed, so a
         * second derivation would be a second definition of the same number.
         */
        flavor_result.magnetization = planewave_iterator->magnetization();
        const auto& hubbard_occupation_traces = planewave_iterator->hubbard_occupation_traces();
        for (std::size_t manifold_index = 0; manifold_index < hubbard_occupation_traces.size() &&
                                             manifold_index < scf_state.hubbard_manifolds.size();
             ++manifold_index) {
            flavor_result.hubbard_occupations.push_back(
                {scf_state.hubbard_manifolds[manifold_index].atom_index,
                 hubbard_occupation_traces[manifold_index]});
        }

        flavor_result.fermi_energy_in_ry = planewave_iterator->quantum_system().fermi_energy();

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
            flavor_result.isolated_system_correction_applied = correction.applied;
            flavor_result.isolated_system_correction_method = correction.method;
            flavor_result.isolated_system_correction_terms_in_ry = correction.terms_in_ry;
            if (correction.applied) {
                log::info("{} isolated-system correction:", correction.method);
                for (const auto& [label, value] : correction.terms_in_ry) {
                    log::info("     {:<24} = {:.8f} Ry", label, value);
                }
                log::info("     {:<24} = {:.8f} Ry",
                          "corrected total energy",
                          flavor_result.total_energy_in_ry + correction.total_in_ry());
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

            // Copy raw values into FlavorResult.
            flavor_result.forces_in_ry_per_bohr = total_forces;
            flavor_result.add_force_contribution("local", forces_contributions.local);
            flavor_result.add_force_contribution("ewald", forces_contributions.ewald);
            flavor_result.add_force_contribution("nonlocal", forces_contributions.nonlocal);
            flavor_result.add_force_contribution("core_correction",
                                                 forces_contributions.core_correction);
            flavor_result.add_force_contribution("augmentation", forces_contributions.augmentation);
            flavor_result.add_force_contribution("dispersion", forces_contributions.dispersion);
            flavor_result.add_force_contribution("hubbard", forces_contributions.hubbard);

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
                flavor_result.stress_is_incomplete = true;
                flavor_result.stress_incomplete_reason =
                    "no DFT+U stress term is implemented for Hubbard projector type '" +
                    scf_state.hubbard_projector_type + "'";
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

                // Copy raw values into FlavorResult.
                for (int row_index = 0; row_index < 3; row_index++) {
                    for (int column_index = 0; column_index < 3; column_index++) {
                        flavor_result.stress_in_kbar[row_index][column_index] =
                            stress_total.values[row_index][column_index];
                    }
                }
                flavor_result.pressure_in_kbar = stress_total.pressure;
                flavor_result.add_stress_contribution("kinetic", stress_contributions.kinetic);
                flavor_result.add_stress_contribution("hartree", stress_contributions.hartree);
                flavor_result.add_stress_contribution("exchange_correlation",
                                                      stress_contributions.exchange_correlation);
                flavor_result.add_stress_contribution("ewald", stress_contributions.ewald);
                flavor_result.add_stress_contribution("local_pseudopotential",
                                                      stress_contributions.local);
                flavor_result.add_stress_contribution("nonlocal_pseudopotential",
                                                      stress_contributions.nonlocal);
                flavor_result.add_stress_contribution("core_correction",
                                                      stress_contributions.core_correction);
                flavor_result.add_stress_contribution("augmentation",
                                                      stress_contributions.augmentation);
                flavor_result.add_stress_contribution("hubbard", stress_contributions.hubbard);
                flavor_result.add_stress_contribution("dispersion",
                                                      stress_contributions.dispersion);
            } else {
                log::warn(
                    "     Stress tensor skipped (not yet supported with G-vector distribution)");
                flavor_result.stress_is_incomplete = true;
                if (!flavor_result.stress_incomplete_reason.empty()) {
                    flavor_result.stress_incomplete_reason += "; ";
                }
                flavor_result.stress_incomplete_reason +=
                    "stress is not implemented when wavefunctions are distributed across an "
                    "MPI band group, so the reported zero tensor is not a physical result";
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
        const auto& eigenvalue_matrix = planewave_iterator->eigenvalues();
        int local_number_of_kpoint_states = eigenvalue_matrix.rows();
        int number_of_bands = eigenvalue_matrix.cols();
        /**
         * kpoint_grid() holds one entry per PHYSICAL k-point; eigenvalue_matrix /
         * occupation() are indexed by k-point STATE (physical kpoint * spin, see
         * build_global_kpoint_grid_from_input / accumulator.hpp for the same
         * convention) -- for a spin-polarized run number_of_kpoints above is the
         * state count, so kpoint_grid() lookups need the physical index instead.
         */
        int local_number_of_physical_kpoints =
            planewave_iterator->kpoint_grid().number_of_kpoints();
        int number_of_spin_components =
            planewave_iterator->quantum_system().number_of_spin_components;
        int number_of_values_per_state = 2 * number_of_bands + 4;
        std::vector<Real> local_state_data(static_cast<std::size_t>(local_number_of_kpoint_states) *
                                           number_of_values_per_state);
        for (int kpoint_index_state = 0; kpoint_index_state < local_number_of_kpoint_states;
             ++kpoint_index_state) {
            int local_physical_kpoint_index = kpoint_index_state % local_number_of_physical_kpoints;
            std::size_t state_offset =
                static_cast<std::size_t>(kpoint_index_state) * number_of_values_per_state;
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                local_state_data[state_offset + band_index] =
                    eigenvalue_matrix(kpoint_index_state, band_index);
                local_state_data[state_offset + number_of_bands + band_index] =
                    planewave_iterator->occupation(kpoint_index_state, band_index);
            }
            const auto& kpoint = planewave_iterator->kpoint_grid()[local_physical_kpoint_index];
            local_state_data[state_offset + 2 * number_of_bands] = kpoint.weight;
            local_state_data[state_offset + 2 * number_of_bands + 1] = kpoint.coordinates.x;
            local_state_data[state_offset + 2 * number_of_bands + 2] = kpoint.coordinates.y;
            local_state_data[state_offset + 2 * number_of_bands + 3] = kpoint.coordinates.z;
        }

        std::vector<Real> global_state_data =
            planewave_iterator->execution_policy().assemble_global_kpoint_state_data(
                local_state_data,
                local_number_of_physical_kpoints,
                number_of_spin_components,
                number_of_values_per_state);
        int global_number_of_kpoint_states =
            static_cast<int>(global_state_data.size()) / number_of_values_per_state;
        flavor_result.eigenvalues_in_ry.resize(global_number_of_kpoint_states);
        flavor_result.occupations.resize(global_number_of_kpoint_states);
        flavor_result.kpoint_weights.resize(global_number_of_kpoint_states);
        flavor_result.kpoint_coordinates.resize(global_number_of_kpoint_states);
        for (int kpoint_index_state = 0; kpoint_index_state < global_number_of_kpoint_states;
             ++kpoint_index_state) {
            std::size_t state_offset =
                static_cast<std::size_t>(kpoint_index_state) * number_of_values_per_state;
            flavor_result.eigenvalues_in_ry[kpoint_index_state].resize(1);
            flavor_result.eigenvalues_in_ry[kpoint_index_state][0].assign(
                global_state_data.begin() + state_offset,
                global_state_data.begin() + state_offset + number_of_bands);
            flavor_result.occupations[kpoint_index_state].resize(1);
            flavor_result.occupations[kpoint_index_state][0].assign(
                global_state_data.begin() + state_offset + number_of_bands,
                global_state_data.begin() + state_offset + 2 * number_of_bands);
            flavor_result.kpoint_weights[kpoint_index_state] =
                global_state_data[state_offset + 2 * number_of_bands];
            flavor_result.kpoint_coordinates[kpoint_index_state] = {
                global_state_data[state_offset + 2 * number_of_bands + 1],
                global_state_data[state_offset + 2 * number_of_bands + 2],
                global_state_data[state_offset + 2 * number_of_bands + 3],
            };
        }

        return flavor_result;
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

    std::unique_ptr<PlanewaveKohnShamSCFIterator> iterator_;
};

} // namespace q3
