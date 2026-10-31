#pragma once

/**
 * NonSelfConsistentFieldFlavor — NSCF calculation flavor.
 *
 * Runs a single-pass diagonalization at a user-specified list of k-points.
 * Requires a v2 checkpoint from a prior SCF calculation to load V_eff.
 *
 * Workflow:
 *   1. Load effective potential V_eff from <prefix>.save/effective-potential.q3cp.
 *   2. Construct an iterator with the explicit kpoints_list from InputData.
 *   3. Inject V_eff via load_effective_potential() (bypasses SCF reconstruction).
 *   4. Set maximum_iterations = 1 so the iterator performs a single diagonalization.
 *   5. Run the iterator.
 *   6. Return eigenvalues + occupations as results.json's own struct.
 *
 * The results contain the eigenvalues (and, since the shared
 * per-iteration flow always calls update_occupations(), occupations too) at
 * all requested k-points, but no total energy (which is meaningless for a
 * non-self-consistent calculation). Occupations reflect a Fermi bisection
 * over *this run's* k-point set, not the original SCF's -- reliable for a
 * proper dense k-grid, not for a non-uniform set like a band path.
 */

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/interpolation/record_and_export.hpp"
#include "io/logger.hpp"
#include "io/mean_field_export/factory.hpp"
#include "io/mean_field_export/record_and_export.hpp"
#include "io/mean_field_record/from_scf_state.hpp"
#include "io/output/results.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/factory.hpp"
#include "layer_4_computation/flavors/base.hpp"
#include "layer_4_computation/interpolator/capture.hpp"
#include "layer_4_computation/iterator/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/kpoint_states.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"
#include "layer_5_properties/projections/planewave/init.hpp"

namespace q3 {

/**
 * The Davidson working basis a non-self-consistent pass runs with.
 *
 * A single pass keeps what the solver returns, and below four times the band
 * count it can return a spectrum with an eigenvalue missing -- see the
 * constant. A default below that is raised; a factor the deck named is kept,
 * since it is a memory choice its author made, and the caller warns.
 */
inline int non_self_consistent_field_davidson_basis_expansion_factor(int factor, bool set_by_deck) {
    if (set_by_deck) {
        return factor;
    }
    return std::max(factor,
                    constants::non_self_consistent_field_minimum_davidson_basis_expansion_factor);
}

class NonSelfConsistentFieldFlavor : public Flavor {
  public:
    /**
     * `iterator_factory` carries the entry point's execution policy (see
     * IteratorFactory); empty runs serially on the CPU.
     */
    explicit NonSelfConsistentFieldFlavor(IteratorFactory iterator_factory = {})
        : iterator_factory_(std::move(iterator_factory)) {}

  protected:
    generated::Results run_calculation(const InputData& input) override {
        if (input.kpoints_list.empty()) {
            throw std::runtime_error(
                "NonSelfConsistentFieldFlavor: 'kpoints_list' is required in the input "
                "for calculation type 'non_self_consistent_field'.");
        }

        const std::string save_directory = input.output_directory + "/" + input.prefix + ".save";
        const std::string effective_potential_checkpoint_path =
            save_directory + "/effective-potential.q3cp";

        // ─── Step 1: Load V_eff from checkpoint ──────────────────────────────
        KohnShamQuantumSystemSnapshot effective_potential_snapshot;
        Q3PlanewaveKohnShamQuantumSystemCheckpoint effective_potential_checkpoint;

        bool loaded_successfully = effective_potential_checkpoint.load(
            effective_potential_checkpoint_path, effective_potential_snapshot);

        if (!loaded_successfully) {
            throw std::runtime_error(
                "NonSelfConsistentFieldFlavor: failed to load effective-potential checkpoint at '" +
                effective_potential_checkpoint_path +
                "'. Run a self_consistent_field calculation first to generate it.");
        }

        if (effective_potential_snapshot.effective_potential_real.empty()) {
            throw std::runtime_error(
                "NonSelfConsistentFieldFlavor: effective-potential checkpoint at '" +
                effective_potential_checkpoint_path +
                "' does not contain V_eff (may be a v1 checkpoint — re-run SCF to generate v2).");
        }

        log::info("     NSCF: loaded effective potential from '{}'",
                  effective_potential_checkpoint_path);
        log::info("     NSCF: diagonalizing at {} explicit k-points",
                  static_cast<int>(input.kpoints_list.size()));

        /**
         * ─── Step 2: Build iterator with explicit k-points ────────────────────
         * Set maximum_iterations = 1 so the base iterate() loop runs exactly once
         * (performs initialization + one diagonalization pass).
         */
        InputData nscf_input = input;
        nscf_input.maximum_iterations = 1;
        /**
         * Use the same convergence threshold as the SCF run (irrelevant for NSCF
         * since we do only one iteration, but must be set to avoid division by zero).
         */
        nscf_input.convergence_threshold = 1.0e-16;
        const int requested_factor = nscf_input.davidson_basis_expansion_factor;
        nscf_input.davidson_basis_expansion_factor =
            non_self_consistent_field_davidson_basis_expansion_factor(
                requested_factor, nscf_input.davidson_basis_expansion_factor_set_by_deck);
        if (nscf_input.davidson_basis_expansion_factor != requested_factor) {
            log::info(
                "     NSCF: Davidson working basis {} times the band count, not the default {}: "
                "a single pass below that can miss eigenvalues",
                nscf_input.davidson_basis_expansion_factor,
                requested_factor);
        } else if (requested_factor <
                   constants::non_self_consistent_field_minimum_davidson_basis_expansion_factor) {
            log::warn("NSCF: method.scf.davidson.basis_expansion_factor is {}, below {}; a single "
                      "pass at that working basis can miss eigenvalues while every residual "
                      "test passes",
                      requested_factor,
                      constants::non_self_consistent_field_minimum_davidson_basis_expansion_factor);
        }

        auto iterator = create_iterator(nscf_input, iterator_factory_);

        auto* planewave_iterator =
            dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(iterator.get());
        if (!planewave_iterator) {
            throw std::runtime_error(
                "NonSelfConsistentFieldFlavor: expected a planewave pseudopotential iterator.");
        }

        /**
         * The potential below is loaded whole, onto every rank, as the SCF
         * saved it. That fits a rank holding whole k-points -- one rank per
         * pool, the topology the SCF itself needs to write the checkpoint at
         * all -- and nothing else: ranks sharing a pool split the density grid
         * between them, and nothing scatters a whole grid onto that split. Asked
         * after the iterator exists, because building it is where a pool count
         * above the k-point count is clamped back into shared pools.
         */
        if (!planewave_iterator->execution_policy().each_process_owns_whole_kpoints()) {
            throw std::runtime_error(
                "NonSelfConsistentFieldFlavor: ranks share a k-point pool here, and a "
                "non-self-consistent run needs one rank per pool: it loads the SCF's "
                "potential whole onto every rank, which ranks splitting the density grid "
                "cannot take. Run it with --pools equal to the rank count (at most the number "
                "of k-points listed), or with fewer ranks.");
        }

        /**
         * ─── Step 3: Inject preloaded V_eff ──────────────────────────────────
         * load_effective_potential() sets has_preloaded_effective_potential_ = true,
         * which prevents update_effective_potential() from overwriting it during
         * initialize_single_particle_states(). Instead, only the G-space FFT is
         * done to populate total_potential_g_ (needed by prepare_hamiltonian()).
         */
        planewave_iterator->load_effective_potential(
            effective_potential_snapshot.effective_potential_real);

        /**
         * ─── Step 3a: Hand over the PAW on-site occupations ───────────────────
         * A PAW dataset's D carries a one-centre term built from ρ_ij, which the
         * SCF refreshes each time it constructs a density. This run
         * diagonalizes before it constructs any, so without the converged ρ_ij
         * that term is empty and every eigenvalue is wrong -- copper's 3s by
         * 103 eV. A checkpoint written before the occupations were stored cannot
         * supply them, and a result built without them is refused rather than
         * returned.
         */
        if (planewave_iterator->requires_one_center_occupations()) {
            if (effective_potential_snapshot.one_center_occupations.empty()) {
                throw std::runtime_error(
                    "NonSelfConsistentFieldFlavor: checkpoint at '" +
                    effective_potential_checkpoint_path +
                    "' carries no PAW on-site occupations, which the one-centre term of D is "
                    "built from (it predates checkpoint version 3). Re-run the "
                    "self_consistent_field calculation to regenerate it.");
            }
            planewave_iterator->load_one_center_occupations(
                effective_potential_snapshot.one_center_occupations);
        }

        /**
         * ─── Step 3b: Set tight diagonalization tolerance ─────────────────────
         * The SCF iterator uses a loose initial tolerance (1e-2 Ry) for iteration 1,
         * which is appropriate for SCF where the potential evolves anyway. For NSCF
         * the potential is frozen and we run only one pass, so we must start with a
         * tight tolerance to ensure Davidson converges all bands to the true lowest
         * eigenvalues. Without this, high-index conduction bands at special k-points
         * (e.g. the 10th band at K in Si: 14.5 eV) can converge to spurious excited
         * states instead of the correct ground-state eigenvectors.
         */
        planewave_iterator->set_diagonalizer_tolerance_for_nscf(
            math::tolerances::minimum_eigenvalue_tolerance);
        planewave_iterator->set_diagonalizer_maximum_iterations_for_nscf(
            constants::non_self_consistent_field_maximum_davidson_iterations);

        // ─── Step 4: Run (single diagonalization pass) ────────────────────────
        iterator->run();

        /**
         * ─── Step 4b: Mean-field export (BerkeleyGW, etc.) ─────────────────────
         * This is the flavor BerkeleyGW normally consumes: an NSCF run with
         * many empty (conduction) bands beyond what SCF itself needs, at
         * whatever k-point set input.kpoints_list requests. Format selection
         * lives in the factory ("none" resolves to the Null object), so
         * there is no branching on the format name here -- see
         * io/mean_field_export/init.hpp.
         *
         * The exported charge density must be the CONVERGED SCF density from the
         * checkpoint, not the one this pass leaves on the iterator. Step 4 runs a
         * single turn of the shared per-iteration flow, which ends in a mixing
         * step, so the iterator's density is a blend of the atomic-superposition
         * initial guess and this pass's output -- governed by `mixing.beta`, a
         * parameter that has no meaning for a non-self-consistent run. Measured on
         * the silicon example (`beta` 0.3): 12% off the converged density in L2
         * over the whole G set (19% when the NSCF k-set is the symmetry-reduced
         * IBZ), against 0.01% once mixing is bypassed -- and, being a *recomputed*
         * density, it would additionally degrade with a coarse k-set even without
         * mixing. pw2bgw draws the same line: `write_rhog` writes QE's *saved*
         * charge density, and recomputing it from the bands is a separate opt-in
         * diagnostic (`calc_rhog`, reached only via `rhog_nvmin`/`rhog_nvmax`).
         */
        auto mean_field_state = planewave_iterator->result();
        if (effective_potential_snapshot.charge_density_real.empty())
            throw std::runtime_error(
                "NonSelfConsistentFieldFlavor: checkpoint at '" +
                effective_potential_checkpoint_path +
                "' carries no charge density, so the converged SCF density cannot be "
                "restored for the mean-field export. Re-run the SCF calculation to "
                "regenerate the checkpoint.");
        restore_converged_charge_density(mean_field_state,
                                         effective_potential_snapshot.charge_density_real);
        /**
         * Vxc(G) likewise belongs to the converged density, and this pass never
         * computes it at all: the potential came from the checkpoint whole, so
         * without this the state carries an empty Vxc(G) and BerkeleyGW's
         * vxc.dat cannot be written. pw2bgw evaluates Vxc from the same saved
         * density (its `vxc_flag` path), and from the valence part of it: see
         * valence_exchange_correlation_potential_g_of().
         */
        mean_field_state.exchange_correlation_potential_g =
            planewave_iterator->valence_exchange_correlation_potential_g_of(
                effective_potential_snapshot.charge_density_real);
        /**
         * Which rank (if any) writes is decided by the execution topology --
         * see export_and_record_mean_field_under_topology()'s comment, which
         * the MPI entry points reach through this flavor as the SCF's do.
         */
        export_and_record_mean_field_under_topology(
            planewave_iterator->execution_policy(),
            [&mean_field_state, &input] {
                auto record = build_mean_field_record(mean_field_state);
                if (input.mean_field_export_kleinman_bylander_projectors)
                    add_kleinman_bylander_projectors_to_mean_field_record(record, mean_field_state);
                return record;
            },
            input,
            input.output_directory);

        // ─── Step 5: Results ─────────────────────────────────────────────────────
        generated::Results results = new_results("non_self_consistent_field");
        /**
         * An NSCF total energy is meaningless — list it as 0 to signal this.
         * A single-pass NSCF is always "converged" — there is no iterative convergence
         * criterion applicable. The SCF iterator reports is_converged()=false because
         * the KohnShamSCFIterator convergence check requires iteration_>1 (which is
         * never reached with maximum_iterations=1). Override to true so that the Python
         * layer does not raise SCFNotConvergedError on a valid NSCF result.
         */
        results.runtime_log.converged = true;
        results.runtime_log.execution = planewave_iterator->execution_policy().execution_record();
        set_energy_records(results, 0.0, PlanewaveKohnShamEnergies{});
        set_fermi_energy(results, planewave_iterator->quantum_system().fermi_energy());

        /**
         * Every k-point state of the run, gathered across pools -- each pool
         * diagonalized only its own share of the list -- in the list's order.
         */
        set_record(results, assemble_global_eigenvalues_at_kpoints(*planewave_iterator));
        const std::size_t number_of_kpoint_states =
            find_record<generated::EigenvaluesAtKpoints>(results)->values.size();
        const int number_of_bands = static_cast<int>(planewave_iterator->eigenvalues().cols());

        // ─── Step 6: Atomic-orbital (ortho-atomic/Löwdin) projections (opt-in) ──
        if (input.compute_projections) {
            /**
             * This pool's k-point states: the projections are taken where the
             * wavefunctions are, then gathered like the eigenvalues above. The
             * state index is spin-major over the pool's physical k-points, so a
             * state splits into the physical k-point (G-space basis,
             * spin-independent) and the state (wavefunction storage,
             * spin-dependent).
             */
            const int local_number_of_kpoint_states =
                static_cast<int>(planewave_iterator->eigenvalues().rows());
            const int local_number_of_physical_kpoints =
                planewave_iterator->kpoint_grid().number_of_kpoints();
            log::info("     NSCF: computing ortho-atomic projections at {} k-points{}",
                      static_cast<int>(number_of_kpoint_states),
                      planewave_iterator->pseudopotential_set().has_augmentation_charges()
                          ? " in the S metric (ultrasoft/PAW augmentation)"
                          : "");
            PlanewaveWavefunctionProjectionCalculator projection_calculator(
                planewave_iterator->crystal_structure(), planewave_iterator->pseudopotential_set());

            generated::ResultsOrbitalProjections orbital_projections;
            for (const auto& orbital_label : projection_calculator.orbital_labels()) {
                generated::ResultsOrbitalProjectionsLabelsItem label;
                label.atom_index = orbital_label.atom_index;
                label.species_type_index = orbital_label.species_type_index;
                label.shell_label = orbital_label.shell_label;
                label.angular_momentum = orbital_label.angular_momentum;
                label.magnetic_quantum_number = orbital_label.magnetic_quantum_number;
                orbital_projections.labels.push_back(label);
            }
            const int number_of_orbitals = static_cast<int>(orbital_projections.labels.size());

            // One flat [band][orbital] block per local k-point state.
            std::vector<std::vector<Real>> local_projections(local_number_of_kpoint_states);
            for (int kpoint_state_index = 0; kpoint_state_index < local_number_of_kpoint_states;
                 kpoint_state_index++) {
                int physical_kpoint_index = kpoint_state_index % local_number_of_physical_kpoints;
                /**
                 * Kept alive alongside the operator that reads it: the ultrasoft
                 * overlap holds the matrix by reference.
                 */
                Matrix<Complex> kleinman_bylander_projectors;
                std::unique_ptr<OverlapOperator> overlap_operator =
                    build_projection_overlap_operator(
                        *planewave_iterator, physical_kpoint_index, kleinman_bylander_projectors);
                const std::vector<std::vector<Real>> projections =
                    projection_calculator.compute_projections_at_kpoint(
                        physical_kpoint_index,
                        kpoint_state_index,
                        planewave_iterator->number_of_plane_waves(physical_kpoint_index),
                        number_of_bands,
                        planewave_iterator->g_vector_indices(physical_kpoint_index),
                        planewave_iterator->crystal_structure(),
                        planewave_iterator->kpoint_grid(),
                        planewave_iterator->planewave_basis(),
                        planewave_iterator->pseudopotential_set(),
                        planewave_iterator->quantum_system(),
                        overlap_operator.get());
                for (const auto& band_projections : projections) {
                    local_projections[kpoint_state_index].insert(
                        local_projections[kpoint_state_index].end(),
                        band_projections.begin(),
                        band_projections.end());
                }
            }

            orbital_projections.values.assign(number_of_kpoint_states,
                                              std::vector<std::vector<double>>(number_of_bands));
            if (number_of_orbitals > 0) {
                const std::vector<std::vector<Real>> global_projections =
                    assemble_global_kpoint_state_blocks(*planewave_iterator,
                                                        local_projections,
                                                        number_of_bands * number_of_orbitals);
                for (std::size_t state_index = 0; state_index < global_projections.size();
                     ++state_index) {
                    for (int band_index = 0; band_index < number_of_bands; ++band_index) {
                        const auto band_begin =
                            global_projections[state_index].begin() +
                            static_cast<std::ptrdiff_t>(band_index) * number_of_orbitals;
                        orbital_projections.values[state_index][band_index].assign(
                            band_begin, band_begin + number_of_orbitals);
                    }
                }
            }
            results.properties.orbital_projections = orbital_projections;
        }

        // ─── Step 7: interpolation model (opt-in) ──────────────────────────────
        if (input.interpolation.requested()) {
            InterpolationRecord record = capture_interpolation_record(
                input.interpolation, *planewave_iterator, number_of_bands);
            write_and_export_interpolation_record(record, input);
        }

        return results;
    }

  private:
    /**
     * The overlap operator the atomic-orbital projections must be taken in at
     * one k-point, or null when the pseudopotentials leave S the identity.
     *
     * Ultrasoft and PAW datasets carry augmentation charges, so S = 1 + Σ q_ij
     * |β_i⟩⟨β_j| and the projection that conserves charge is ⟨φ|S|ψ⟩. Building
     * that operator needs this k-point's Kleinman-Bylander projector matrix,
     * which is returned through `kleinman_bylander_projectors` because the
     * operator holds it by reference and this is the only caller that knows how
     * long it must live. Dispatching on the pseudopotential set here, rather
     * than inside the projection calculator, keeps that calculator free of the
     * question -- the same split build_wannier_interface_record() makes for the
     * Wannier interface matrices.
     */
    [[nodiscard]] static std::unique_ptr<OverlapOperator>
    build_projection_overlap_operator(const PlanewavePseudopotentialKohnShamSCFIterator& iterator,
                                      int physical_kpoint_index,
                                      Matrix<Complex>& kleinman_bylander_projectors) {
        if (!iterator.pseudopotential_set().has_augmentation_charges()) {
            return nullptr;
        }
        const NormConservingPlanewaveKohnShamQuantumSystem& quantum_system =
            iterator.norm_conserving_quantum_system();
        /**
         * create_overlap_operator() reads the same predicate to choose between
         * the identity and the augmentation overlap, so a disagreement here
         * would not fail -- it would hand back S = I and write complete-looking
         * projections in the wrong metric, which is the failure this whole path
         * exists to remove. Refuse instead, as the Wannier interface builder
         * does for the same disagreement.
         */
        if (!quantum_system.has_ultrasoft()) {
            throw std::logic_error(
                "NSCF projections: the pseudopotentials carry augmentation charges but the "
                "quantum system is not the ultrasoft one; the iterator factory and this flavor "
                "disagree about the run's pseudopotential type.");
        }
        const auto& pseudopotential_basis = quantum_system.pseudopotential_basis();
        compute_kleinman_bylander_projectors(
            iterator.kpoint_grid()[physical_kpoint_index].coordinates,
            iterator.crystal_structure(),
            iterator.planewave_basis(),
            pseudopotential_basis.pseudopotential_set(),
            pseudopotential_basis.beta_function_table(),
            iterator.g_vector_indices(physical_kpoint_index),
            iterator.number_of_plane_waves(physical_kpoint_index),
            pseudopotential_basis.total_number_of_projectors(),
            kleinman_bylander_projectors);
        return create_overlap_operator(iterator.crystal_structure(),
                                       quantum_system,
                                       kleinman_bylander_projectors,
                                       iterator.execution_policy());
    }

    /**
     * Replaces `state`'s charge density with the converged SCF density loaded
     * from the checkpoint, keeping the G-space copy consistent with the
     * real-space one.
     *
     * Both fields are overwritten deliberately. `charge_density_real` is what
     * the mean-field exporters read, but `charge_density_g` is what the force
     * and stress calculators read (see forces/planewave/pseudopotential and
     * stress_tensor/planewave/internal/exchange_correlation), so leaving it
     * holding the mixed density would plant exactly the same inconsistency one
     * layer down for whoever computes a property off an NSCF state next.
     * The G-space density is the spin-SUMMED total, matching how
     * PlanewaveEffectivePotentialCalculator::compute() derives it.
     */
    static void restore_converged_charge_density(
        NormConservingPlanewaveKohnShamSCFState& state,
        const std::vector<std::vector<Real>>& converged_charge_density_real) {
        int local_grid_points = state.planewave_basis->fft_rho().local_grid_points();
        /**
         * Checked rather than assumed: the loop below indexes the checkpoint's
         * arrays by THIS run's grid size, so a checkpoint written at a different
         * `cutoffs.density` (hence a different charge-density FFT grid) would
         * otherwise read past the end. Fail with something the user can act on.
         */
        for (const auto& spin_density : converged_charge_density_real) {
            if (static_cast<int>(spin_density.size()) != local_grid_points)
                throw std::runtime_error(
                    "NonSelfConsistentFieldFlavor: the checkpoint's charge density has " +
                    std::to_string(spin_density.size()) + " grid points but this run's " +
                    "charge-density FFT grid has " + std::to_string(local_grid_points) +
                    ". The NSCF run must use the same cutoffs.density as the SCF run that "
                    "wrote the checkpoint.");
        }
        state.charge_density_real = converged_charge_density_real;

        std::vector<Real> total_charge_density_real(local_grid_points, 0.0);
        for (const auto& spin_density : converged_charge_density_real) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                total_charge_density_real[grid_index] += spin_density[grid_index];
            }
        }
        state.planewave_basis->rho_r_to_g(total_charge_density_real, state.charge_density_g);
    }

    IteratorFactory iterator_factory_;
};

} // namespace q3
