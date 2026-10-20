#pragma once

/**
 * KohnShamSCFIteratorFactory — dispatch by basis_type and representation_type.
 *
 * Two creation paths:
 *   1. create_kohn_sham_iterator(InputData, execution_policy) — production path.
 *      Performs all method-specific preparation (PP loading, crystal construction,
 *      symmetry, k-points) and transfers resource ownership into the iterator.
 *      The execution policy governs k-point distribution and pool-level reductions.
 *   2. create_kohn_sham_iterator_from_config(Config) — test-only path with
 *      externally-owned resources.
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "io/input/data.hpp"
#include "io/input/system_builder.hpp"
#include "io/pseudopotential/loader.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_4_computation/iterator/kohn_sham/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/factory.hpp"

namespace q3 {

/**
 * Create a Kohn-Sham iterator from a pre-built config struct.
 *
 * Test-only path: the caller owns the Crystal, KPointsGrid, and
 * PseudopotentialSet and must keep them alive for the iterator's lifetime.
 * Production code should use create_kohn_sham_iterator(InputData, policy) instead.
 */
inline std::unique_ptr<KohnShamSCFIterator>
create_kohn_sham_iterator_from_config(KohnShamSCFIteratorConfig config) {
    if (config.basis_type == "planewave" && config.representation_type == "pseudopotential") {
        return create_planewave_pseudopotential_iterator(std::move(config));
    }
    throw std::runtime_error("Unsupported basis_type/representation_type combination: " +
                             config.basis_type + "/" + config.representation_type);
}

/**
 * Create a Kohn-Sham iterator from parsed InputData and an execution policy.
 *
 * Performs all method-specific preparation:
 *   1. Loads pseudopotentials (patches species valence_charge)
 *   2. Builds Crystal from input lattice/atom data
 *   3. Detects symmetry
 *   4. Generates and reduces k-points (global grid)
 *   5. Calls execution_policy.local_kpoint_grid() to get this pool's subset
 *   6. Dispatches NC vs US based on loaded PP types
 *
 * The returned iterator owns the Crystal, local KPointsGrid, and PseudopotentialSet.
 *
 * @param input            Parsed input parameters.
 * @param execution_policy Policy governing k-point distribution and pool reductions.
 *                         Its lifetime must exceed the iterator's lifetime.
 */
inline std::unique_ptr<PlanewaveKohnShamSCFIterator>
create_kohn_sham_iterator(const InputData& input,
                          const execution::PlanewaveExecutionPolicy& execution_policy) {
    // 1. Load pseudopotentials and patch species valence_charge
    std::vector<AtomSpecies> patched_species = input.species;
    PseudopotentialSet pseudopotential_set =
        load_pseudopotentials(patched_species, input.pseudopotential_directory);

    /**
     * 2. Build crystal with patched valence charges (shared builder — see
     *    io/input/system_builder.hpp, also used by the q3_core bindings)
     */
    Crystal crystal = build_crystal_from_input(input, patched_species);

    // 3. Detect symmetry
    SymmetryAnalyzer::find_spacegroup(crystal);

    /**
     * 4. Generate global k-point grid (explicit list or MP generation, IBZ
     *    reduction, LSDA spin-channel weight scaling)
     */
    KPointsGrid global_kpoint_grid = build_global_kpoint_grid_from_input(input, crystal);

    /**
     * 5. Distribute k-points: each pool receives its local subset.
     *    In serial mode this is an identity operation (returns global_kpoint_grid unchanged).
     *    If the requested pool count exceeds the number of IBZ k-points, clamp to the
     *    k-point count and emit a warning. This clamp is Q3's own: an empty
     *    pool would skip wavefunction initialization while other pools call
     *    MPI_Allreduce inside reduce_vector, and the mismatched calls then pair
     *    up across the inter-pool communicator -- MPI_ERR_TRUNCATE at best.
     *    Quantum ESPRESSO does not clamp: it rejects only pool counts that
     *    exceed or do not divide the rank count, and hands surplus pools zero
     *    k-points with an informational message. Its k-point count bounds only
     *    the pool count it picks for itself when the user requested none.
     */
    {
        int number_of_kpoints = global_kpoint_grid.number_of_kpoints();

        /**
         * With no pool count requested, choose one: the largest divisor of the
         * rank count that does not exceed the k-point count, which gives every
         * rank k-points to work on and leaves as few as possible sharing a pool.
         *
         * The previous default put every rank in a single pool, so extra ranks
         * subdivided the FFT instead of taking k-points. That path has no
         * measured speedup -- it is slower on the supercell it was benchmarked
         * against -- and it disables gamma-only, stress, checkpointing and
         * export. Nobody chose it; it was simply what happened when --pools was
         * omitted.
         */
        if (!execution_policy.number_of_pools_was_requested()) {
            int number_of_processes = execution_policy.number_of_processes();
            int chosen_number_of_pools = 1;
            for (int candidate = 1; candidate <= number_of_processes; candidate++) {
                if (number_of_processes % candidate == 0 && candidate <= number_of_kpoints) {
                    chosen_number_of_pools = candidate;
                }
            }
            if (chosen_number_of_pools != execution_policy.number_of_pools()) {
                log::info("MPI topology (automatic): {} rank(s) -> {} k-point pool(s) x {} "
                          "rank(s) per pool, from {} irreducible k-point(s)",
                          number_of_processes,
                          chosen_number_of_pools,
                          number_of_processes / chosen_number_of_pools,
                          number_of_kpoints);
                const_cast<execution::PlanewaveExecutionPolicy&>(execution_policy)
                    .reconfigure_topology(chosen_number_of_pools);
            }

            /**
             * If ranks are still left over, they would form a band group by
             * default. Refuse instead of quietly running the slow path: the
             * caller can still ask for it explicitly.
             */
            if (execution_policy.band_size() > 1) {
                throw std::invalid_argument(
                    "q3_mpi was started on " + std::to_string(number_of_processes) +
                    " ranks, but this input has only " + std::to_string(number_of_kpoints) +
                    " irreducible k-point(s), so " + std::to_string(execution_policy.band_size()) +
                    " ranks would share each pool. Q3 parallelizes over k-points; ranks sharing a "
                    "pool distribute the FFT instead, which has no measured speedup and disables "
                    "gamma-only, stress, checkpointing and export. Use a rank count that divides "
                    "into single-rank pools (here: " +
                    std::to_string(number_of_kpoints) +
                    "), give the remaining cores to OMP_NUM_THREADS, or ask for the shared-pool "
                    "layout explicitly with --pools.");
            }
        }

        int number_of_pools = execution_policy.number_of_pools();
        if (number_of_pools > number_of_kpoints) {
            log::warn("Requested k-point pools ({}) exceeds available IBZ k-points ({}). "
                      "Clamping to {} pools to match QE behaviour.",
                      number_of_pools,
                      number_of_kpoints,
                      number_of_kpoints);
            /**
             * Delegate to the virtual reconfigure_topology so the MPI policy can
             * reconfigure its communicators. The serial policy's default no-op
             * implementation is never reached here (serial always has 1 pool).
             */
            const_cast<execution::PlanewaveExecutionPolicy&>(execution_policy)
                .reconfigure_topology(number_of_kpoints);
        }
    }
    KPointsGrid local_kpoint_grid = execution_policy.local_kpoint_grid(global_kpoint_grid);

    // 6. Build config and dispatch NC vs US
    KohnShamSCFIteratorConfig scf_config;
    scf_config.crystal_structure = nullptr;
    scf_config.number_of_spin_components = input.number_of_spin_components;
    scf_config.convergence_threshold = input.convergence_threshold;
    scf_config.maximum_iterations = input.maximum_iterations;
    scf_config.exchange_correlation_functional = input.exchange_correlation_functional;
    scf_config.mixing_mode = input.mixing_mode;
    scf_config.mixing_beta = input.mixing_beta;
    scf_config.mixing_history_size = input.mixing_history_size;
    scf_config.mixing_preconditioning = input.mixing_preconditioning;
    scf_config.mixing_q0 = input.mixing_q0;
    scf_config.smearing = input.smearing;
    scf_config.smearing_width = input.smearing_width;
    scf_config.methfessel_paxton_order = input.methfessel_paxton_order;
    scf_config.dispersion_correction = input.dispersion_correction;
    scf_config.total_charge = input.total_charge;
    scf_config.hubbard = input.hubbard;
    scf_config.hubbard_starting_eigenvalues = input.hubbard_starting_eigenvalues;
    scf_config.hubbard_fixed_occupation_iterations = input.hubbard_fixed_occupation_iterations;
    scf_config.hubbard_projector_type = input.hubbard_projector_type;
    scf_config.wavefunction_energy_cutoff = input.wavefunction_energy_cutoff;
    scf_config.charge_density_energy_cutoff = input.charge_density_energy_cutoff;
    scf_config.number_of_bands = input.number_of_bands;
    scf_config.diagonalization_method = input.diagonalization_method;
    scf_config.davidson_basis_expansion_factor = input.davidson_basis_expansion_factor;
    scf_config.use_deterministic_fft_planning = input.use_deterministic_fft_planning;
    scf_config.use_exact_wavefunction_grid = input.use_exact_wavefunction_grid;
    scf_config.disable_symmetry = input.disable_symmetry;
    scf_config.calculation = input.calculation;
    scf_config.starting_magnetization = input.starting_magnetization;
    scf_config.atomic_starting_magnetization = input.atomic_starting_magnetization;
    scf_config.band_path = input.band_path;
    scf_config.band_path_points_per_segment = input.band_path_points_per_segment;
    scf_config.nonlocal_stress_derivative_method = input.nonlocal_stress_derivative_method;
    scf_config.execution_policy = &execution_policy;

    /**
     * PAW first: a PAW dataset also has augmentation charges, so testing that
     * capability first would route it to the ultrasoft iterator and drop every
     * one-center term without a word.
     */
    bool is_paw =
        pseudopotential_set.has_pseudopotential_type(PseudopotentialType::projector_augmented_wave);
    bool needs_augmentation = pseudopotential_set.has_augmentation_charges();

    std::unique_ptr<PlanewavePseudopotentialKohnShamSCFIterator> iterator;
    if (is_paw) {
        iterator =
            std::make_unique<PawPlanewavePseudopotentialKohnShamSCFIterator>(std::move(scf_config));
    } else if (needs_augmentation) {
        iterator = std::make_unique<UltrasoftPlanewavePseudopotentialKohnShamSCFIterator>(
            std::move(scf_config));
    } else {
        iterator =
            std::make_unique<PlanewavePseudopotentialKohnShamSCFIterator>(std::move(scf_config));
    }

    // Transfer ownership of resources into the iterator.
    iterator->set_owned_crystal(std::move(crystal));
    iterator->set_owned_kpoint_grid(std::move(local_kpoint_grid));
    iterator->set_owned_pseudopotential_set(std::move(pseudopotential_set));

    return iterator;
}

} // namespace q3
