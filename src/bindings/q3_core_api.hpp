#pragma once

/**
 * q3_core_api.hpp — In-process Q3 calculation API for the nanobind module.
 *
 * Exposes run_scf() and run_nscf() as free functions that accept a YAML
 * input file path, run the corresponding flavor in-process, and return a
 * JSON results string.
 *
 * Each call resets the ProfilingCounters singleton so that metrics from
 * successive invocations do not accumulate.
 *
 * Thread safety: NOT thread-safe.  The singleton XC functional state and
 * FFTW plan cache are not protected by mutexes.  Callers must serialize
 * invocations (GIL or external lock).
 */

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "io/input/parser/factory.hpp"
#include "io/input/system_builder.hpp"
#include "io/logger.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/flavors/factory.hpp"
#include "layer_4_computation/flavors/result.hpp"

namespace q3 {
namespace bindings {

/**
 * Run a self-consistent field calculation and return results as a JSON string.
 *
 * Parameters
 * ----------
 * input_path:
 *     Path to a Q3 YAML input file with calculation = scf (or equivalent).
 * write_output_files:
 *     When true (default), also write results.json and checkpoint files to
 *     the outdir specified in the YAML.  When false, results are only
 *     returned as the JSON string (useful for in-memory workflows).
 *
 * Returns
 * -------
 * JSON string with the same schema as results.json.
 */
inline std::string run_scf(const std::string& input_path, bool write_output_files = true) {
    ProfilingCounters::instance().reset();

    InputData input = parse_input(input_path);
    /**
     * No command line to defer to in-process, so the deck alone decides -- and
     * the empty second argument is what makes a "default" deck put the level
     * back rather than inherit whatever the previous call left it at.
     */
    log::apply_verbosity(input.verbosity, "");
    input.calculation = "self_consistent_field";

    FlavorResult result = run_flavor(input);

    if (write_output_files) {
        if (!input.output_directory.empty()) {
            std::filesystem::create_directories(input.output_directory);
        }
        std::string results_path = input.output_directory + "/" + input.prefix + "_results.json";
        result.write_results_json(results_path);
    }

    return result.to_json_string();
}

/**
 * Run a non-self-consistent field (single-pass diagonalization) and return
 * results as a JSON string.
 *
 * Requires a prior SCF run to have written an effective-potential checkpoint
 * in <outdir>/<prefix>.save/effective-potential.q3cp.
 */
inline std::string run_nscf(const std::string& input_path, bool write_output_files = true) {
    ProfilingCounters::instance().reset();

    InputData input = parse_input(input_path);
    log::apply_verbosity(input.verbosity, "");
    input.calculation = "non_self_consistent_field";

    FlavorResult result = run_flavor(input);

    if (write_output_files) {
        if (!input.output_directory.empty()) {
            std::filesystem::create_directories(input.output_directory);
        }
        std::string results_path = input.output_directory + "/" + input.prefix + "_results.json";
        result.write_results_json(results_path);
    }

    return result.to_json_string();
}

/**
 * Stateful SCF Solver for in-process calculations.
 *
 * Caches loaded pseudopotentials to avoid disk I/O between successive
 * calculations (e.g. during geometry optimization steps).
 */
class SCFSolver {
  public:
    SCFSolver(const SCFSolver&) = delete;
    SCFSolver& operator=(const SCFSolver&) = delete;
    SCFSolver(SCFSolver&&) = default;
    SCFSolver& operator=(SCFSolver&&) = default;

    explicit SCFSolver(const std::string& input_path) {
        input_ = parse_input(input_path);
        log::apply_verbosity(input_.verbosity, "");
        patched_species_ = input_.species;
        pseudopotential_set_ =
            load_pseudopotentials(patched_species_, input_.pseudopotential_directory);
        /**
         * Built once here rather than per solve(): nothing provenance records
         * varies between steps of a relaxation -- it describes the engine, the
         * datasets and the requested numerics, never the coordinates -- and
         * re-reading and re-hashing every dataset on each of a hundred steps
         * would be pure repetition.
         */
        provenance_ = build_provenance_json(input_);
    }

    /**
     * Solve the SCF problem for the given atomic coordinates.
     */
    std::string solve(const std::vector<std::vector<double>>& coordinates,
                      bool write_output_files = true) {
        for (std::size_t atom_index = 0;
             atom_index < coordinates.size() && atom_index < input_.atoms.size();
             ++atom_index) {
            if (coordinates[atom_index].size() < 3) {
                throw std::runtime_error(
                    "SCFSolver::solve: coordinates must have 3 components per atom.");
            }
            input_.atoms[atom_index].position.x = coordinates[atom_index][0];
            input_.atoms[atom_index].position.y = coordinates[atom_index][1];
            input_.atoms[atom_index].position.z = coordinates[atom_index][2];
        }

        /**
         * Shared builders (io/input/system_builder.hpp) — the same crystal and
         * k-grid construction the iterator factory uses; input semantics must
         * never drift between the two entry points.
         */
        Crystal crystal = build_crystal_from_input(input_, patched_species_);

        SymmetryAnalyzer::find_spacegroup(crystal);

        KPointsGrid kpoint_grid = build_global_kpoint_grid_from_input(input_, crystal);

        KohnShamSCFIteratorConfig scf_config;
        scf_config.crystal_structure = nullptr;
        scf_config.pseudopotential_set = &pseudopotential_set_;
        scf_config.number_of_spin_components = input_.number_of_spin_components;
        scf_config.convergence_threshold = input_.convergence_threshold;
        scf_config.maximum_iterations = input_.maximum_iterations;
        scf_config.exchange_correlation_functional = input_.exchange_correlation_functional;
        scf_config.mixing_mode = input_.mixing_mode;
        scf_config.mixing_beta = input_.mixing_beta;
        scf_config.mixing_history_size = input_.mixing_history_size;
        scf_config.mixing_preconditioning = input_.mixing_preconditioning;
        scf_config.mixing_q0 = input_.mixing_q0;
        scf_config.smearing = input_.smearing;
        scf_config.smearing_width = input_.smearing_width;
        scf_config.methfessel_paxton_order = input_.methfessel_paxton_order;
        scf_config.wavefunction_energy_cutoff = input_.wavefunction_energy_cutoff;
        scf_config.charge_density_energy_cutoff = input_.charge_density_energy_cutoff;
        scf_config.number_of_bands = input_.number_of_bands;
        scf_config.diagonalization_method = input_.diagonalization_method;
        scf_config.calculation = input_.calculation;
        scf_config.starting_magnetization = input_.starting_magnetization;
        scf_config.atomic_starting_magnetization = input_.atomic_starting_magnetization;
        scf_config.band_path = input_.band_path;
        scf_config.band_path_points_per_segment = input_.band_path_points_per_segment;
        scf_config.nonlocal_stress_derivative_method = input_.nonlocal_stress_derivative_method;

        auto iterator = create_planewave_pseudopotential_iterator(std::move(scf_config));

        iterator->set_owned_crystal(std::move(crystal));
        iterator->set_owned_kpoint_grid(std::move(kpoint_grid));
        iterator->set_owned_pseudopotential_set(pseudopotential_set_.clone());

        if (!last_charge_density_.empty()) {
            log::debug("SCFSolver solve: Injecting cached charge density.");
            iterator->load_charge_density(last_charge_density_);
        } else {
            log::debug("SCFSolver solve: No cached charge density available.");
        }
        if (last_quantum_system_ != nullptr) {
            log::debug("SCFSolver solve: Injecting cached quantum system.");
            iterator->load_quantum_system(*last_quantum_system_);
        } else {
            log::debug("SCFSolver solve: No cached quantum system available.");
        }

        ProfilingCounters::instance().reset();
        iterator->run();

        const bool converged = iterator->is_converged();
        auto scf_state = iterator->result();

        if (converged) {
            log::debug(
                "SCFSolver solve: SCF converged. Caching charge density and quantum states.");
            last_charge_density_ = scf_state.charge_density_real;
            last_quantum_system_ =
                std::make_unique<PlanewaveKohnShamQuantumSystem>(iterator->quantum_system());
        }

        FlavorResult flavor_result;
        flavor_result.flavor = "self_consistent_field";
        flavor_result.converged = converged;
        flavor_result.number_of_self_consistent_field_iterations = iterator->iteration();
        flavor_result.provenance = provenance_;

        const auto& energies = scf_state.energies();
        flavor_result.total_energy_in_ry = energies.total();
        flavor_result.band_energy_in_ry = energies.band_energy;
        flavor_result.hartree_energy_in_ry = energies.hartree_energy;
        flavor_result.exchange_correlation_energy_in_ry = energies.exchange_correlation_energy;
        flavor_result.ewald_energy_in_ry = energies.ewald_energy;
        flavor_result.double_counting_correction_in_ry = energies.double_counting_correction;
        flavor_result.smearing_entropy_contribution_in_ry = energies.smearing_entropy_contribution;
        flavor_result.dispersion_energy_in_ry = energies.dispersion_energy();

        flavor_result.fermi_energy_in_ry = iterator->quantum_system().fermi_energy();

        /**
         * The same quantities SelfConsistentFieldFlavor carries, read from the
         * same accessors. This populator and that one are two hand-maintained
         * fillers of one FlavorResult, so a quantity added to one and not the
         * other makes the bindings and the command line disagree about what a
         * run reported -- which is exactly what results_json() being a single
         * serializer cannot protect against.
         */
        flavor_result.magnetization = iterator->magnetization();
        const auto& hubbard_occupation_traces = iterator->hubbard_occupation_traces();
        for (std::size_t manifold_index = 0; manifold_index < hubbard_occupation_traces.size() &&
                                             manifold_index < scf_state.hubbard_manifolds.size();
             ++manifold_index) {
            flavor_result.hubbard_occupations.push_back(
                {scf_state.hubbard_manifolds[manifold_index].atom_index,
                 hubbard_occupation_traces[manifold_index]});
        }

        if (converged) {
            auto [total_forces, forces_contributions] = ForceCalculator::compute_forces(
                scf_state, input_.dispersion_correction, input_.disable_symmetry);
            flavor_result.forces_in_ry_per_bohr = total_forces;
            flavor_result.add_force_contribution("local", forces_contributions.local);
            flavor_result.add_force_contribution("ewald", forces_contributions.ewald);
            flavor_result.add_force_contribution("nonlocal", forces_contributions.nonlocal);
            flavor_result.add_force_contribution("core_correction",
                                                 forces_contributions.core_correction);
            flavor_result.add_force_contribution("augmentation", forces_contributions.augmentation);
            flavor_result.add_force_contribution("dispersion", forces_contributions.dispersion);
            flavor_result.add_force_contribution("hubbard", forces_contributions.hubbard);

            auto [stress_total, stress_contributions] =
                compute_stress(scf_state,
                               input_.nonlocal_stress_derivative_method,
                               input_.dispersion_correction,
                               input_.disable_symmetry);

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
            flavor_result.add_stress_contribution("dispersion", stress_contributions.dispersion);
        }

        const auto& eigenvalue_matrix = iterator->eigenvalues();
        int number_of_kpoints = eigenvalue_matrix.rows();
        int number_of_bands = eigenvalue_matrix.cols();
        /**
         * kpoint_grid() holds one entry per PHYSICAL k-point; eigenvalue_matrix is
         * indexed by k-point STATE (physical kpoint * spin) -- see the identical
         * convention in flavors/self_consistent_field.hpp.
         */
        int number_of_physical_kpoints = iterator->kpoint_grid().number_of_kpoints();

        flavor_result.eigenvalues_in_ry.resize(number_of_kpoints);
        flavor_result.kpoint_weights.resize(number_of_kpoints);
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            flavor_result.eigenvalues_in_ry[kpoint_index].resize(1);
            flavor_result.eigenvalues_in_ry[kpoint_index][0].resize(number_of_bands);
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                flavor_result.eigenvalues_in_ry[kpoint_index][0][band_index] =
                    eigenvalue_matrix(kpoint_index, band_index);
            }
            flavor_result.kpoint_weights[kpoint_index] =
                iterator->kpoint_grid()[kpoint_index % number_of_physical_kpoints].weight;
        }

        if (write_output_files) {
            if (!input_.output_directory.empty()) {
                std::filesystem::create_directories(input_.output_directory);
            }
            std::string results_path =
                input_.output_directory + "/" + input_.prefix + "_results.json";
            flavor_result.write_results_json(results_path);
        }

        return flavor_result.to_json_string();
    }

  private:
    InputData input_;
    std::vector<AtomSpecies> patched_species_;
    PseudopotentialSet pseudopotential_set_;
    nlohmann::json provenance_;
    std::vector<std::vector<Real>> last_charge_density_;
    std::unique_ptr<PlanewaveKohnShamQuantumSystem> last_quantum_system_;
};

} // namespace bindings
} // namespace q3
