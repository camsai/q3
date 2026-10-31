#include "layer_0_core/profiling/counters.hpp"
#include <chrono>
#include <filesystem>
#include <string>

#include "io/addon_registration.hpp"
#include "io/cli.hpp"
#include "io/input/parser/factory.hpp"
#include "io/logger.hpp"
#include "io/output/companion_outputs.hpp"
#include "layer_4_computation/flavors/factory.hpp"
#include "utils/time.hpp"

using namespace q3;

// ─── CLI helpers ──────────────────────────────────────────────────────────────

/**
 * Report what an input resolved to, for `--check`.
 *
 * Deliberately says what the *calculation* is rather than echoing keys back:
 * the failure this exists to catch is a deck being read as a different
 * calculation than it describes, and the keys a user would have to compare by
 * hand to see that are exactly these.
 */
static void report_resolved_input(const InputData& input) {
    log::info("Input reads as:");
    log::info("  calculation       {}", input.calculation);
    log::info("  functional        {}",
              input.exchange_correlation_functional.empty()
                  ? "unresolved: named nowhere, and a species has no pseudopotential to state it"
                  : input.exchange_correlation_functional);
    log::info("  cutoffs           wavefunction {} Ry, density {} Ry",
              input.wavefunction_energy_cutoff,
              input.charge_density_energy_cutoff);
    log::info("  spin components   {}", input.number_of_spin_components);
    log::info("  atoms             {} in {} species", input.atoms.size(), input.species.size());
    // An explicit list takes precedence over the grid, so it is the one reported.
    if (!input.kpoints_list.empty())
        log::info("  k-points          {} listed explicitly", input.kpoints_list.size());
    else
        log::info("  k-point grid      {}x{}x{} shifted {}{}{}",
                  input.kpoint_grid_dimension_1,
                  input.kpoint_grid_dimension_2,
                  input.kpoint_grid_dimension_3,
                  input.kpoint_grid_shift_1,
                  input.kpoint_grid_shift_2,
                  input.kpoint_grid_shift_3);
    if (!input.hubbard.empty()) {
        for (const auto& hubbard : input.hubbard)
            log::info("  DFT+U             {} {} U = {} Ry ({} projectors)",
                      hubbard.species_symbol,
                      hubbard.orbital_label,
                      hubbard.u_rydberg,
                      input.hubbard_projector_type);
    }
    if (input.dispersion_correction != "none")
        log::info("  dispersion        {}", input.dispersion_correction);
    if (input.isolated_system_correction != "none")
        log::info("  isolated system   {}", input.isolated_system_correction);
    if (input.disable_symmetry)
        log::info("  symmetry          disabled");
}

// ─── Entry point ──────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    q3::addons::register_all();
    auto options = cli::parse_options(argc, argv, "q3");

    q3::log::init(options.log_file);

    /**
     * Answered before the input is resolved. `--version` must work in a
     * directory holding no input, where cli::read_input() fails.
     */
    if (options.print_version) {
        cli::print_version("q3");
        return 0;
    }

    if (!options.log_level.empty()) {
        log::set_level(options.log_level);
    }
    cli::print_header();

    int exit_code = 0;
    try {
        auto input = cli::read_input(options, "q3");

        if (options.check_only) {
            report_resolved_input(input);
            cli::print_footer();
            return 0;
        }

        // Dispatch based on calculation type.
        const std::string& calculation_type = input.calculation;
        log::info("     Calculation type: {}", calculation_type);

        /**
         * Dispatch to the appropriate flavor (scf or non_self_consistent_field).
         * Multi-step workflows (band_structure, dos, relaxation) are orchestrated by the
         * Python layer, which chains these single-flavor runs.
         */
        generated::Results results = run_flavor(input);
        write_results_json(results, input.output_directory + "/" + input.prefix + "_results.json");
        write_companion_outputs(input, results);

    } catch (const std::exception& e) {
        log::error("Error: {}", e.what());
        exit_code = 1;
    }

    auto& profiling = q3::ProfilingCounters::instance();
    q3::log::info("     Total FFT calls:                {}", profiling.fft_call_count.load());
    q3::log::info("     Total Davidson iterations:      {}",
                  profiling.davidson_iteration_count.load());
    q3::log::info("     Total Hamiltonian applications: {}",
                  profiling.hamiltonian_application_count.load());
    profiling.print_memory_summary();
    profiling.davidson_timing.print_summary();
    cli::print_footer();
    return exit_code;
}
