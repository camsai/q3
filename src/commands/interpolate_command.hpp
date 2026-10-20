#pragma once

/**
 * InterpolateCommand — `q3_utility interpolate <input.yml>`: evaluate a saved
 * interpolation record at the k-points an input asks for, and on a dense mesh
 * for a density of states.
 *
 * The input is an ordinary Q3 manifest with `calculation: interpolation` and
 * a `control.interpolation` block. Its `outdir`/`prefix` name the run whose
 * NSCF built the record (`<outdir>/<prefix>.save/interpolation.q3ip`); its
 * `control.kpoints_list`, or a `band_path` whose vertices carry coordinates,
 * names the k-points; `control.interpolation.dos_mesh` with `dos_broadening`
 * and `dos_energy_steps` asks for the DOS. Nothing here touches plane waves:
 * the record is read, the method's interpolator comes from the umbrella
 * factory, and the layer-5 calculators evaluate it -- which is why this is a
 * utility subcommand and not an engine flavor.
 *
 * Output: `<outdir>/<prefix>_interpolation_results.json` (or `--output`),
 * with `eigenvalues_ry` shaped [k-point][spin][band] like the engine's own
 * results file so the Python extractors read it unchanged, the `kpoints` it
 * was evaluated at, the `density_of_states`, the method's diagnostics, and
 * the same provenance stamp every result carries.
 */

#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/input/data.hpp"
#include "io/input/parser/factory.hpp"
#include "io/interpolation/checkpoint.hpp"
#include "io/interpolation/record_and_export.hpp"
#include "io/interpolation/results_document.hpp"
#include "io/logger.hpp"
#include "io/provenance/factory.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/construction.hpp"
#include "layer_4_computation/interpolator/factory.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"
#include "layer_4_computation/interpolator/input_data/specification.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/quality_assessor.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/init.hpp"
#include "layer_5_properties/band_structure/interpolated/init.hpp"
#include "layer_5_properties/density_of_states/interpolated/init.hpp"

namespace q3 {

class InterpolateCommand {
  public:
    static void print_usage() {
        log::error("Usage: q3_utility interpolate <input.yml> [--output <file>]");
        log::error("  <input.yml>      a manifest with calculation: interpolation and a "
                   "control.interpolation block;");
        log::error("                   its control.outdir/prefix name the run whose NSCF saved the "
                   "record");
        log::error("  --output <file>  where to write the JSON results (default "
                   "<outdir>/<prefix>_interpolation_results.json)");
    }

    /// Returns the process exit code; argument errors print usage, everything else throws.
    static int run(const std::vector<std::string>& arguments) {
        std::string input_file;
        std::string output_file;
        for (std::size_t index = 0; index < arguments.size(); index++) {
            const std::string& argument = arguments[index];
            auto take_value = [&](const char* flag) -> std::string {
                if (index + 1 >= arguments.size()) {
                    throw std::runtime_error(std::string(flag) + " needs a value");
                }
                return arguments[++index];
            };
            if (argument == "--output") {
                output_file = take_value("--output");
            } else if (argument == "-h" || argument == "--help") {
                print_usage();
                return 0;
            } else if (argument.rfind("-", 0) == 0) {
                log::error("Unknown option '{}'", argument);
                print_usage();
                return 1;
            } else if (input_file.empty()) {
                input_file = argument;
            } else {
                log::error("Unexpected extra argument '{}'", argument);
                print_usage();
                return 1;
            }
        }
        if (input_file.empty()) {
            print_usage();
            return 1;
        }
        evaluate(input_file, output_file);
        return 0;
    }

  private:
    /// The record the input names, evaluated where the input asks, written as the results file.
    static void evaluate(const std::string& input_file, const std::string& output_file) {
        const InputData input = evaluation_input(input_file);
        const std::string record_path =
            interpolation_record_path(input.output_directory, input.prefix);
        const InterpolationRecord record = saved_record(record_path, input.interpolation);

        const QuantumStateInterpolatorConstruction built =
            create_quantum_state_interpolator(record, input.interpolation);
        const QuantumStateInterpolator& interpolator = *built.interpolator;
        log::info("q3_utility interpolate: {} from {}", interpolator.description(), record_path);

        InterpolationResultsDocument document(build_provenance_json(input),
                                              record.fermi_energy_rydberg);
        document.describe_model(record.method, record_path, interpolator);
        document.add_construction(
            built.wannier,
            WannierQualityAssessor(record.real_lattice_bohr, atom_positions_crystal(input))
                .assess(built.wannier));

        const std::vector<Vector3D> kpoints = evaluation_kpoints(input);
        if (!kpoints.empty()) {
            document.add_bands(kpoints,
                               InterpolatedBandStructureCalculator(interpolator).compute(kpoints));
            log::info("evaluated {} k-points", kpoints.size());
        }
        if (input.interpolation.wants_density_of_states()) {
            const std::array<int, 3>& mesh = input.interpolation.dos_mesh;
            document.add_density_of_states(
                mesh,
                input.dos_broadening,
                InterpolatedDensityOfStatesCalculator(interpolator)
                    .compute(
                        mesh[0], mesh[1], mesh[2], input.dos_broadening, input.dos_energy_steps));
            log::info(
                "evaluated a density of states on a {}x{}x{} mesh", mesh[0], mesh[1], mesh[2]);
        }
        if (kpoints.empty() && !input.interpolation.wants_density_of_states()) {
            log::warn("nothing to evaluate: give control.kpoints_list (or a band_path whose "
                      "vertices carry coordinates) for bands, or control.interpolation.dos_mesh "
                      "for a density of states");
        }

        document.write(output_file.empty() ? input.output_directory + "/" + input.prefix +
                                                 "_interpolation_results.json"
                                           : output_file);
    }

    /// The manifest, refused unless it is an evaluation request.
    [[nodiscard]] static InputData evaluation_input(const std::string& input_file) {
        InputData input = parse_input(input_file);
        if (input.calculation != "interpolation") {
            throw std::runtime_error(
                "q3_utility interpolate evaluates inputs with calculation: interpolation; this one "
                "has calculation: " +
                input.calculation +
                ". A build request (calculation: nscf with a control.interpolation block) runs "
                "with q3, which saves the record this command reads.");
        }
        if (!input.interpolation.requested()) {
            throw std::runtime_error(
                "the input has no control.interpolation block, so there is nothing to evaluate");
        }
        return input;
    }

    /// The record an NSCF run saved at `record_path`, refused if the manifest names another method.
    [[nodiscard]] static InterpolationRecord
    saved_record(const std::string& record_path, const InterpolationSpecification& specification) {
        if (!std::filesystem::exists(record_path)) {
            throw std::runtime_error(
                "no interpolation record at '" + record_path +
                "'. An NSCF run (calculation: nscf) whose control.interpolation block asked for "
                "it writes the record into its own outdir/prefix; point this input's "
                "control.outdir and control.prefix at that run, or import Wannier90 files with "
                "`q3_utility convert --from wannier90`.");
        }
        InterpolationRecord record = read_interpolation_record(record_path);
        if (!specification.method.empty() && specification.method != record.method) {
            throw std::runtime_error("the input asks for control.interpolation.method '" +
                                     specification.method + "' but the record at '" + record_path +
                                     "' was built by '" + record.method + "'");
        }
        return record;
    }

    /// The atoms of the manifest's material, which the Wannier centres are judged against.
    [[nodiscard]] static std::vector<Vector3D> atom_positions_crystal(const InputData& input) {
        std::vector<Vector3D> positions;
        positions.reserve(input.atoms.size());
        for (const Atom& atom : input.atoms) {
            positions.push_back(atom.position);
        }
        return positions;
    }

    /**
     * The k-points to evaluate: an explicit `kpoints_list` as given, or a
     * `band_path` whose vertices carry coordinates, walked
     * `band_path_points_per_segment` steps per segment plus the final vertex
     * -- the same sampling the band-structure workflow uses.
     *
     * Labels without coordinates are refused at parse time (yaml.hpp): this
     * command has no high-symmetry table (that lookup lives in
     * q3.workflows.kpath), and evaluating a label at the origin would
     * silently return Gamma's eigenvalues under another name.
     */
    [[nodiscard]] static std::vector<Vector3D> evaluation_kpoints(const InputData& input) {
        std::vector<Vector3D> kpoints;
        if (!input.kpoints_list.empty()) {
            kpoints.reserve(input.kpoints_list.size());
            for (const auto& entry : input.kpoints_list) {
                kpoints.push_back(entry.first);
            }
            return kpoints;
        }
        if (input.band_path.empty()) {
            return kpoints;
        }
        const int per_segment = std::max(1, input.band_path_points_per_segment);
        for (std::size_t segment = 0; segment + 1 < input.band_path.size(); segment++) {
            const Vector3D& start = input.band_path[segment].second;
            const Vector3D& end = input.band_path[segment + 1].second;
            for (int step = 0; step < per_segment; step++) {
                const Real fraction = static_cast<Real>(step) / static_cast<Real>(per_segment);
                kpoints.push_back(start + (end - start) * fraction);
            }
        }
        kpoints.push_back(input.band_path.back().second);
        return kpoints;
    }
};

} // namespace q3
