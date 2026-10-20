#pragma once

/**
 * create_input_parser — factory for input file parsers.
 *
 * The parser comes from the registered input format (input_format_registry.hpp)
 * that the path or `--format` names. Core registers two:
 *   ".in" / ".qe"  → QEInputParser
 *   all others     → YAMLInputParser
 * and a build-time add-on may register more. A format recognised by file name
 * or directory contents outranks one recognised by extension, which outranks
 * the YAML fallback, so a path goes to the most specific format that claims it.
 *
 * Also provides parse_input(), which every entry point reads its input through,
 * and substitute_material(), the `--material` override.
 */

#include <cctype>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "io/input/parser/init.hpp"
#include "io/input/parser/input_format.hpp"
#include "io/input/parser/input_format_registry.hpp"
#include "io/pseudopotential/loader.hpp"

namespace q3 {

/**
 * The registered format an input is read as.
 *
 * @param format  an explicit format name, as `--format` gives it; empty detects
 *                from the path. Detection cannot see through a renamed file or
 *                a nested directory, so the name is the way to say what an
 *                input is. An unregistered name is an error rather than a
 *                silent fall-through to YAML.
 */
inline const InputFormat& input_format_for(const std::string& filename,
                                           const std::string& format = "") {
    const auto& registry = input_format_registry();
    if (!format.empty()) {
        if (const InputFormat* named = registry.find(format))
            return *named;
        throw std::runtime_error("Unknown input format '" + format +
                                 "'; expected one of: " + registry.canonical_names(", "));
    }
    if (const InputFormat* detected = detect_input_format(filename))
        return *detected;
    throw std::runtime_error(
        "No registered input format recognises '" + filename +
        "'; the formats this build reads are: " + registry.canonical_names(", "));
}

/// Create the input parser for this path, or for the format `--format` names.
inline std::unique_ptr<InputParser> create_input_parser(const std::string& filename,
                                                        const std::string& format = "") {
    return input_format_for(filename, format).create_parser();
}

/**
 * What an argument-less invocation should read, or empty if nothing.
 *
 * Some formats are run with no arguments from the directory that holds the
 * input, so an entry point given no input path asks the registered formats
 * whether one claims the working directory. Every entry point shares this
 * rule rather than each inventing its own, and a missing argument stays an
 * error when no format claims the directory.
 */
inline std::string resolve_argument_less_input() {
    return format_claiming_directory(".") != nullptr ? "." : std::string{};
}

/**
 * Substitute the structure of *input* with the ESSE material in *material_file*
 * (`--material <file>`): its cell and positions replace the input's, as a YAML
 * input's `material` key would. Everything else — settings, species, the order
 * of the atoms and their constraints — stays the input's, so a pw.x deck runs at
 * a new geometry unchanged. The cell keeps the input's lattice parameter, so its
 * alat-based units stay fixed across a relaxation's steps, as in pw.x.
 */
inline void substitute_material(InputData& input, const std::string& material_file) {
    InputData material;
    YAMLInputParser::parse_material(detail::read_json_file(material_file), material);
    if (material.atoms.size() != input.atoms.size())
        throw std::runtime_error("--material " + material_file + " has " +
                                 std::to_string(material.atoms.size()) + " atoms; the input has " +
                                 std::to_string(input.atoms.size()));
    if (!material.has_explicit_lattice_vectors)
        throw std::runtime_error("--material " + material_file + " must give lattice.vectors");
    Matrix3x3 lattice_vectors_bohr;
    for (int row = 0; row < 3; ++row)
        lattice_vectors_bohr[row] = material.lattice_vectors[row] * material.cell_dimension_1;
    input.set_free_lattice_vectors(lattice_vectors_bohr, input.cell_dimension_1);
    for (size_t atom_index = 0; atom_index < input.atoms.size(); ++atom_index)
        input.atoms[atom_index].position = material.atoms[atom_index].position;
}

/**
 * Parse input using the given or auto-detected format.
 *
 * A relaxation input ('relax', 'vc-relax') runs here as one SCF at its own
 * geometry, logged as such: a relaxation is driven from Python, one SCF per
 * geometry, on whichever binary it is given.
 *
 * Every entry point reads its input through here, so this is also where an
 * input that names no exchange-correlation functional gets the one its
 * pseudopotentials state. The InputData returned names the functional the run
 * will use, which `--check`, the provenance record and the SCF all read. It
 * stays empty only for an input missing a dataset, which cannot run.
 *
 * The name is lower-cased here, whichever reader or dataset supplied it and
 * however the input spelled it: `pbe`, the spelling the example inputs and
 * ESSE use. The readers keep what they read, so this is the one place the
 * convention is applied.
 *
 * The format's canonical name is recorded as InputData::source_format, so the
 * finished run can be handed back to that same format for any companion
 * outputs its conventions expect (io/output/companion_outputs.hpp).
 */
inline InputData parse_input(const std::string& filename, const std::string& format = "") {
    const InputFormat& input_format = input_format_for(filename, format);
    InputData input = input_format.create_parser()->parse(filename);
    input.source_format = input_format.names().front();
    if (input.calculation == "relaxation" || input.calculation == "variable_cell_relaxation") {
        log::info("     Calculation '{}' runs as one SCF at this geometry, starting from a saved "
                  "density when one fits: a step of the relaxation q3.workflows."
                  "run_relax_workflow / run_vc_relax_workflow drives",
                  input.calculation);
        input.calculation = "self_consistent_field";
        input.start_from_saved_charge_density = true;
        input.is_relaxation_step = true;
    }
    if (input.exchange_correlation_functional.empty())
        input.exchange_correlation_functional =
            functional_from_pseudopotentials(input.species, input.pseudopotential_directory);
    for (auto& character : input.exchange_correlation_functional)
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    return input;
}

} // namespace q3
