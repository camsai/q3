#pragma once

/**
 * Shared CLI helpers for the q3 and q3_mpi entry points.
 *
 * Parsing does NOT apply side effects (e.g. log level): it only records the
 * requested values. Entry points apply them after their logger is initialized,
 * so that a later log::init() / log::init_mpi() cannot silently discard a
 * --log-level request.
 */

#include "io/input/parser/factory.hpp"
#include "io/input/parser/input_format_registry.hpp"
#include "io/logger.hpp"
#include "layer_0_core/addon_identity.hpp"
#include "layer_0_core/version.hpp"
#include "utils/time.hpp"

#include <cstdlib>
#include <string>

namespace q3 {
namespace cli {

/**
 * One line naming the build: version, and the commit if the build supplied one.
 *
 * Assembled here rather than at each entry point so the five binaries cannot
 * drift into reporting themselves differently. The commit is empty for a build
 * from a dirty tree or from no repository at all -- see version.hpp for why
 * that is reported as unknown rather than guessed.
 */
inline std::string build_identity(const std::string& binary_name) {
    std::string identity = binary_name + " " + std::string(q3::version);
    if (std::string(q3::commit).empty()) {
        return identity + " (commit unknown: built from a modified tree or without git)";
    }
    return identity + " (" + std::string(q3::commit) + ")";
}

/**
 * Print what `--version` reports: the build identity on its own first line,
 * which scripts parse, then the add-ons compiled in, when there are any.
 */
inline void print_version(const std::string& binary_name) {
    log::info("{}", build_identity(binary_name));
    if (active_addons().empty())
        return;
    std::string addons;
    for (const AddonIdentity& addon : active_addons())
        addons += (addons.empty() ? "" : ", ") + std::string(addon.name) + " " + addon.version;
    log::info("add-ons: {}", addons);
}

/// The `--format` values every registered input format accepts, as usage text: `<yaml|qe|...>`.
inline std::string input_format_choices() {
    return "<" + input_format_registry().canonical_names("|") + ">";
}

/// What a usage line says an input path may name: every registered format, in order.
inline std::string input_format_descriptions() {
    const auto& formats = input_format_registry().entries();
    std::string descriptions;
    for (std::size_t index = 0; index < formats.size(); index++) {
        if (index > 0)
            descriptions += index + 1 == formats.size() ? ", or " : ", ";
        descriptions += formats[index]->display_name();
    }
    return descriptions;
}

struct Options {
    /**
     * The input path. Empty when none was given, which is not an error when a
     * registered input format claims the working directory: some formats are
     * run with no arguments from the directory holding the input. The entry
     * point decides what an empty value means.
     */
    std::string input_file;
    /**
     * Explicit input format from `--format`, e.g. "qe"; empty means detect.
     *
     * Detection is by file name, extension or directory contents, so an input
     * renamed by a workflow tool, or one in a directory named something
     * unexpected, needs a way to say what it is.
     */
    std::string format;
    /// Requested log level ("trace" ... "off"); empty means keep the default.
    std::string log_level;
    /// Number of k-point pools (MPI binary only). 0 = execution policy default.
    int number_of_kpoint_pools = 0;
    /// Whether `--pools` was present, so an input-file value can be honored otherwise.
    bool number_of_kpoint_pools_was_set = false;
    /**
     * Divide the bands of a pool across its ranks instead of its reciprocal-space
     * coefficients (MPI binary only). Each rank then keeps whole coefficient
     * vectors, so its transforms are local and serial, and gamma-only, stress and
     * export stay available -- all of which the slab split gives up.
     */
    bool distribute_bands = false;
    /**
     * Requested precision mode (GPU binaries only): "double" or "mixed".
     * Empty means the binary's default. Validated by the GPU entry point.
     */
    std::string precision;
    /**
     * `--check`: read the input, report what it resolved to, and exit without
     * running anything. For a drop-in replacement the first question a user has
     * about someone else's deck is "will this even be read, and as what?", and
     * answering it should not cost an SCF.
     */
    bool check_only = false;
    /**
     * `--log-file <path>`: also write the log to a file. Empty means console
     * only. Under MPI each non-root rank writes to `<path>.<rank>`, since the
     * console carries rank 0 alone and a rank that fails on its own has
     * nowhere else to say so.
     */
    std::string log_file;
    /**
     * `--version`: print the build identity and exit. Separate from the run
     * because the answer must not depend on having a valid input: the question
     * is asked precisely when a result is in doubt, and by scripts recording
     * what produced one.
     */
    bool print_version = false;
    /**
     * `--material <file>`: run the input as one geometry step of a relaxation,
     * an SCF at the cell and positions of this ESSE material (see
     * run_at_material). How q3.workflows relaxes any input on any binary.
     */
    std::string material_file;
};

/**
 * The usage every binary prints: one text, so the binaries cannot drift apart.
 * A flag a binary has no use for is read and ignored there, as documented in
 * docs/io_reference.md.
 */
inline void print_usage(const std::string& binary_name) {
    log::error("Usage: {} [input] [--format {}] [--material <file>] [--check]",
               binary_name,
               input_format_choices());
    log::error("       [--pools <n>] [--distribute-bands] [--precision double|mixed]");
    log::error("       [--log-level <level>] [--log-file <path>] [--version]");
    log::error("  input               {}", input_format_descriptions());
    log::error("  --material <file>   run the input at this ESSE structure (cell, positions)");
    log::error("  --check             read the input, report it, and run nothing");
    log::error("  --pools <n>         k-point pools (MPI binaries)");
    log::error("  --distribute-bands  split bands, not reciprocal space, in a pool (MPI)");
    log::error("  --precision <mode>  double or mixed (GPU binaries)");
    log::error("  --log-file <path>   also log to a file (MPI rank N writes <path>.N)");
    log::error("  --version           print the build identity and exit");
}

/**
 * Parse command-line options shared by the q3, q3_mpi, and q3_gpu* binaries.
 *
 * Recognized flags: --log-level <level>, --log-file <path>, --pools <n>, --distribute-bands,
 * --precision <mode>, --format <name>, --material <file>, --check, --version. The input path is
 * the first argument that is not a flag or
 * a flag's value, so it may appear before or after the flags — `q3 --format
 * qe scf.in` and `q3 scf.in --format qe` both work, and so does giving no path
 * at all.
 *
 * Leaving the path empty is **not** an error here. Only the entry point knows
 * whether an argument-less invocation is meaningful: it is when a registered
 * input format claims the working directory (resolve_argument_less_input()).
 *
 * @param argc         Argument count from main().
 * @param argv         Argument vector from main().
 * @param binary_name  Named in the usage text, printed and followed by exit(1)
 *                     only for a malformed flag.
 */
inline Options parse_options(int argc, char* argv[], const std::string& binary_name) {
    Options options;

    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        std::string argument(argv[argument_index]);

        auto take_value = [&](const char* flag_name) -> std::string {
            if (argument_index + 1 >= argc) {
                log::error("{} needs a value", flag_name);
                print_usage(binary_name);
                std::exit(1);
            }
            return argv[++argument_index];
        };

        if (argument == "--log-level") {
            options.log_level = take_value("--log-level");
        } else if (argument == "--log-file") {
            options.log_file = take_value("--log-file");
        } else if (argument == "--version" || argument == "-v") {
            options.print_version = true;
        } else if (argument == "--pools") {
            options.number_of_kpoint_pools = std::stoi(take_value("--pools"));
            options.number_of_kpoint_pools_was_set = true;
        } else if (argument == "--distribute-bands") {
            options.distribute_bands = true;
        } else if (argument == "--precision") {
            options.precision = take_value("--precision");
        } else if (argument == "--format") {
            options.format = take_value("--format");
        } else if (argument == "--material") {
            options.material_file = take_value("--material");
        } else if (argument == "--check") {
            options.check_only = true;
        } else if (argument.rfind("--", 0) == 0) {
            /**
             * Refused rather than ignored. Silently swallowing it made
             * `q3 --typo deck/` take "--typo" as the input path and discard
             * the directory, which fails much further from the cause.
             */
            log::error("Unrecognized option {}", argument);
            print_usage(binary_name);
            std::exit(1);
        } else if (options.input_file.empty()) {
            options.input_file = argument;
        } else {
            log::error(
                "Unexpected extra argument {} (input is already {})", argument, options.input_file);
            print_usage(binary_name);
            std::exit(1);
        }
    }

    return options;
}

/**
 * Read the input every binary runs, the same way in each: resolve an omitted
 * path (the working directory, when a registered input format claims it),
 * parse it, substitute `--material`'s structure, and apply the input's
 * verbosity.
 *
 * Throws, after printing the usage, when no input was given and none is found.
 */
inline InputData read_input(Options& options, const std::string& binary_name) {
    if (options.input_file.empty()) {
        const InputFormat* claiming_format = format_claiming_directory(".");
        if (claiming_format == nullptr) {
            print_usage(binary_name);
            throw std::runtime_error(
                "no input given, and no input format claims the working directory");
        }
        options.input_file = ".";
        log::info("No input given; found {} in the current directory",
                  claiming_format->display_name());
    }
    InputData input = parse_input(options.input_file, options.format);
    if (!options.material_file.empty())
        substitute_material(input, options.material_file);
    log::apply_verbosity(input.verbosity, options.log_level);
    return input;
}

/**
 * Print an 80-column banner with a centered, timestamped label.
 */
inline void print_timestamped_banner(const std::string& label_prefix) {
    std::string divider_line(80, '=');
    std::string label = label_prefix + get_iso_timestamp_with_timezone();
    int left_padding = (78 - static_cast<int>(label.length())) / 2;
    int right_padding = 78 - static_cast<int>(label.length()) - left_padding;
    std::string banner_line =
        "=" + std::string(left_padding, ' ') + label + std::string(right_padding, ' ') + "=";

    log::info("{}", divider_line);
    log::info("{}", banner_line);
    log::info("{}", divider_line);
}

inline void print_header() {
    print_timestamped_banner("BEGIN: ");
}

inline void print_footer() {
    print_timestamped_banner("END: ");
}

} // namespace cli
} // namespace q3
