/**
 * q3_utility — Q3's utility binary: subcommands that consume a finished run's
 * saved records rather than an input manifest.
 *
 *   q3_utility interpolate <input.yml>    evaluate a saved interpolation record
 *                                      along a k-path and/or on a DOS mesh
 *   q3_utility convert <save-directory>   re-export a saved record (mean-field ->
 *                                      BerkeleyGW, interpolation -> Wannier90),
 *                                      or import Wannier90 files into one (--from)
 *   q3_utility volumetric <input.yml>     a saved density, potential or band in a
 *                                      registered volumetric format (cube, xsf, ...)
 *
 * One binary with subcommands rather than one binary per task. The q3,
 * q3_mpi and q3_gpu* names encode WHERE the engine runs, and a science- or
 * format-facing suffix beside them (the former q3-wannier and q3-convert)
 * mixed two axes on one name. Everything here is serial by construction --
 * dense algebra on a few hundred orbitals, or file conversion -- so it never
 * needs a variant. Each subcommand's implementation is one class under
 * src/commands/; this file only dispatches.
 */

#include <exception>
#include <string>
#include <vector>

#include "commands/convert_command.hpp"
#include "commands/interpolate_command.hpp"
#include "commands/volumetric_command.hpp"
#include "io/addon_registration.hpp"
#include "io/logger.hpp"
#include "io/volumetric/factory.hpp"
#include "layer_0_core/diagnostic_error.hpp"

using namespace q3;

namespace {

void print_usage() {
    log::error("Usage: q3_utility [--log-level <level>] <subcommand> [arguments]");
    log::error("  interpolate <input.yml>    evaluate a saved interpolation record");
    log::error("  convert <save-directory>   re-export a saved record, or import Wannier90 files "
               "(--from wannier90)");
    log::error("  volumetric <input.yml>     a saved density, potential or band as {}",
               volumetric_field_writer_registry().canonical_names(", "));
    log::error("  <subcommand> --help        that subcommand's arguments");
}

} // namespace

int main(int argc, char* argv[]) {
    q3::addons::register_all();
    log::init();
    std::vector<std::string> arguments(argv + 1, argv + argc);
    if (arguments.size() >= 2 && arguments[0] == "--log-level") {
        log::set_level(arguments[1]);
        arguments.erase(arguments.begin(), arguments.begin() + 2);
    }
    if (arguments.empty()) {
        print_usage();
        return 1;
    }
    const std::string subcommand = arguments.front();
    if (subcommand == "-h" || subcommand == "--help") {
        print_usage();
        return 0;
    }
    const std::vector<std::string> rest(arguments.begin() + 1, arguments.end());
    try {
        if (subcommand == "interpolate") {
            return InterpolateCommand::run(rest);
        }
        if (subcommand == "convert") {
            return ConvertCommand::run(rest);
        }
        if (subcommand == "volumetric") {
            return VolumetricCommand::run(rest);
        }
        log::error("Unknown subcommand '{}'", subcommand);
        print_usage();
        return 1;
    } catch (const DiagnosticError& error) {
        /**
         * The prose first, as for any error; then the failure's own lines,
         * the code first, so a driver routes on the code while a reader
         * still gets the sentence.
         */
        log::error("Error: {}", error.what());
        for (const std::string& line : error.diagnostic_lines()) {
            log::error("{}", line);
        }
        return 1;
    } catch (const std::exception& error) {
        log::error("Error: {}", error.what());
        return 1;
    }
}
