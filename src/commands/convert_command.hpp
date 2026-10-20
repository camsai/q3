#pragma once

/**
 * ConvertCommand — `q3_utility convert`: a saved record into an external format,
 * or an external format into a saved record.
 *
 * Two records live in a run's `<prefix>.save/` directory today, and each has
 * its own formats: the mean-field record (`mean-field.q3mf`, §4.8 of the IO
 * reference) exports to BerkeleyGW, the interpolation record
 * (`interpolation.q3ip`, §4.9) exports to the Wannier90 file set. Converting
 * after the fact drives exactly the writer code the in-process export uses,
 * so a second format, or a re-export after a format fix, costs a file read
 * instead of a re-run -- Q3's analogue of Quantum ESPRESSO's `pw2bgw.x` and
 * `pw2wannier90.x` in one place.
 *
 *   q3_utility convert <save-directory> [--format berkeleygw|wannier90] [--outdir <dir>]
 *   q3_utility convert --from wannier90 --seedname <s> --grid nx ny nz [--outdir <dir>] [--prefix
 * <p>]
 *
 * With no `--format`, the one format the directory's record allows is
 * taken; a directory holding both records needs the flag. The import
 * direction turns another code's `.amn`/`.eig`/`.nnkp` into an interpolation
 * record that `q3_utility interpolate` then evaluates like a native one, which
 * is how the engine is checked against Quantum ESPRESSO's matrices.
 */

#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/interpolation/checkpoint.hpp"
#include "io/interpolation/factory.hpp"
#include "io/interpolation/record_and_export.hpp"
#include "io/interpolation/wannier90/importer.hpp"
#include "io/logger.hpp"
#include "io/mean_field_export/factory.hpp"
#include "io/mean_field_record/checkpoint.hpp"

namespace q3 {

class ConvertCommand {
  public:
    static void print_usage() {
        log::error(
            "Usage: q3_utility convert <save-directory> [--format <name>] [--outdir <directory>]");
        log::error(
            "       q3_utility convert --from wannier90 --seedname <s> --grid <nx> <ny> <nz> "
            "[--outdir <directory>] [--prefix <p>]");
        log::error("  <save-directory>  a run's <prefix>.save directory, holding mean-field.q3mf "
                   "and/or interpolation.q3ip");
        log::error("  --format          berkeleygw (from the mean-field record) or wannier90 (from "
                   "the interpolation record);");
        log::error("                    defaults to the one format the directory's record allows");
        log::error("  --outdir          where to write (default: the save directory's parent, or . "
                   "for an import)");
        log::error("  --from wannier90  import <seedname>.amn/.eig/.nnkp on --grid into "
                   "<outdir>/<prefix>.save/interpolation.q3ip");
        log::error("  --prefix          the run prefix an import is filed under (default: the "
                   "seedname's basename)");
    }

    /// Returns the process exit code; argument errors print usage, everything else throws.
    static int run(const std::vector<std::string>& arguments) {
        std::string save_directory;
        std::string format;
        std::string output_directory;
        std::string from;
        std::string seedname;
        std::string prefix;
        std::array<int, 3> grid = {0, 0, 0};
        for (std::size_t index = 0; index < arguments.size(); index++) {
            const std::string& argument = arguments[index];
            auto take_value = [&](const char* flag) -> std::string {
                if (index + 1 >= arguments.size()) {
                    throw std::runtime_error(std::string(flag) + " needs a value");
                }
                return arguments[++index];
            };
            if (argument == "--format") {
                format = take_value("--format");
            } else if (argument == "--outdir") {
                output_directory = take_value("--outdir");
            } else if (argument == "--from") {
                from = take_value("--from");
            } else if (argument == "--seedname") {
                seedname = take_value("--seedname");
            } else if (argument == "--prefix") {
                prefix = take_value("--prefix");
            } else if (argument == "--grid") {
                for (int axis = 0; axis < 3; axis++) {
                    grid[axis] = parse_positive_integer("--grid", take_value("--grid"));
                }
            } else if (argument == "-h" || argument == "--help") {
                print_usage();
                return 0;
            } else if (argument.rfind("-", 0) == 0) {
                log::error("Unknown option '{}'", argument);
                print_usage();
                return 1;
            } else if (save_directory.empty()) {
                save_directory = argument;
            } else {
                log::error("Unexpected extra argument '{}'", argument);
                print_usage();
                return 1;
            }
        }

        if (!from.empty()) {
            return import_record(from, seedname, grid, output_directory, prefix);
        }
        if (save_directory.empty()) {
            print_usage();
            return 1;
        }
        return export_record(save_directory, format, output_directory);
    }

  private:
    static int parse_positive_integer(const char* flag, const std::string& text) {
        std::size_t consumed = 0;
        int value = 0;
        try {
            value = std::stoi(text, &consumed);
        } catch (const std::exception&) {
            throw std::runtime_error(std::string(flag) + " expects positive integers, got '" +
                                     text + "'");
        }
        if (consumed != text.size() || value <= 0) {
            throw std::runtime_error(std::string(flag) + " expects positive integers, got '" +
                                     text + "'");
        }
        return value;
    }

    static int import_record(const std::string& from,
                             const std::string& seedname,
                             const std::array<int, 3>& grid,
                             const std::string& output_directory,
                             const std::string& prefix) {
        if (from != "wannier90") {
            throw std::runtime_error("--from supports only wannier90, not '" + from + "'");
        }
        if (seedname.empty() || grid[0] <= 0) {
            throw std::runtime_error(
                "--from wannier90 needs --seedname <s> (the files are <s>.amn/.eig/.nnkp) and "
                "--grid nx ny nz (a .nnkp lists k-points without saying what grid they came "
                "from)");
        }
        const InterpolationRecord record = import_wannier90_files(seedname, grid);
        const std::string prefix_name =
            prefix.empty() ? std::filesystem::path(seedname).filename().string() : prefix;
        const std::string directory =
            output_directory.empty() ? std::string(".") : output_directory;
        std::filesystem::create_directories(directory + "/" + prefix_name + ".save");
        const std::string record_path = interpolation_record_path(directory, prefix_name);
        write_interpolation_record(record, record_path);
        log::info("Imported {} k-points, {} bands, {} Wannier functions from {}.amn/.eig/.nnkp "
                  "into {}",
                  record.wannier.number_of_kpoints(),
                  record.wannier.number_of_bands,
                  record.wannier.number_of_trial_orbitals,
                  seedname,
                  record_path);
        return 0;
    }

    static int export_record(const std::string& save_directory,
                             std::string format,
                             std::string output_directory) {
        /**
         * `out/si.save/` (what shell completion produces) and `.` must name
         * the same directory as `out/si.save`: with a trailing separator,
         * filename() is empty and parent_path() is the save directory
         * itself, which would put the export inside it and hand the prefix
         * logic below the wrong path component.
         */
        std::filesystem::path save_path =
            std::filesystem::absolute(save_directory).lexically_normal();
        if (save_path.filename().empty()) {
            save_path = save_path.parent_path();
        }
        const std::filesystem::path mean_field_path = save_path / "mean-field.q3mf";
        const std::filesystem::path interpolation_path = save_path / "interpolation.q3ip";
        const bool has_mean_field = std::filesystem::exists(mean_field_path);
        const bool has_interpolation = std::filesystem::exists(interpolation_path);

        if (format.empty()) {
            if (has_mean_field && has_interpolation) {
                throw std::runtime_error(
                    "'" + save_directory +
                    "' holds both a mean-field and an interpolation record; say which to "
                    "convert with --format berkeleygw or --format wannier90.");
            }
            if (has_mean_field) {
                format = "berkeleygw";
            } else if (has_interpolation) {
                format = "wannier90";
            } else {
                /**
                 * Name the likely cause rather than just the missing files: each
                 * record is written only by a run that asked for it.
                 */
                throw std::runtime_error(
                    "No record in '" + save_directory +
                    "': neither mean-field.q3mf (written by a "
                    "run with control.mean_field_export set to something other than 'none') nor "
                    "interpolation.q3ip (written by an NSCF run with a control.interpolation "
                    "block). Re-run that calculation with the export enabled to produce one.");
            }
        }

        if (output_directory.empty()) {
            const std::filesystem::path parent = save_path.parent_path();
            output_directory = parent.empty() ? std::string(".") : parent.string();
        }

        if (format == "berkeleygw") {
            if (!has_mean_field) {
                throw std::runtime_error("No mean-field record at '" + mean_field_path.string() +
                                         "'. It is written only by a run that set "
                                         "control.mean_field_export to something other than "
                                         "'none'; re-run that calculation with the export "
                                         "enabled to produce it.");
            }
            const MeanFieldRecord record = read_mean_field_record(mean_field_path.string());
            log::info("Read mean-field record: {} k-points, {} bands, {} G-vectors",
                      record.number_of_kpoints(),
                      record.number_of_bands,
                      record.number_of_g_vectors());
            create_mean_field_exporter(format)->export_to_directory(record, output_directory);
            log::info("Converted to '{}' in {}", format, output_directory);
            return 0;
        }
        if (format == "wannier90") {
            if (!has_interpolation) {
                throw std::runtime_error("No interpolation record at '" +
                                         interpolation_path.string() +
                                         "'. It is written by an NSCF run with a "
                                         "control.interpolation block; re-run that NSCF, or "
                                         "import Wannier90 files with --from wannier90.");
            }
            const InterpolationRecord record =
                read_interpolation_record(interpolation_path.string());
            /// `<outdir>/<prefix>.save` -> the run's prefix, which names the exported files.
            std::string prefix_name = save_path.filename().string();
            const std::string suffix = ".save";
            if (prefix_name.size() > suffix.size() &&
                prefix_name.compare(prefix_name.size() - suffix.size(), suffix.size(), suffix) ==
                    0) {
                prefix_name.erase(prefix_name.size() - suffix.size());
            }
            log::info("Read interpolation record ({}): {} k-points, {} bands",
                      record.method,
                      record.wannier.number_of_kpoints(),
                      record.wannier.number_of_bands);
            create_interpolation_exporter(format)->export_to_directory(
                record, output_directory, prefix_name);
            log::info("Converted to '{}' in {}", format, output_directory);
            return 0;
        }
        throw std::runtime_error("Unknown format '" + format +
                                 "'. Supported: berkeleygw (from the mean-field record), "
                                 "wannier90 (from the interpolation record)");
    }
};

} // namespace q3
