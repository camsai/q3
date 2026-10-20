#pragma once

/**
 * write_and_export_interpolation_record() — the single call the NSCF flavor
 * makes once it has built a record.
 *
 * Two things, in this order: the native record goes to
 * `<outdir>/<prefix>.save/interpolation.q3ip`, always, because it is what
 * `q3_utility interpolate` evaluates and what `q3_utility convert` re-exports; then
 * each format `control.interpolation.export_formats` lists is written beside
 * it as `<outdir>/<prefix>.*`. Unlike the mean-field export, the record is
 * not an optional by-product here -- it IS the output of the step -- so a
 * failure to write it propagates instead of being logged.
 */

#include <filesystem>
#include <string>

#include "io/input/data.hpp"
#include "io/interpolation/checkpoint.hpp"
#include "io/interpolation/factory.hpp"
#include "io/logger.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"

namespace q3 {

/// Where a run with `prefix` in `output_directory` keeps its interpolation record.
[[nodiscard]] inline std::string interpolation_record_path(const std::string& output_directory,
                                                           const std::string& prefix) {
    return output_directory + "/" + prefix + ".save/interpolation.q3ip";
}

inline void write_and_export_interpolation_record(const InterpolationRecord& record,
                                                  const InputData& input) {
    std::filesystem::create_directories(input.output_directory + "/" + input.prefix + ".save");
    const std::string record_path = interpolation_record_path(input.output_directory, input.prefix);
    write_interpolation_record(record, record_path);
    log::info("     Wrote interpolation record ({}): {}", record.method, record_path);

    for (const std::string& format : input.interpolation.export_formats) {
        create_interpolation_exporter(format)->export_to_directory(
            record, input.output_directory, input.prefix);
    }
}

} // namespace q3
