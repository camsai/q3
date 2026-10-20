#pragma once

/**
 * create_interpolation_exporter — resolve an export format name to its
 * InterpolationExporter. "none" and the empty string are the Null object, so
 * a call site exports unconditionally; an unknown name is refused with the
 * list of known ones. Mirrors create_mean_field_exporter().
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "io/interpolation/init.hpp"
#include "io/interpolation/null.hpp"
#include "io/interpolation/wannier90/exporter.hpp"

namespace q3 {

[[nodiscard]] inline std::unique_ptr<InterpolationExporter>
create_interpolation_exporter(const std::string& format) {
    if (format.empty() || format == "none") {
        return std::make_unique<NullInterpolationExporter>();
    }
    if (format == "wannier90") {
        return std::make_unique<Wannier90Exporter>();
    }
    throw std::runtime_error("Unknown interpolation export format '" + format +
                             "'. Supported: none, wannier90");
}

} // namespace q3
