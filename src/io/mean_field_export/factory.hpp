#pragma once

/**
 * Factory for MeanFieldExporter implementations.
 *
 * Resolves the input's export format name (InputData::mean_field_export /
 * YAML `control.mean_field_export`) to a concrete exporter, mirroring
 * create_isolated_system_correction_calculator(). "none" (or empty) resolves
 * to NullMeanFieldExporter so call sites need no branching; unknown names
 * fail loudly rather than silently skipping the export the user asked for
 * (the YAML parser already validates the value, so a throw here means a
 * programmatic caller bypassed parsing). Adding a format = one new class
 * implementing MeanFieldExporter + one case here.
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "io/mean_field_export/berkeleygw/init.hpp"
#include "io/mean_field_export/init.hpp"
#include "io/mean_field_export/null.hpp"

namespace q3 {

[[nodiscard]] inline std::unique_ptr<MeanFieldExporter>
create_mean_field_exporter(const std::string& mean_field_export) {
    if (mean_field_export.empty() || mean_field_export == "none") {
        return std::make_unique<NullMeanFieldExporter>();
    }
    if (mean_field_export == "berkeleygw") {
        return std::make_unique<BerkeleyGWExporter>();
    }
    throw std::runtime_error("Unknown mean_field_export '" + mean_field_export +
                             "'. Supported: none, berkeleygw");
}

} // namespace q3
