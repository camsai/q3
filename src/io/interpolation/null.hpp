#pragma once

/**
 * NullInterpolationExporter — the "none" format: writes nothing, so a call
 * site can export unconditionally and never branch on whether an export was
 * requested.
 */

#include <string>

#include "io/interpolation/init.hpp"

namespace q3 {

class NullInterpolationExporter : public InterpolationExporter {
  public:
    void export_to_directory(const InterpolationRecord& /*record*/,
                             const std::string& /*directory*/,
                             const std::string& /*prefix*/) const override {}

    [[nodiscard]] std::string name() const override {
        return "none";
    }
};

} // namespace q3
