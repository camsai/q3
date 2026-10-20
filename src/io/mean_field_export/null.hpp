#pragma once

/**
 * NullMeanFieldExporter — the "no export" format.
 *
 * Null Object implementation of MeanFieldExporter: export_to_directory() is a
 * no-op (no files are created). Selected by
 * create_mean_field_exporter("none") so the SCF/NSCF flavors can call the
 * export interface unconditionally instead of branching on the format name.
 */

#include <string>

#include "io/mean_field_export/init.hpp"

namespace q3 {

class NullMeanFieldExporter : public MeanFieldExporter {
  public:
    void export_to_directory(const MeanFieldRecord& /*record*/,
                             const std::string& /*directory*/) const override {}

    [[nodiscard]] std::string name() const override {
        return "none";
    }
};

} // namespace q3
