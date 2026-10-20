#pragma once

/**
 * MeanFieldExporter — abstract interface for mean-field data export formats.
 *
 * A "mean-field export" writes the converged Kohn-Sham wavefunctions, charge
 * density, and (where supported) exchange-correlation potential to disk in a
 * format consumed by an external many-body-perturbation-theory code (e.g.
 * BerkeleyGW's Sigma/epsilon codes) rather than by Q3 itself. Concrete
 * formats (BerkeleyGW today; other MBPT front ends are the natural future
 * additions) implement this interface and are selected by name through
 * create_mean_field_exporter() (factory.hpp), so orchestrators never branch
 * on the format themselves -- "none" resolves to NullMeanFieldExporter
 * (null.hpp), which is a silent no-op, removing the need for any if/else at
 * the call sites (see layer_4_computation/flavors/{self_consistent_field,
 * non_self_consistent_field}.hpp, both of which call export_to_directory()
 * unconditionally).
 */

#include <string>

#include "io/mean_field_record/init.hpp"

namespace q3 {

class MeanFieldExporter {
  public:
    virtual ~MeanFieldExporter() = default;

    /**
     * Write this format's mean-field data files for `record` into `directory`
     * (created if it does not already exist). A no-op for the Null exporter.
     *
     * Taking a MeanFieldRecord rather than a live SCF state is what lets the
     * same writer serve both the in-process export and the standalone
     * converter -- see io/mean_field_record/init.hpp.
     */
    virtual void export_to_directory(const MeanFieldRecord& record,
                                     const std::string& directory) const = 0;

    /// Format name as accepted by the input ("none", "berkeleygw", ...).
    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace q3
