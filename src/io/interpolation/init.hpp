#pragma once

/**
 * InterpolationExporter — abstract interface for writing an
 * InterpolationRecord in an external code's format.
 *
 * The native record (`<prefix>.save/interpolation.q3ip`, checkpoint.hpp) is
 * what Q3 itself reads back. An export is for another program: today the
 * Wannier90 file set, so `wannier90.x` can localize what Q3 built and the
 * tools built on Wannier90 can consume it; other tight-binding and
 * interpolation front ends are the natural future additions. Concrete
 * formats implement this interface and are selected by name through
 * create_interpolation_exporter() (factory.hpp), so the code that writes a
 * record never branches on the format -- "none" resolves to the Null
 * exporter, a silent no-op. Same shape as io/mean_field_export/init.hpp,
 * for the same reason.
 */

#include <string>

#include "layer_4_computation/interpolator/input_data/record.hpp"

namespace q3 {

class InterpolationExporter {
  public:
    virtual ~InterpolationExporter() = default;

    /**
     * Write this format's files for `record` as `<directory>/<prefix>.*`.
     * A no-op for the Null exporter.
     */
    virtual void export_to_directory(const InterpolationRecord& record,
                                     const std::string& directory,
                                     const std::string& prefix) const = 0;

    /// Format name as accepted by the input ("none", "wannier90", ...).
    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace q3
