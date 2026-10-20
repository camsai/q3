#pragma once

/**
 * VolumetricFieldWriter — abstract writer for one volumetric file format.
 *
 * Every format Q3 can put a real-space field into implements this: Gaussian
 * cube, XCrySDen XSF, and any a build-time add-on registers. The caller builds
 * a VolumetricField once and
 * asks the factory for a writer by name, so adding a format never touches
 * the code that produced the field.
 */

#include <string>

#include "io/volumetric/field.hpp"

namespace q3 {

class VolumetricFieldWriter {
  public:
    virtual ~VolumetricFieldWriter() = default;

    /// Write `field` to `path`, throwing std::runtime_error if it cannot be written.
    virtual void write(const std::string& path, const VolumetricField& field) const = 0;

    /**
     * What a file of this format is conventionally called, given a run's
     * prefix and the field's name. Cube and XSF take a suffix; a format whose
     * file has a fixed name says so by returning that name unchanged.
     */
    [[nodiscard]] virtual std::string default_file_name(const std::string& stem) const = 0;
};

} // namespace q3
