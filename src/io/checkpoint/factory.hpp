#pragma once

/**
 * create_binary_record_format — factory for binary record I/O formats.
 *
 * Supported: "qe" (Fortran unformatted), "q3" (raw sequential).
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "io/checkpoint/init.hpp"
#include "io/checkpoint/q3.hpp"
#include "io/checkpoint/qe.hpp"

namespace q3 {

/// Create a binary record format by name.
inline std::unique_ptr<BinaryRecordFormat>
create_binary_record_format(const std::string& format_name = "qe") {
    if (format_name == "qe") {
        return std::make_unique<QEBinaryRecordFormat>();
    }
    if (format_name == "q3") {
        return std::make_unique<Q3BinaryRecordFormat>();
    }
    throw std::runtime_error("Unknown binary record format: " + format_name);
}

} // namespace q3
