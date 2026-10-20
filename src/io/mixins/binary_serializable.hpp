#pragma once

/**
 * Mixin providing binary serialization with file output.
 *
 * Classes inheriting this mixin must implement write_to_file()
 * to handle the actual binary writing.
 */

#include <string>

namespace q3 {

class BinarySerializable {
  public:
    virtual ~BinarySerializable() = default;

    /// Write binary data to a file or directory.
    virtual void write_to_file(const std::string& path) const = 0;
};

} // namespace q3
