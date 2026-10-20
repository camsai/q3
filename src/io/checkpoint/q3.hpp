#pragma once

/**
 * Q3BinaryRecordFormat — raw sequential binary record I/O.
 *
 * No record framing — data is written as raw bytes. Used by Q3's
 * native checkpoint format for compact, fast serialization.
 */

#include "io/checkpoint/init.hpp"

namespace q3 {

class Q3BinaryRecordFormat : public BinaryRecordFormat {
  public:
    void write_record(std::ofstream& output_file,
                      const void* data,
                      int32_t number_of_bytes) const override {
        output_file.write(reinterpret_cast<const char*>(data), number_of_bytes);
    }

    void
    read_record(std::ifstream& input_file, void* data, int32_t number_of_bytes) const override {
        input_file.read(reinterpret_cast<char*>(data), number_of_bytes);
    }
};

} // namespace q3
