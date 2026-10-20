#pragma once

/**
 * BinaryRecordFormat — abstract interface for binary record I/O formats.
 *
 * Defines the contract for writing and reading framed binary records.
 * Concrete implementations handle the specific byte-level framing:
 *   - QE: Fortran unformatted sequential [4-byte len][data][4-byte len]
 *   - Q3: raw sequential (no framing, just raw bytes)
 *
 * Use create_binary_record_format() from factory.hpp to obtain an instance.
 */

#include <cstdint>
#include <fstream>

namespace q3 {

class BinaryRecordFormat {
  public:
    virtual ~BinaryRecordFormat() = default;

    /**
     * Write a single framed record to the output stream.
     */
    virtual void
    write_record(std::ofstream& output_file, const void* data, int32_t number_of_bytes) const = 0;

    /**
     * Read a single framed record from the input stream.
     */
    virtual void
    read_record(std::ifstream& input_file, void* data, int32_t number_of_bytes) const = 0;
};

} // namespace q3
