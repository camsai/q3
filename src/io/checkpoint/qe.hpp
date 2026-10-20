#pragma once

/**
 * QEBinaryRecordFormat — Fortran unformatted sequential record I/O.
 *
 * Each record is framed as: [4-byte length prefix][data bytes][4-byte length suffix].
 * Matches Quantum ESPRESSO's Fortran I/O (FORM='UNFORMATTED', ACCESS='SEQUENTIAL').
 */

#include "io/checkpoint/init.hpp"

namespace q3 {

class QEBinaryRecordFormat : public BinaryRecordFormat {
  public:
    void write_record(std::ofstream& output_file,
                      const void* data,
                      int32_t number_of_bytes) const override {
        output_file.write(reinterpret_cast<const char*>(&number_of_bytes), 4);
        output_file.write(reinterpret_cast<const char*>(data), number_of_bytes);
        output_file.write(reinterpret_cast<const char*>(&number_of_bytes), 4);
    }

    void
    read_record(std::ifstream& input_file, void* data, int32_t number_of_bytes) const override {
        int32_t prefix_length;
        int32_t suffix_length;
        input_file.read(reinterpret_cast<char*>(&prefix_length), 4);
        if (prefix_length != number_of_bytes) {
            input_file.setstate(std::ios::failbit);
            return;
        }
        input_file.read(reinterpret_cast<char*>(data), number_of_bytes);
        input_file.read(reinterpret_cast<char*>(&suffix_length), 4);
        if (suffix_length != prefix_length) {
            input_file.setstate(std::ios::failbit);
        }
    }
};

} // namespace q3
