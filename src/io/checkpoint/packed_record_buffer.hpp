#pragma once

/**
 * PackedRecordBuffer — byte buffer for a Fortran unformatted record whose
 * fields are mixed-width scalars.
 *
 * A Fortran WRITE of a list of scalars lays them end to end with no
 * alignment padding; a C struct with the same members does not. Appending
 * each field's bytes here reproduces the packed layout exactly, independent
 * of compiler alignment rules, and never exposes uninitialized bytes.
 */

#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

namespace q3 {

class PackedRecordBuffer {
  public:
    template <typename Scalar> void append(Scalar value) {
        static_assert(std::is_trivially_copyable<Scalar>::value,
                      "PackedRecordBuffer appends raw bytes of trivially copyable scalars");
        const std::size_t offset = bytes_.size();
        bytes_.resize(offset + sizeof(Scalar));
        std::memcpy(bytes_.data() + offset, &value, sizeof(Scalar));
    }

    [[nodiscard]] const char* data() const {
        return bytes_.data();
    }
    [[nodiscard]] std::size_t size() const {
        return bytes_.size();
    }

  private:
    std::vector<char> bytes_;
};

} // namespace q3
