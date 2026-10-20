#pragma once

/**
 * Length-prefixed little-endian primitives for Q3's own binary records.
 *
 * Every read is bounded before it allocates: a count is checked against the
 * bytes actually left in the file, so a corrupt or foreign length field is
 * diagnosed as a corrupt file rather than becoming a multi-gigabyte allocation
 * that surfaces as `std::bad_alloc` or libc++'s bare "vector".
 *
 * `io/mean_field_record/checkpoint.hpp` carries its own copy of these, which
 * has already diverged: it returns zero-filled values on a truncated read
 * where these throw. Unifying the two is worth doing and is deliberately not
 * done here, since it changes the behaviour of a record this branch does not
 * otherwise touch.
 */

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3::internal::interpolation_stream {

template <typename ValueType> void write_scalar(std::ostream& stream, const ValueType& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(ValueType));
}

template <typename ValueType> ValueType read_scalar(std::istream& stream) {
    ValueType value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(ValueType));
    if (!stream) {
        throw std::runtime_error("interpolation record: the file ended early; it is truncated.");
    }
    return value;
}

/**
 * A count read from the file, times an element size, must fit in what is
 * left of the file. Without this a corrupt or foreign length field becomes a
 * multi-gigabyte allocation (std::bad_alloc, or libc++'s bare "vector"
 * length_error) instead of the "corrupt" diagnosis every other check gives.
 */
inline void require_bytes_available(std::istream& stream, std::uint64_t bytes) {
    const std::streampos position = stream.tellg();
    stream.seekg(0, std::ios::end);
    const std::streampos end = stream.tellg();
    stream.seekg(position);
    if (position < 0 || end < position || static_cast<std::uint64_t>(end - position) < bytes) {
        throw std::runtime_error("interpolation record: a length field exceeds the bytes left in "
                                 "the file; it is corrupt or truncated.");
    }
}

inline void write_string(std::ostream& stream, const std::string& text) {
    write_scalar<int64_t>(stream, static_cast<int64_t>(text.size()));
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
}

inline std::string read_string(std::istream& stream) {
    const int64_t length = read_scalar<int64_t>(stream);
    if (length < 0 || length > 1024) {
        throw std::runtime_error("interpolation record: implausible string length; the file is "
                                 "corrupt.");
    }
    std::string text(static_cast<std::size_t>(length), '\0');
    stream.read(text.data(), length);
    if (!stream) {
        throw std::runtime_error("interpolation record: the file ended early; it is truncated.");
    }
    return text;
}

inline void write_vector3d(std::ostream& stream, const Vector3D& vector) {
    write_scalar<double>(stream, vector.x);
    write_scalar<double>(stream, vector.y);
    write_scalar<double>(stream, vector.z);
}

inline Vector3D read_vector3d(std::istream& stream) {
    Vector3D vector;
    vector.x = read_scalar<double>(stream);
    vector.y = read_scalar<double>(stream);
    vector.z = read_scalar<double>(stream);
    return vector;
}

inline void write_matrix(std::ostream& stream, const Matrix3x3& matrix) {
    for (int row = 0; row < 3; row++) {
        write_vector3d(stream, matrix[row]);
    }
}

inline Matrix3x3 read_matrix(std::istream& stream) {
    Matrix3x3 matrix;
    for (int row = 0; row < 3; row++) {
        matrix[row] = read_vector3d(stream);
    }
    return matrix;
}

/// Length-prefixed array of trivially copyable values.
template <typename ValueType>
void write_array(std::ostream& stream, const std::vector<ValueType>& values) {
    write_scalar<int64_t>(stream, static_cast<int64_t>(values.size()));
    if (!values.empty()) {
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(ValueType)));
    }
}

template <typename ValueType> std::vector<ValueType> read_array(std::istream& stream) {
    const int64_t count = read_scalar<int64_t>(stream);
    if (count < 0) {
        throw std::runtime_error("interpolation record: negative array length; the file is "
                                 "corrupt.");
    }
    require_bytes_available(stream, static_cast<std::uint64_t>(count) * sizeof(ValueType));
    std::vector<ValueType> values(static_cast<std::size_t>(count));
    if (count > 0) {
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(ValueType)));
        if (!stream) {
            throw std::runtime_error(
                "interpolation record: the file ended early; it is truncated.");
        }
    }
    return values;
}

template <typename ValueType>
void write_nested_arrays(std::ostream& stream, const std::vector<std::vector<ValueType>>& rows) {
    write_scalar<int64_t>(stream, static_cast<int64_t>(rows.size()));
    for (const std::vector<ValueType>& row : rows) {
        write_array(stream, row);
    }
}

template <typename ValueType>
std::vector<std::vector<ValueType>> read_nested_arrays(std::istream& stream) {
    const int64_t count = read_scalar<int64_t>(stream);
    if (count < 0) {
        throw std::runtime_error("interpolation record: negative array count; the file is "
                                 "corrupt.");
    }
    require_bytes_available(stream, static_cast<std::uint64_t>(count) * sizeof(int64_t));
    std::vector<std::vector<ValueType>> rows(static_cast<std::size_t>(count));
    for (std::vector<ValueType>& row : rows) {
        row = read_array<ValueType>(stream);
    }
    return rows;
}

} // namespace q3::internal::interpolation_stream
