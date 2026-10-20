#pragma once

/**
 * The native interpolation record on disk: `<prefix>.save/interpolation.q3ip`.
 *
 * An NSCF run with a `control.interpolation` block writes one; `q3_utility
 * interpolate` reads it back and evaluates the model without the engine, which
 * is the point of saving it at all.
 *
 * This file owns the ENVELOPE only -- the magic number, the format version,
 * the method's name, the lattice and the Fermi level -- and dispatches the
 * method's own payload to a file beside it. A second interpolation method adds
 * a payload file and one case in each direction here.
 *
 * The version gate is load-bearing: a field added to a payload without bumping
 * it decodes as garbage far from the mistake, which is exactly how a stale
 * record once failed inside a consumer with a bare "Error: vector".
 */

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "io/interpolation/internal/binary_stream.hpp"
#include "io/interpolation/wannier/payload.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"

namespace q3 {

namespace internal::interpolation_checkpoint {

constexpr uint32_t magic = 0x50493351; // "Q3IP"
constexpr uint32_t version = 2;        // 2: trial-orbital centres in the Wannier payload

} // namespace internal::interpolation_checkpoint

inline void write_interpolation_record(const InterpolationRecord& record,
                                       const std::string& file_path) {
    namespace stream_io = internal::interpolation_stream;
    namespace constants = internal::interpolation_checkpoint;
    if (record.method.empty()) {
        throw std::invalid_argument(
            "interpolation record: the record names no method; nothing to write.");
    }
    std::ofstream stream(file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("interpolation record: cannot open '" + file_path +
                                 "' for writing.");
    }
    stream_io::write_scalar<uint32_t>(stream, constants::magic);
    stream_io::write_scalar<uint32_t>(stream, constants::version);
    stream_io::write_string(stream, record.method);
    stream_io::write_matrix(stream, record.real_lattice_bohr);
    stream_io::write_matrix(stream, record.reciprocal_lattice_inverse_bohr);
    stream_io::write_scalar<double>(stream, record.fermi_energy_rydberg);
    if (record.method == "wannier") {
        internal::wannier_payload::write(stream, record.wannier);
    } else {
        throw std::invalid_argument("interpolation record: no serializer for method '" +
                                    record.method + "'.");
    }
    if (!stream) {
        throw std::runtime_error("interpolation record: writing '" + file_path + "' failed.");
    }
}

[[nodiscard]] inline InterpolationRecord read_interpolation_record(const std::string& file_path) {
    namespace stream_io = internal::interpolation_stream;
    namespace constants = internal::interpolation_checkpoint;
    std::ifstream stream(file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("interpolation record: cannot open '" + file_path + "'.");
    }
    if (stream_io::read_scalar<uint32_t>(stream) != constants::magic) {
        throw std::runtime_error("interpolation record: '" + file_path +
                                 "' is not an interpolation record (bad magic).");
    }
    const uint32_t version = stream_io::read_scalar<uint32_t>(stream);
    if (version != constants::version) {
        throw std::runtime_error("interpolation record: '" + file_path + "' has format version " +
                                 std::to_string(version) + "; this build reads version " +
                                 std::to_string(constants::version) + ".");
    }
    InterpolationRecord record;
    record.method = stream_io::read_string(stream);
    record.real_lattice_bohr = stream_io::read_matrix(stream);
    record.reciprocal_lattice_inverse_bohr = stream_io::read_matrix(stream);
    record.fermi_energy_rydberg = stream_io::read_scalar<double>(stream);
    if (record.method == "wannier") {
        record.wannier = internal::wannier_payload::read(stream);
    } else {
        throw std::runtime_error("interpolation record: '" + file_path +
                                 "' was built by unknown method '" + record.method + "'.");
    }
    return record;
}

} // namespace q3
