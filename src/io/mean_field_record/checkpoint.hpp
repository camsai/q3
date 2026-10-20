#pragma once

/**
 * Persistence for MeanFieldRecord — the file half of the converter seam.
 *
 * `write_mean_field_record()` drops a complete mean-field description into
 * `<prefix>.save/mean-field.q3mf` at the end of a run; `read_mean_field_record()`
 * loads it back, so the standalone converter can produce any supported format
 * without re-running the SCF. Getting a file wrong then costs a re-convert
 * rather than a re-calculation.
 *
 * This is a NEW file rather than an extension of `effective-potential.q3cp`
 * because the two answer different questions. That checkpoint exists to restart
 * an NSCF run and carries what the engine needs to resume; this one exists to
 * describe a converged mean field to an outside code and carries what any
 * external format needs (symmetry, k-point metadata, occupations, V_xc). They
 * overlap but neither contains the other, and conflating them would make each
 * carry fields the other's consumer must learn to ignore.
 *
 * Format: "Q3MF" magic, a version, then length-prefixed arrays, and finally the
 * wavefunction coefficients as one contiguous block. Fields are written
 * component by component rather than by memcpy-ing structs, so the file does
 * not silently depend on host padding. Reads validate the magic and version and
 * refuse anything else instead of misparsing it.
 *
 * The coefficient block is laid out so `FileMeanFieldBandSource` can seek
 * straight to a band: all bands at one k-point share that k-point's plane-wave
 * count, so a k-point's offset is a running sum of
 * `number_of_bands * plane_waves_per_kpoint[k]`, computed on load. Nothing is
 * held in memory beyond the k-point currently being read.
 */

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/mean_field_record/init.hpp"

namespace q3 {

namespace internal {

constexpr uint32_t mean_field_record_magic = 0x464D3351; // "Q3MF"
constexpr uint32_t mean_field_record_version = 1;

template <typename ValueType> void write_scalar(std::ostream& stream, const ValueType& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(ValueType));
}

template <typename ValueType> ValueType read_scalar(std::istream& stream) {
    ValueType value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(ValueType));
    return value;
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
    int64_t count = read_scalar<int64_t>(stream);
    if (count < 0)
        throw std::runtime_error("mean-field record: negative array length; the file is corrupt.");
    std::vector<ValueType> values(static_cast<size_t>(count));
    if (count > 0) {
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(ValueType)));
    }
    return values;
}

} // namespace internal

/**
 * Band source that reads coefficients back from a mean-field record file.
 *
 * Caches exactly one k-point: exporters walk (k, band) in order, so this turns
 * into one sequential read per k-point while keeping peak memory to a single
 * k-point's worth of coefficients rather than the whole wavefunction set.
 */
class FileMeanFieldBandSource : public MeanFieldBandSource {
  public:
    FileMeanFieldBandSource(std::string file_path,
                            std::streamoff coefficient_block_offset,
                            int number_of_bands,
                            std::vector<int> plane_waves_per_kpoint)
        : file_path_(std::move(file_path))
        , coefficient_block_offset_(coefficient_block_offset)
        , number_of_bands_(number_of_bands)
        , plane_waves_per_kpoint_(std::move(plane_waves_per_kpoint)) {
        kpoint_start_offsets_.resize(plane_waves_per_kpoint_.size() + 1, 0);
        for (size_t kpoint_index = 0; kpoint_index < plane_waves_per_kpoint_.size();
             kpoint_index++) {
            kpoint_start_offsets_[kpoint_index + 1] =
                kpoint_start_offsets_[kpoint_index] +
                static_cast<std::streamoff>(number_of_bands_) *
                    plane_waves_per_kpoint_[kpoint_index] *
                    static_cast<std::streamoff>(sizeof(Complex));
        }
    }

    [[nodiscard]] const Complex* coefficients(int kpoint_state_index,
                                              int band_index) const override {
        if (kpoint_state_index != cached_kpoint_state_index_) {
            load_kpoint(kpoint_state_index);
        }
        int plane_wave_count = plane_waves_per_kpoint_[kpoint_state_index];
        return cached_coefficients_.data() +
               static_cast<size_t>(band_index) * static_cast<size_t>(plane_wave_count);
    }

  private:
    void load_kpoint(int kpoint_state_index) const {
        if (kpoint_state_index < 0 ||
            kpoint_state_index >= static_cast<int>(plane_waves_per_kpoint_.size())) {
            throw std::runtime_error("mean-field record: k-point state index " +
                                     std::to_string(kpoint_state_index) + " is out of range.");
        }
        std::ifstream stream(file_path_, std::ios::binary);
        if (!stream.is_open()) {
            throw std::runtime_error("mean-field record: cannot reopen '" + file_path_ +
                                     "' to read wavefunction coefficients.");
        }
        int plane_wave_count = plane_waves_per_kpoint_[kpoint_state_index];
        size_t value_count = static_cast<size_t>(number_of_bands_) * plane_wave_count;
        cached_coefficients_.resize(value_count);
        stream.seekg(coefficient_block_offset_ + kpoint_start_offsets_[kpoint_state_index]);
        stream.read(reinterpret_cast<char*>(cached_coefficients_.data()),
                    static_cast<std::streamsize>(value_count * sizeof(Complex)));
        if (!stream) {
            throw std::runtime_error("mean-field record: '" + file_path_ +
                                     "' ended while reading coefficients for k-point state " +
                                     std::to_string(kpoint_state_index) +
                                     "; the file is truncated or corrupt.");
        }
        cached_kpoint_state_index_ = kpoint_state_index;
    }

    std::string file_path_;
    std::streamoff coefficient_block_offset_;
    int number_of_bands_;
    std::vector<int> plane_waves_per_kpoint_;
    std::vector<std::streamoff> kpoint_start_offsets_;
    mutable std::vector<Complex> cached_coefficients_;
    mutable int cached_kpoint_state_index_ = -1;
};

/// Serializes `record` to `file_path`. Throws on any write failure.
inline void write_mean_field_record(const MeanFieldRecord& record, const std::string& file_path) {
    using namespace internal;
    std::ofstream stream(file_path, std::ios::binary);
    if (!stream.is_open())
        throw std::runtime_error("mean-field record: cannot open '" + file_path + "' for writing.");

    write_scalar<uint32_t>(stream, mean_field_record_magic);
    write_scalar<uint32_t>(stream, mean_field_record_version);

    write_scalar<int32_t>(stream, record.bravais_lattice_index);
    write_scalar<double>(stream, record.lattice_parameter);
    write_scalar<double>(stream, record.two_pi_over_lattice_parameter);
    write_scalar<double>(stream, record.cell_volume);
    for (int row = 0; row < 3; row++)
        write_vector3d(stream, record.lattice_vectors[row]);
    for (int row = 0; row < 3; row++)
        write_vector3d(stream, record.reciprocal_lattice_vectors[row]);

    write_scalar<int64_t>(stream, static_cast<int64_t>(record.atomic_numbers.size()));
    for (const auto& position : record.atom_positions_cartesian)
        write_vector3d(stream, position);
    write_array(stream, record.atomic_numbers);

    write_scalar<int64_t>(stream, static_cast<int64_t>(record.symmetry_operations.size()));
    for (const auto& operation : record.symmetry_operations) {
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                write_scalar<int32_t>(stream, operation.rotation[row][column]);
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                write_scalar<int32_t>(stream, operation.reciprocal_rotation[row][column]);
        write_vector3d(stream, operation.fractional_translation);
        write_array(stream, operation.atom_map);
    }

    for (int axis = 0; axis < 3; axis++)
        write_scalar<int32_t>(stream, record.fft_grid_dimensions[axis]);
    write_array(stream, record.charge_density_miller_indices);
    write_scalar<uint8_t>(stream, record.is_gamma_only ? 1 : 0);
    write_scalar<double>(stream, record.charge_density_energy_cutoff);
    write_scalar<double>(stream, record.wavefunction_energy_cutoff);

    for (int axis = 0; axis < 3; axis++)
        write_scalar<int32_t>(stream, record.kpoint_grid_dimensions[axis]);
    for (int axis = 0; axis < 3; axis++)
        write_scalar<int32_t>(stream, record.kpoint_grid_shifts[axis]);
    write_scalar<int64_t>(stream, static_cast<int64_t>(record.kpoint_coordinates.size()));
    for (const auto& coordinates : record.kpoint_coordinates)
        write_vector3d(stream, coordinates);
    write_array(stream, record.kpoint_weights);
    write_array(stream, record.plane_waves_per_kpoint);
    write_scalar<int64_t>(stream, static_cast<int64_t>(record.kpoint_g_vector_indices.size()));
    for (const auto& indices : record.kpoint_g_vector_indices)
        write_array(stream, indices);

    write_scalar<int32_t>(stream, record.number_of_bands);
    write_scalar<int32_t>(stream, record.number_of_spin_components);
    write_scalar<int64_t>(stream, static_cast<int64_t>(record.eigenvalues_rydberg.size()));
    for (const auto& row : record.eigenvalues_rydberg)
        write_array(stream, row);
    write_scalar<int64_t>(stream, static_cast<int64_t>(record.occupations.size()));
    for (const auto& row : record.occupations)
        write_array(stream, row);

    write_scalar<int64_t>(stream, static_cast<int64_t>(record.charge_density_g.size()));
    for (const auto& spin_density : record.charge_density_g)
        write_array(stream, spin_density);
    write_array(stream, record.exchange_correlation_potential_g);

    /**
     * Coefficients last, as one contiguous block, so their start offset is
     * everything written so far and a reader can seek per k-point.
     */
    if (!record.band_source) {
        throw std::runtime_error("mean-field record: the record has no band source, so its "
                                 "wavefunction coefficients cannot be written.");
    }
    for (int state_index = 0; state_index < record.number_of_kpoint_states(); state_index++) {
        int plane_wave_count = record.plane_waves_per_kpoint[state_index];
        for (int band_index = 0; band_index < record.number_of_bands; band_index++) {
            const Complex* coefficients = record.band_source->coefficients(state_index, band_index);
            stream.write(reinterpret_cast<const char*>(coefficients),
                         static_cast<std::streamsize>(plane_wave_count * sizeof(Complex)));
        }
    }

    if (!stream)
        throw std::runtime_error("mean-field record: writing '" + file_path + "' failed.");
}

/// Loads a record written by write_mean_field_record(). Throws on any problem.
[[nodiscard]] inline MeanFieldRecord read_mean_field_record(const std::string& file_path) {
    using namespace internal;
    std::ifstream stream(file_path, std::ios::binary);
    if (!stream.is_open())
        throw std::runtime_error("mean-field record: cannot open '" + file_path + "' for reading.");

    uint32_t magic = read_scalar<uint32_t>(stream);
    if (magic != mean_field_record_magic) {
        throw std::runtime_error("mean-field record: '" + file_path +
                                 "' is not a Q3 mean-field record (bad magic number).");
    }
    uint32_t version = read_scalar<uint32_t>(stream);
    if (version != mean_field_record_version) {
        throw std::runtime_error("mean-field record: '" + file_path + "' is version " +
                                 std::to_string(version) + ", but this build reads version " +
                                 std::to_string(mean_field_record_version) +
                                 ". Re-run the calculation to regenerate it.");
    }

    MeanFieldRecord record;
    record.bravais_lattice_index = read_scalar<int32_t>(stream);
    record.lattice_parameter = read_scalar<double>(stream);
    record.two_pi_over_lattice_parameter = read_scalar<double>(stream);
    record.cell_volume = read_scalar<double>(stream);
    for (int row = 0; row < 3; row++)
        record.lattice_vectors[row] = read_vector3d(stream);
    for (int row = 0; row < 3; row++)
        record.reciprocal_lattice_vectors[row] = read_vector3d(stream);

    int64_t number_of_atoms = read_scalar<int64_t>(stream);
    record.atom_positions_cartesian.resize(static_cast<size_t>(number_of_atoms));
    for (auto& position : record.atom_positions_cartesian)
        position = read_vector3d(stream);
    record.atomic_numbers = read_array<int>(stream);

    int64_t number_of_symmetry_operations = read_scalar<int64_t>(stream);
    record.symmetry_operations.resize(static_cast<size_t>(number_of_symmetry_operations));
    for (auto& operation : record.symmetry_operations) {
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                operation.rotation[row][column] = read_scalar<int32_t>(stream);
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                operation.reciprocal_rotation[row][column] = read_scalar<int32_t>(stream);
        operation.fractional_translation = read_vector3d(stream);
        operation.atom_map = read_array<int>(stream);
    }

    for (int axis = 0; axis < 3; axis++)
        record.fft_grid_dimensions[axis] = read_scalar<int32_t>(stream);
    record.charge_density_miller_indices = read_array<IntegerVector3D>(stream);
    record.is_gamma_only = read_scalar<uint8_t>(stream) != 0;
    record.charge_density_energy_cutoff = read_scalar<double>(stream);
    record.wavefunction_energy_cutoff = read_scalar<double>(stream);

    for (int axis = 0; axis < 3; axis++)
        record.kpoint_grid_dimensions[axis] = read_scalar<int32_t>(stream);
    for (int axis = 0; axis < 3; axis++)
        record.kpoint_grid_shifts[axis] = read_scalar<int32_t>(stream);
    int64_t number_of_kpoints = read_scalar<int64_t>(stream);
    record.kpoint_coordinates.resize(static_cast<size_t>(number_of_kpoints));
    for (auto& coordinates : record.kpoint_coordinates)
        coordinates = read_vector3d(stream);
    record.kpoint_weights = read_array<Real>(stream);
    record.plane_waves_per_kpoint = read_array<int>(stream);
    int64_t kpoint_g_set_count = read_scalar<int64_t>(stream);
    record.kpoint_g_vector_indices.resize(static_cast<size_t>(kpoint_g_set_count));
    for (auto& indices : record.kpoint_g_vector_indices)
        indices = read_array<int>(stream);

    record.number_of_bands = read_scalar<int32_t>(stream);
    record.number_of_spin_components = read_scalar<int32_t>(stream);
    int64_t eigenvalue_row_count = read_scalar<int64_t>(stream);
    record.eigenvalues_rydberg.resize(static_cast<size_t>(eigenvalue_row_count));
    for (auto& row : record.eigenvalues_rydberg)
        row = read_array<Real>(stream);
    int64_t occupation_row_count = read_scalar<int64_t>(stream);
    record.occupations.resize(static_cast<size_t>(occupation_row_count));
    for (auto& row : record.occupations)
        row = read_array<Real>(stream);

    int64_t spin_channel_count = read_scalar<int64_t>(stream);
    record.charge_density_g.resize(static_cast<size_t>(spin_channel_count));
    for (auto& spin_density : record.charge_density_g)
        spin_density = read_array<Complex>(stream);
    record.exchange_correlation_potential_g = read_array<Complex>(stream);

    if (!stream) {
        throw std::runtime_error("mean-field record: '" + file_path +
                                 "' ended before the header was fully read; it is truncated.");
    }

    record.band_source = std::make_shared<FileMeanFieldBandSource>(
        file_path, stream.tellg(), record.number_of_bands, record.plane_waves_per_kpoint);
    return record;
}

} // namespace q3
