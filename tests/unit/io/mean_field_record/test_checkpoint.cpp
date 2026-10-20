#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "io/mean_field_record/checkpoint.hpp"
#include "io/mean_field_record/init.hpp"

using namespace q3;

namespace {

std::string make_temporary_path(const std::string& suffix) {
    return std::string("/tmp/q3_mean_field_record_test") + suffix;
}

/// Band source over caller-owned coefficients, so a record needs no SCF state.
class VectorBandSource : public MeanFieldBandSource {
  public:
    VectorBandSource(std::vector<Complex> values, int number_of_bands, int plane_waves_per_kpoint)
        : values_(std::move(values))
        , number_of_bands_(number_of_bands)
        , plane_waves_per_kpoint_(plane_waves_per_kpoint) {}

    [[nodiscard]] const Complex* coefficients(int kpoint_state_index,
                                              int band_index) const override {
        size_t stride = static_cast<size_t>(number_of_bands_) * plane_waves_per_kpoint_;
        return values_.data() + kpoint_state_index * stride +
               static_cast<size_t>(band_index) * plane_waves_per_kpoint_;
    }

  private:
    std::vector<Complex> values_;
    int number_of_bands_;
    int plane_waves_per_kpoint_;
};

/**
 * A record with every field set to a distinct, recognizable value.
 *
 * Distinct on purpose: a round-trip test only has teeth if a field written into
 * the wrong slot changes what comes back, which shared placeholder values would
 * hide.
 */
MeanFieldRecord make_populated_record(int number_of_kpoints = 2,
                                      int number_of_bands = 3,
                                      int plane_waves_per_kpoint = 5) {
    MeanFieldRecord record;
    record.bravais_lattice_index = 2;
    record.lattice_parameter = 10.25;
    record.two_pi_over_lattice_parameter = 0.613;
    record.cell_volume = 256.75;
    for (int row = 0; row < 3; row++) {
        record.lattice_vectors[row] =
            Vector3D{0.1 * row + 0.11, 0.2 * row + 0.22, 0.3 * row + 0.33};
        record.reciprocal_lattice_vectors[row] =
            Vector3D{0.4 * row + 0.44, 0.5 * row + 0.55, 0.6 * row + 0.66};
    }

    record.atom_positions_cartesian = {Vector3D{0.125, 0.25, 0.375}, Vector3D{0.5, 0.625, 0.75}};
    record.atomic_numbers = {14, 32};

    SymmetryOperation operation;
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            operation.rotation[row][column] = row * 3 + column;
            operation.reciprocal_rotation[row][column] = 100 + row * 3 + column;
        }
    }
    operation.fractional_translation = Vector3D{0.5, 0.25, 0.125};
    operation.atom_map = {1, 0};
    record.symmetry_operations = {operation};

    record.fft_grid_dimensions = {12, 15, 18};
    for (int g_index = 0; g_index < 7; g_index++) {
        record.charge_density_miller_indices.push_back(
            IntegerVector3D{g_index, -g_index, 2 * g_index});
    }
    record.is_gamma_only = true;
    record.charge_density_energy_cutoff = 128.0;
    record.wavefunction_energy_cutoff = 32.0;

    record.kpoint_grid_dimensions = {2, 3, 4};
    record.kpoint_grid_shifts = {1, 0, 1};
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
        record.kpoint_coordinates.push_back(
            Vector3D{0.1 * kpoint_index, 0.2 * kpoint_index, 0.3 * kpoint_index});
        record.kpoint_weights.push_back(0.5 + kpoint_index);
        record.plane_waves_per_kpoint.push_back(plane_waves_per_kpoint);
        std::vector<int> indices;
        for (int g_index = 0; g_index < plane_waves_per_kpoint; g_index++)
            indices.push_back((g_index + kpoint_index) % 7);
        record.kpoint_g_vector_indices.push_back(indices);
    }

    record.number_of_bands = number_of_bands;
    record.number_of_spin_components = 1;
    for (int state_index = 0; state_index < number_of_kpoints; state_index++) {
        std::vector<Real> eigenvalues;
        std::vector<Real> occupations;
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            eigenvalues.push_back(-1.5 + state_index + 0.25 * band_index);
            occupations.push_back(band_index == 0 ? 1.0 : 0.25 * band_index);
        }
        record.eigenvalues_rydberg.push_back(eigenvalues);
        record.occupations.push_back(occupations);
    }

    record.charge_density_g.resize(1);
    for (int g_index = 0; g_index < 7; g_index++)
        record.charge_density_g[0].push_back(Complex(0.5 * g_index, -0.25 * g_index));
    for (int g_index = 0; g_index < 7; g_index++)
        record.exchange_correlation_potential_g.push_back(
            Complex(-0.75 * g_index, 0.125 * g_index));

    std::vector<Complex> coefficients;
    for (int state_index = 0; state_index < number_of_kpoints; state_index++)
        for (int band_index = 0; band_index < number_of_bands; band_index++)
            for (int plane_wave = 0; plane_wave < plane_waves_per_kpoint; plane_wave++)
                coefficients.push_back(
                    Complex(state_index * 100 + band_index * 10 + plane_wave, -plane_wave));
    record.band_source = std::make_shared<VectorBandSource>(
        std::move(coefficients), number_of_bands, plane_waves_per_kpoint);

    return record;
}

} // namespace

TEST(MeanFieldRecordCheckpoint, RoundTripPreservesEveryField) {
    MeanFieldRecord original = make_populated_record();
    std::string path = make_temporary_path(".roundtrip");
    write_mean_field_record(original, path);
    MeanFieldRecord loaded = read_mean_field_record(path);

    EXPECT_EQ(loaded.bravais_lattice_index, original.bravais_lattice_index);
    EXPECT_DOUBLE_EQ(loaded.lattice_parameter, original.lattice_parameter);
    EXPECT_DOUBLE_EQ(loaded.two_pi_over_lattice_parameter, original.two_pi_over_lattice_parameter);
    EXPECT_DOUBLE_EQ(loaded.cell_volume, original.cell_volume);
    for (int row = 0; row < 3; row++) {
        EXPECT_DOUBLE_EQ(loaded.lattice_vectors[row].x, original.lattice_vectors[row].x);
        EXPECT_DOUBLE_EQ(loaded.lattice_vectors[row].z, original.lattice_vectors[row].z);
        EXPECT_DOUBLE_EQ(loaded.reciprocal_lattice_vectors[row].y,
                         original.reciprocal_lattice_vectors[row].y);
    }

    ASSERT_EQ(loaded.atomic_numbers, original.atomic_numbers);
    ASSERT_EQ(loaded.atom_positions_cartesian.size(), original.atom_positions_cartesian.size());
    EXPECT_DOUBLE_EQ(loaded.atom_positions_cartesian[1].y, original.atom_positions_cartesian[1].y);

    ASSERT_EQ(loaded.symmetry_operations.size(), 1u);
    EXPECT_EQ(loaded.symmetry_operations[0].rotation[2][1],
              original.symmetry_operations[0].rotation[2][1]);
    EXPECT_EQ(loaded.symmetry_operations[0].reciprocal_rotation[0][2],
              original.symmetry_operations[0].reciprocal_rotation[0][2]);
    EXPECT_DOUBLE_EQ(loaded.symmetry_operations[0].fractional_translation.z,
                     original.symmetry_operations[0].fractional_translation.z);
    EXPECT_EQ(loaded.symmetry_operations[0].atom_map, original.symmetry_operations[0].atom_map);

    EXPECT_EQ(loaded.fft_grid_dimensions, original.fft_grid_dimensions);
    ASSERT_EQ(loaded.charge_density_miller_indices.size(),
              original.charge_density_miller_indices.size());
    EXPECT_EQ(loaded.charge_density_miller_indices[3].y,
              original.charge_density_miller_indices[3].y);
    EXPECT_EQ(loaded.is_gamma_only, original.is_gamma_only);
    EXPECT_DOUBLE_EQ(loaded.charge_density_energy_cutoff, original.charge_density_energy_cutoff);
    EXPECT_DOUBLE_EQ(loaded.wavefunction_energy_cutoff, original.wavefunction_energy_cutoff);

    EXPECT_EQ(loaded.kpoint_grid_dimensions, original.kpoint_grid_dimensions);
    EXPECT_EQ(loaded.kpoint_grid_shifts, original.kpoint_grid_shifts);
    EXPECT_EQ(loaded.kpoint_weights, original.kpoint_weights);
    EXPECT_EQ(loaded.plane_waves_per_kpoint, original.plane_waves_per_kpoint);
    EXPECT_EQ(loaded.kpoint_g_vector_indices, original.kpoint_g_vector_indices);
    EXPECT_DOUBLE_EQ(loaded.kpoint_coordinates[1].z, original.kpoint_coordinates[1].z);

    EXPECT_EQ(loaded.number_of_bands, original.number_of_bands);
    EXPECT_EQ(loaded.number_of_spin_components, original.number_of_spin_components);
    EXPECT_EQ(loaded.eigenvalues_rydberg, original.eigenvalues_rydberg);
    EXPECT_EQ(loaded.occupations, original.occupations);
    EXPECT_EQ(loaded.charge_density_g, original.charge_density_g);
    EXPECT_EQ(loaded.exchange_correlation_potential_g, original.exchange_correlation_potential_g);

    std::remove(path.c_str());
}

TEST(MeanFieldRecordCheckpoint, RoundTripPreservesWavefunctionCoefficients) {
    /**
     * The coefficients are the one field that does not travel in the record
     * itself -- they go through MeanFieldBandSource, are written as a trailing
     * block, and are read back by seeking. That makes them the most likely
     * thing to be silently misaligned, so every (k, band, G) value is checked
     * rather than a sample.
     */
    MeanFieldRecord original = make_populated_record();
    std::string path = make_temporary_path(".coefficients");
    write_mean_field_record(original, path);
    MeanFieldRecord loaded = read_mean_field_record(path);

    ASSERT_NE(loaded.band_source, nullptr);
    for (int state_index = 0; state_index < original.number_of_kpoint_states(); state_index++) {
        for (int band_index = 0; band_index < original.number_of_bands; band_index++) {
            const Complex* expected = original.band_source->coefficients(state_index, band_index);
            const Complex* actual = loaded.band_source->coefficients(state_index, band_index);
            for (int plane_wave = 0; plane_wave < original.plane_waves_per_kpoint[state_index];
                 plane_wave++) {
                EXPECT_EQ(actual[plane_wave], expected[plane_wave])
                    << "k-point state " << state_index << ", band " << band_index << ", G "
                    << plane_wave;
            }
        }
    }

    std::remove(path.c_str());
}

TEST(MeanFieldRecordCheckpoint, RejectsAForeignFile) {
    std::string path = make_temporary_path(".foreign");
    {
        std::ofstream stream(path, std::ios::binary);
        stream << "this is not a Q3 mean-field record at all";
    }
    EXPECT_THROW(read_mean_field_record(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(MeanFieldRecordCheckpoint, RejectsAnUnknownVersion) {
    /**
     * A future writer bumping the version must not be misparsed by an older
     * reader as though the layout were unchanged -- that would produce
     * plausible-looking garbage rather than an error.
     */
    MeanFieldRecord original = make_populated_record();
    std::string path = make_temporary_path(".version");
    write_mean_field_record(original, path);
    {
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(sizeof(uint32_t)); // straight past the magic
        uint32_t future_version = internal::mean_field_record_version + 1;
        stream.write(reinterpret_cast<const char*>(&future_version), sizeof(future_version));
    }
    EXPECT_THROW(read_mean_field_record(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(MeanFieldRecordCheckpoint, RejectsATruncatedCoefficientBlock) {
    /**
     * Truncation in the trailing coefficient block is invisible to the header
     * parse, so it has to be caught at read time rather than producing zeros.
     */
    MeanFieldRecord original = make_populated_record();
    std::string path = make_temporary_path(".truncated");
    write_mean_field_record(original, path);

    std::streamoff full_size = 0;
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        full_size = stream.tellg();
    }
    {
        std::vector<char> bytes(static_cast<size_t>(full_size));
        std::ifstream input(path, std::ios::binary);
        input.read(bytes.data(), full_size);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(bytes.data(), full_size - static_cast<std::streamoff>(sizeof(Complex)));
    }

    MeanFieldRecord loaded = read_mean_field_record(path);
    int last_state = loaded.number_of_kpoint_states() - 1;
    int last_band = loaded.number_of_bands - 1;
    EXPECT_THROW(loaded.band_source->coefficients(last_state, last_band), std::runtime_error);

    std::remove(path.c_str());
}
