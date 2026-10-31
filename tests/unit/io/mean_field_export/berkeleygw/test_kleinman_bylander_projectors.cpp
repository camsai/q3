#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/mean_field_export/berkeleygw/kleinman_bylander_projectors.hpp"
#include "io/mean_field_record/init.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

namespace {

/// <k+G|β> = (100 k + projector) + i G, so every value names its own position.
class LabelledProjectorSource : public MeanFieldProjectorSource {
  public:
    LabelledProjectorSource(int number_of_projectors, std::vector<int> plane_waves_per_kpoint)
        : number_of_projectors_(number_of_projectors)
        , plane_waves_per_kpoint_(std::move(plane_waves_per_kpoint)) {}

    [[nodiscard]] const Matrix<Complex>& projectors(int kpoint_index) const override {
        projectors_.resize(number_of_projectors_, plane_waves_per_kpoint_[kpoint_index]);
        for (int projector = 0; projector < number_of_projectors_; projector++) {
            for (int g_index = 0; g_index < plane_waves_per_kpoint_[kpoint_index]; g_index++) {
                projectors_(projector, g_index) =
                    Complex(100.0 * kpoint_index + projector, static_cast<Real>(g_index));
            }
        }
        return projectors_;
    }

  private:
    int number_of_projectors_;
    std::vector<int> plane_waves_per_kpoint_;
    mutable Matrix<Complex> projectors_;
};

/**
 * Three atoms of two species in the order B, A, B -- so the file's
 * species-grouped order (A; then B, B) differs from the crystal's -- with
 * 2 channels for species 0 (A) and 3 for species 1 (B).
 */
MeanFieldRecord synthetic_record() {
    MeanFieldRecord record;
    record.lattice_parameter = 10.0;
    record.two_pi_over_lattice_parameter = 2.0 * 3.141592653589793 / 10.0;
    record.cell_volume = 1000.0;
    record.lattice_vectors[0] = Vector3D(1.0, 0.0, 0.0);
    record.lattice_vectors[1] = Vector3D(0.0, 1.0, 0.0);
    record.lattice_vectors[2] = Vector3D(0.0, 0.0, 1.0);
    record.reciprocal_lattice_vectors = record.lattice_vectors;
    record.atom_positions_cartesian = {
        Vector3D(0.0, 0.0, 0.0), Vector3D(0.5, 0.5, 0.5), Vector3D(0.25, 0.25, 0.25)};
    record.atomic_numbers = {8, 3, 8};
    record.fft_grid_dimensions = {4, 4, 4};
    record.charge_density_miller_indices = {IntegerVector3D(0, 0, 0),
                                            IntegerVector3D(1, 0, 0),
                                            IntegerVector3D(-1, 0, 0),
                                            IntegerVector3D(0, 1, 0)};
    record.charge_density_energy_cutoff = 40.0;
    record.wavefunction_energy_cutoff = 10.0;
    record.kpoint_grid_dimensions = {2, 1, 1};
    record.kpoint_coordinates = {Vector3D(0.0, 0.0, 0.0), Vector3D(0.5, 0.0, 0.0)};
    record.kpoint_weights = {1.0, 1.0};
    record.plane_waves_per_kpoint = {3, 2};
    record.kpoint_g_vector_indices = {{0, 1, 2}, {3, 0}};
    record.number_of_bands = 1;
    record.number_of_spin_components = 1;

    record.atom_species_indices = {1, 0, 1};
    record.projectors_per_species = {2, 3};
    record.nonlocal_coefficients = {{}};
    for (int atom_index = 0; atom_index < 3; atom_index++) {
        int channels = record.projectors_per_species[record.atom_species_indices[atom_index]];
        std::vector<Real> block(static_cast<size_t>(channels) * channels);
        for (int column = 0; column < channels; column++) {
            for (int row = 0; row < channels; row++) {
                block[row + channels * column] = 10.0 * atom_index + row + 0.1 * column;
            }
        }
        record.nonlocal_coefficients[0].push_back(block);
    }
    record.projector_source =
        std::make_shared<LabelledProjectorSource>(3 + 2 + 3, record.plane_waves_per_kpoint);
    return record;
}

/// Every Fortran sequential record of a file, payload only, framing checked.
std::vector<std::vector<char>> read_all_records(const std::string& path) {
    std::ifstream input_file(path, std::ios::binary);
    std::vector<std::vector<char>> records;
    int32_t leading_length = 0;
    while (input_file.read(reinterpret_cast<char*>(&leading_length), 4)) {
        std::vector<char> payload(leading_length);
        input_file.read(payload.data(), leading_length);
        int32_t trailing_length = 0;
        input_file.read(reinterpret_cast<char*>(&trailing_length), 4);
        EXPECT_EQ(leading_length, trailing_length) << "record " << records.size() + 1;
        records.push_back(std::move(payload));
    }
    return records;
}

template <typename T> std::vector<T> values_of(const std::vector<char>& payload) {
    std::vector<T> values(payload.size() / sizeof(T));
    std::memcpy(values.data(), payload.data(), payload.size());
    return values;
}

} // namespace

TEST(BerkeleyGWKleinmanBylanderProjectorFile, RecordCountFollowsTheParaBandsReader) {
    MeanFieldRecord record = synthetic_record();
    std::string path = test_support::make_temporary_path("_vkb");
    BerkeleyGWKleinmanBylanderProjectorFile(record).write_to_file(path);

    auto records = read_all_records(path);
    /**
     * 8 header + the k-point records + the three projector records + 3 for the G-set; then per
     * k-point 3 for its G-vectors and 3 per channel: 8 channels at each of 2 k-points.
     */
    EXPECT_EQ(records.size(), 8u + 3u + 3u + 3u + 2u * (3u + 3u * 8u));
}

TEST(BerkeleyGWKleinmanBylanderProjectorFile, DimensionRecordCarriesProjectorCounts) {
    MeanFieldRecord record = synthetic_record();
    std::string path = test_support::make_temporary_path("_vkb");
    BerkeleyGWKleinmanBylanderProjectorFile(record).write_to_file(path);

    auto records = read_all_records(path);
    // ns ng ntran cell_symmetry nat (int32) ecutrho (f64) nk nsp nkb nhm ngkmax (int32) ecutwfc
    const auto& dimension_record = records[1];
    ASSERT_EQ(dimension_record.size(), 5u * 4u + 8u + 5u * 4u + 8u);
    std::vector<int32_t> integers_after_cutoff(5);
    std::memcpy(integers_after_cutoff.data(), dimension_record.data() + 28, 5 * sizeof(int32_t));
    EXPECT_EQ(integers_after_cutoff[0], 2); // nk
    EXPECT_EQ(integers_after_cutoff[1], 2); // nsp
    EXPECT_EQ(integers_after_cutoff[2], 8); // nkb = 3 + 2 + 3
    EXPECT_EQ(integers_after_cutoff[3], 3); // nhm
    EXPECT_EQ(integers_after_cutoff[4], 3); // ngkmax
}

TEST(BerkeleyGWKleinmanBylanderProjectorFile, SpeciesCountsAndPaddedCoefficients) {
    MeanFieldRecord record = synthetic_record();
    std::string path = test_support::make_temporary_path("_vkb");
    BerkeleyGWKleinmanBylanderProjectorFile(record).write_to_file(path);

    auto records = read_all_records(path);
    EXPECT_EQ(values_of<double>(records[9]), std::vector<double>({0.5, 0.5}));   // kw sums to 1
    EXPECT_EQ(values_of<int32_t>(records[11]), std::vector<int32_t>({2, 1, 2})); // ityp, 1-based
    EXPECT_EQ(values_of<int32_t>(records[12]), std::vector<int32_t>({2, 3}));    // nh

    auto coefficients = values_of<double>(records[13]); // D(3, 3, 3 atoms, 1 spin), padded
    ASSERT_EQ(coefficients.size(), 27u);
    auto coefficient = [&](int row, int column, int atom) {
        return coefficients[row + 3 * (column + 3 * atom)];
    };
    EXPECT_DOUBLE_EQ(coefficient(2, 1, 0), 0.0 + 2 + 0.1);  // atom 0, species B, 3 channels
    EXPECT_DOUBLE_EQ(coefficient(1, 0, 1), 10.0 + 1 + 0.0); // atom 1, species A, 2 channels
    EXPECT_DOUBLE_EQ(coefficient(2, 2, 1), 0.0);            // padding beyond A's 2 channels
    EXPECT_DOUBLE_EQ(coefficient(0, 2, 2), 20.0 + 0 + 0.2);
}

TEST(BerkeleyGWKleinmanBylanderProjectorFile, BodyGroupsAtomsBySpecies) {
    MeanFieldRecord record = synthetic_record();
    std::string path = test_support::make_temporary_path("_vkb");
    BerkeleyGWKleinmanBylanderProjectorFile(record).write_to_file(path);

    auto records = read_all_records(path);
    size_t index = 17; // after the header, records 9-14 and the G-set
    for (int kpoint_index = 0; kpoint_index < 2; kpoint_index++) {
        int plane_waves = record.plane_waves_per_kpoint[kpoint_index];
        EXPECT_EQ(values_of<int32_t>(records[index + 1]), std::vector<int32_t>({plane_waves}));
        index += 3;
        /**
         * Crystal order puts atom 1 (species A) at projector rows 3-4; the file
         * takes species A first, then atoms 0 and 2 (rows 0-2, 5-7).
         */
        for (int projector_row : {3, 4, 0, 1, 2, 5, 6, 7}) {
            auto values = values_of<Complex>(records[index + 2]);
            ASSERT_EQ(values.size(), static_cast<size_t>(plane_waves));
            EXPECT_DOUBLE_EQ(values[0].real(), 100.0 * kpoint_index + projector_row);
            EXPECT_DOUBLE_EQ(values.back().imag(), plane_waves - 1.0);
            index += 3;
        }
    }
    EXPECT_EQ(index, records.size());
}

TEST(BerkeleyGWKleinmanBylanderProjectorFile, RefusesARecordWithoutProjectors) {
    MeanFieldRecord record = synthetic_record();
    record.projector_source = nullptr;
    std::string path = test_support::make_temporary_path("_vkb");
    EXPECT_THROW(BerkeleyGWKleinmanBylanderProjectorFile(record).write_to_file(path),
                 std::runtime_error);
}
