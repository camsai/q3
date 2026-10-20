#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "io/interpolation/checkpoint.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

namespace {

/**
 * A record with every field set to a distinct, recognizable value. Distinct
 * on purpose: a round-trip test only has teeth if a field written into the
 * wrong slot changes what comes back.
 */
InterpolationRecord populated_record() {
    InterpolationRecord record;
    record.method = "wannier";
    record.fermi_energy_rydberg = 0.4375;
    for (int row = 0; row < 3; row++) {
        record.real_lattice_bohr[row] = Vector3D(10.0 + 0.5 * row, 0.25 * row, 0.5 + 0.125 * row);
        record.reciprocal_lattice_inverse_bohr[row] =
            Vector3D(0.6 + 0.1 * row, 0.2 * row, 0.01 + 0.05 * row);
    }
    WannierInputs& wannier = record.wannier;
    wannier.kpoint_grid_dimensions = {2, 1, 1};
    wannier.kpoint_coordinates = {Vector3D(0.0, 0.0, 0.0), Vector3D(0.5, 0.0, 0.0)};
    wannier.number_of_bands = 2;
    wannier.number_of_trial_orbitals = 2;
    wannier.number_of_spin_components = 1;
    wannier.fermi_energy_rydberg = 0.375;
    KPointNeighborDirection direction;
    direction.grid_step = IntegerVector3D{1, 0, 0};
    direction.reciprocal_lattice_translation = Vector3D(0.3, -0.1, 0.05);
    direction.weight = 0.5;
    wannier.neighbor_directions = {direction};
    wannier.overlap_matrices = {
        {{Complex(1.0, 0.0), Complex(0.1, 0.2), Complex(-0.1, 0.05), Complex(0.9, 0.0)}},
        {{Complex(0.8, 0.1), Complex(0.0, 0.0), Complex(0.2, 0.0), Complex(1.0, -0.1)}}};
    wannier.projection_matrices = {
        {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)},
        {Complex(0.7, 0.1), Complex(0.2, 0.0), Complex(-0.1, 0.3), Complex(0.9, 0.0)}};
    wannier.eigenvalues_rydberg = {{-0.5, 0.25}, {-0.4, 0.3}};
    wannier.projectabilities = {{1.0, 1.0}, {0.6, 0.9}};
    return record;
}

void expect_equal(const Vector3D& left, const Vector3D& right) {
    EXPECT_DOUBLE_EQ(left.x, right.x);
    EXPECT_DOUBLE_EQ(left.y, right.y);
    EXPECT_DOUBLE_EQ(left.z, right.z);
}

} // namespace

TEST(InterpolationRecordCheckpoint, RoundTripsEveryField) {
    const InterpolationRecord written = populated_record();
    const std::string path = test_support::make_temporary_path(".q3ip");
    write_interpolation_record(written, path);
    const InterpolationRecord read = read_interpolation_record(path);
    std::remove(path.c_str());

    EXPECT_EQ(read.method, "wannier");
    EXPECT_DOUBLE_EQ(read.fermi_energy_rydberg, 0.4375);
    for (int row = 0; row < 3; row++) {
        expect_equal(read.real_lattice_bohr[row], written.real_lattice_bohr[row]);
        expect_equal(read.reciprocal_lattice_inverse_bohr[row],
                     written.reciprocal_lattice_inverse_bohr[row]);
    }
    const WannierInputs& expected = written.wannier;
    const WannierInputs& actual = read.wannier;
    EXPECT_EQ(actual.kpoint_grid_dimensions, expected.kpoint_grid_dimensions);
    EXPECT_EQ(actual.number_of_bands, expected.number_of_bands);
    EXPECT_EQ(actual.number_of_trial_orbitals, expected.number_of_trial_orbitals);
    EXPECT_EQ(actual.number_of_spin_components, expected.number_of_spin_components);
    EXPECT_DOUBLE_EQ(actual.fermi_energy_rydberg, expected.fermi_energy_rydberg);
    ASSERT_EQ(actual.kpoint_coordinates.size(), expected.kpoint_coordinates.size());
    for (std::size_t index = 0; index < expected.kpoint_coordinates.size(); index++) {
        expect_equal(actual.kpoint_coordinates[index], expected.kpoint_coordinates[index]);
    }
    ASSERT_EQ(actual.neighbor_directions.size(), 1u);
    EXPECT_EQ(actual.neighbor_directions[0].grid_step.x, 1);
    EXPECT_EQ(actual.neighbor_directions[0].grid_step.y, 0);
    EXPECT_EQ(actual.neighbor_directions[0].grid_step.z, 0);
    expect_equal(actual.neighbor_directions[0].reciprocal_lattice_translation,
                 expected.neighbor_directions[0].reciprocal_lattice_translation);
    EXPECT_DOUBLE_EQ(actual.neighbor_directions[0].weight, 0.5);
    EXPECT_EQ(actual.overlap_matrices, expected.overlap_matrices);
    EXPECT_EQ(actual.projection_matrices, expected.projection_matrices);
    EXPECT_EQ(actual.eigenvalues_rydberg, expected.eigenvalues_rydberg);
    EXPECT_EQ(actual.projectabilities, expected.projectabilities);
}

/**
 * The trial-orbital centres are what the localiser guides by, and they exist
 * only because the capture wrote them: nothing downstream can reconstruct
 * where an orbital was placed. So the file has to carry them exactly, and a
 * count that disagrees with the orbital count is corruption, not a default.
 */
TEST(InterpolationRecordCheckpoint, RoundTripsTheTrialOrbitalCenters) {
    InterpolationRecord record = populated_record();
    record.wannier.trial_orbital_centers_bohr = {Vector3D(0.125, -2.5, 3.0),
                                                 Vector3D(4.0, 0.0, -0.75)};
    const std::string path = test_support::make_temporary_path(".q3ip");
    write_interpolation_record(record, path);
    const InterpolationRecord restored = read_interpolation_record(path);
    std::remove(path.c_str());

    ASSERT_EQ(restored.wannier.trial_orbital_centers_bohr.size(), 2u);
    expect_equal(restored.wannier.trial_orbital_centers_bohr[0], Vector3D(0.125, -2.5, 3.0));
    expect_equal(restored.wannier.trial_orbital_centers_bohr[1], Vector3D(4.0, 0.0, -0.75));
}

TEST(InterpolationRecordCheckpoint, ARecordWithoutTrialOrbitalCentersRoundTripsEmpty) {
    InterpolationRecord record = populated_record();
    record.wannier.trial_orbital_centers_bohr.clear();
    const std::string path = test_support::make_temporary_path(".q3ip");
    write_interpolation_record(record, path);
    const InterpolationRecord restored = read_interpolation_record(path);
    std::remove(path.c_str());
    EXPECT_TRUE(restored.wannier.trial_orbital_centers_bohr.empty());
}

TEST(InterpolationRecordCheckpoint, RoundTripsARecordWithoutOverlaps) {
    /// An imported record carries no overlaps; that must survive the file, not become garbage.
    InterpolationRecord written = populated_record();
    written.wannier.overlap_matrices.clear();
    written.wannier.neighbor_directions.clear();
    const std::string path = test_support::make_temporary_path(".q3ip");
    write_interpolation_record(written, path);
    const InterpolationRecord read = read_interpolation_record(path);
    std::remove(path.c_str());
    EXPECT_TRUE(read.wannier.overlap_matrices.empty());
    EXPECT_TRUE(read.wannier.neighbor_directions.empty());
    EXPECT_EQ(read.wannier.eigenvalues_rydberg, written.wannier.eigenvalues_rydberg);
}

TEST(InterpolationRecordCheckpoint, RefusesAFileThatIsNotARecord) {
    const std::string path = test_support::make_temporary_path(".q3ip");
    {
        std::ofstream stream(path, std::ios::binary);
        stream << "this is not an interpolation record, not even close";
    }
    EXPECT_THROW(read_interpolation_record(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(InterpolationRecordCheckpoint, RefusesAnUnknownVersion) {
    const std::string path = test_support::make_temporary_path(".q3ip");
    {
        std::ofstream stream(path, std::ios::binary);
        const uint32_t magic = internal::interpolation_checkpoint::magic;
        const uint32_t version = internal::interpolation_checkpoint::version + 7;
        stream.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
    }
    EXPECT_THROW(read_interpolation_record(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(InterpolationRecordCheckpoint, RefusesAMissingFileAndAMethodlessRecord) {
    EXPECT_THROW(read_interpolation_record("/nonexistent/directory/interpolation.q3ip"),
                 std::runtime_error);
    InterpolationRecord record;
    EXPECT_THROW(write_interpolation_record(record, test_support::make_temporary_path(".q3ip")),
                 std::invalid_argument);
}
