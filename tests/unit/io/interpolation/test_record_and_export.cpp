#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "io/input/data.hpp"
#include "io/interpolation/checkpoint.hpp"
#include "io/interpolation/record_and_export.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

namespace {

InterpolationRecord small_record() {
    InterpolationRecord record;
    record.method = "wannier";
    for (int row = 0; row < 3; row++) {
        record.real_lattice_bohr[row] =
            Vector3D(row == 0 ? 10.0 : 0.0, row == 1 ? 10.0 : 0.0, row == 2 ? 10.0 : 0.0);
        record.reciprocal_lattice_inverse_bohr[row] =
            Vector3D(row == 0 ? 0.6283 : 0.0, row == 1 ? 0.6283 : 0.0, row == 2 ? 0.6283 : 0.0);
    }
    WannierInputs& wannier = record.wannier;
    wannier.kpoint_grid_dimensions = {2, 1, 1};
    wannier.kpoint_coordinates = {Vector3D(0.0, 0.0, 0.0), Vector3D(0.5, 0.0, 0.0)};
    wannier.number_of_bands = 1;
    wannier.number_of_trial_orbitals = 1;
    KPointNeighborDirection direction;
    direction.grid_step = IntegerVector3D{1, 0, 0};
    direction.reciprocal_lattice_translation = Vector3D(0.3142, 0.0, 0.0);
    direction.weight = 1.0;
    wannier.neighbor_directions = {direction};
    wannier.overlap_matrices = {{{Complex(0.9, 0.1)}}, {{Complex(0.9, -0.1)}}};
    wannier.projection_matrices = {{Complex(1.0, 0.0)}, {Complex(0.8, 0.2)}};
    wannier.eigenvalues_rydberg = {{-0.5}, {-0.3}};
    wannier.projectabilities = {{1.0}, {0.68}};
    return record;
}

std::string fresh_directory() {
    std::string directory = test_support::make_temporary_path("_interpolation_export");
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

} // namespace

TEST(InterpolationRecordAndExport, WritesTheRecordUnderTheSaveDirectory) {
    const std::string directory = fresh_directory();
    InputData input;
    input.output_directory = directory;
    input.prefix = "si";
    input.interpolation.method = "wannier";

    write_and_export_interpolation_record(small_record(), input);

    const std::string record_path = interpolation_record_path(directory, "si");
    EXPECT_EQ(record_path, directory + "/si.save/interpolation.q3ip");
    ASSERT_TRUE(std::filesystem::exists(record_path));
    const InterpolationRecord read = read_interpolation_record(record_path);
    EXPECT_EQ(read.method, "wannier");
    EXPECT_EQ(read.wannier.eigenvalues_rydberg, small_record().wannier.eigenvalues_rydberg);
    /// No export format was listed, so no Wannier90 files appear beside it.
    EXPECT_FALSE(std::filesystem::exists(directory + "/si.amn"));
    std::filesystem::remove_all(directory);
}

TEST(InterpolationRecordAndExport, ExportsEveryListedFormatBesideTheRecord) {
    const std::string directory = fresh_directory();
    InputData input;
    input.output_directory = directory;
    input.prefix = "si";
    input.interpolation.method = "wannier";
    input.interpolation.export_formats = {"wannier90"};

    write_and_export_interpolation_record(small_record(), input);

    for (const char* extension : {".mmn", ".amn", ".eig", ".nnkp"}) {
        EXPECT_TRUE(std::filesystem::exists(directory + "/si" + extension)) << extension;
    }
    EXPECT_TRUE(std::filesystem::exists(interpolation_record_path(directory, "si")));
    std::filesystem::remove_all(directory);
}

TEST(InterpolationRecordAndExport, AnUnknownFormatIsRefusedAfterTheRecordIsSafe) {
    /**
     * The record is the step's output and goes first; a bad export name must
     * not cost the user the model they computed.
     */
    const std::string directory = fresh_directory();
    InputData input;
    input.output_directory = directory;
    input.prefix = "si";
    input.interpolation.method = "wannier";
    input.interpolation.export_formats = {"wannier95"};

    EXPECT_THROW(write_and_export_interpolation_record(small_record(), input), std::runtime_error);
    EXPECT_TRUE(std::filesystem::exists(interpolation_record_path(directory, "si")));
    std::filesystem::remove_all(directory);
}
