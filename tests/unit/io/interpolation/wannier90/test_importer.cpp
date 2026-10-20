#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "io/interpolation/wannier90/exporter.hpp"
#include "io/interpolation/wannier90/importer.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

namespace {

/**
 * Two k-points, two bands, two trial orbitals, one neighbour direction; distinct values
 * throughout.
 */
InterpolationRecord exportable_record() {
    InterpolationRecord record;
    record.method = "wannier";
    record.real_lattice_bohr[0] = Vector3D(-5.1672, 0.0, 5.1672);
    record.real_lattice_bohr[1] = Vector3D(0.0, 5.1672, 5.1672);
    record.real_lattice_bohr[2] = Vector3D(-5.1672, 5.1672, 0.0);
    record.reciprocal_lattice_inverse_bohr[0] = Vector3D(-0.6080, -0.6080, 0.6080);
    record.reciprocal_lattice_inverse_bohr[1] = Vector3D(0.6080, 0.6080, 0.6080);
    record.reciprocal_lattice_inverse_bohr[2] = Vector3D(-0.6080, 0.6080, -0.6080);
    WannierInputs& wannier = record.wannier;
    wannier.kpoint_grid_dimensions = {2, 1, 1};
    wannier.kpoint_coordinates = {Vector3D(0.0, 0.0, 0.0), Vector3D(0.5, 0.0, 0.0)};
    wannier.number_of_bands = 2;
    wannier.number_of_trial_orbitals = 2;
    wannier.fermi_energy_rydberg = 0.42;
    KPointNeighborDirection direction;
    direction.grid_step = IntegerVector3D{1, 0, 0};
    direction.reciprocal_lattice_translation = Vector3D(0.3, -0.1, 0.05);
    direction.weight = 0.5;
    wannier.neighbor_directions = {direction};
    wannier.overlap_matrices = {
        {{Complex(1.0, 0.0), Complex(0.1, 0.2), Complex(-0.1, 0.05), Complex(0.9, 0.0)}},
        {{Complex(0.8, 0.1), Complex(0.0, 0.0), Complex(0.2, 0.0), Complex(1.0, -0.1)}}};
    wannier.projection_matrices = {
        {Complex(0.95, 0.05), Complex(0.1, 0.0), Complex(0.0, -0.2), Complex(0.9, 0.1)},
        {Complex(0.7, 0.1), Complex(0.2, 0.0), Complex(-0.1, 0.3), Complex(0.9, 0.0)}};
    wannier.eigenvalues_rydberg = {{-0.5, 0.25}, {-0.4, 0.3}};
    wannier.projectabilities = {{0.9125, 0.86}, {0.55, 0.94}};
    return record;
}

std::string fresh_directory() {
    std::string directory = test_support::make_temporary_path("_wannier90_import");
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

} // namespace

TEST(Wannier90Importer, RoundTripsWhatTheInterpolationUsesThroughTheFiles) {
    const std::string directory = fresh_directory();
    const InterpolationRecord written = exportable_record();
    Wannier90Exporter().export_to_directory(written, directory, "si");

    const InterpolationRecord imported = import_wannier90_files(directory + "/si", {2, 1, 1});
    std::filesystem::remove_all(directory);

    EXPECT_EQ(imported.method, "wannier");
    const WannierInputs& expected = written.wannier;
    const WannierInputs& actual = imported.wannier;
    EXPECT_EQ(actual.kpoint_grid_dimensions, (std::array<int, 3>{2, 1, 1}));
    EXPECT_EQ(actual.number_of_bands, 2);
    EXPECT_EQ(actual.number_of_trial_orbitals, 2);
    ASSERT_EQ(actual.kpoint_coordinates.size(), 2u);
    for (std::size_t index = 0; index < 2; index++) {
        EXPECT_NEAR(actual.kpoint_coordinates[index].x, expected.kpoint_coordinates[index].x, 1e-8);
        EXPECT_NEAR(actual.kpoint_coordinates[index].y, expected.kpoint_coordinates[index].y, 1e-8);
        EXPECT_NEAR(actual.kpoint_coordinates[index].z, expected.kpoint_coordinates[index].z, 1e-8);
    }
    /// Angstrom out, bohr back in: the lattice survives the unit round trip.
    for (int row = 0; row < 3; row++) {
        EXPECT_NEAR(imported.real_lattice_bohr[row].x, written.real_lattice_bohr[row].x, 1e-6);
        EXPECT_NEAR(imported.real_lattice_bohr[row].y, written.real_lattice_bohr[row].y, 1e-6);
        EXPECT_NEAR(imported.real_lattice_bohr[row].z, written.real_lattice_bohr[row].z, 1e-6);
        EXPECT_NEAR(imported.reciprocal_lattice_inverse_bohr[row].x,
                    written.reciprocal_lattice_inverse_bohr[row].x,
                    1e-6);
    }
    /// eV out, Rydberg back in.
    ASSERT_EQ(actual.eigenvalues_rydberg.size(), 2u);
    for (std::size_t kpoint = 0; kpoint < 2; kpoint++) {
        for (std::size_t band = 0; band < 2; band++) {
            EXPECT_NEAR(actual.eigenvalues_rydberg[kpoint][band],
                        expected.eigenvalues_rydberg[kpoint][band],
                        1e-10);
            EXPECT_NEAR(std::abs(actual.projection_matrices[kpoint][band] -
                                 expected.projection_matrices[kpoint][band]),
                        0.0,
                        1e-10);
            EXPECT_NEAR(std::abs(actual.projection_matrices[kpoint][2 + band] -
                                 expected.projection_matrices[kpoint][2 + band]),
                        0.0,
                        1e-10);
        }
    }
    /// Recomputed from the projections, not read: p_nk = sum_m |A_mn|^2.
    ASSERT_EQ(actual.projectabilities.size(), 2u);
    for (std::size_t kpoint = 0; kpoint < 2; kpoint++) {
        for (int band = 0; band < 2; band++) {
            Real expected_projectability = 0.0;
            for (int orbital = 0; orbital < 2; orbital++) {
                expected_projectability +=
                    std::norm(expected.projection_matrices[kpoint][orbital * 2 + band]);
            }
            EXPECT_NEAR(actual.projectabilities[kpoint][band], expected_projectability, 1e-9);
        }
    }
    /// Deliberately not imported: nothing consumes them yet.
    EXPECT_TRUE(actual.overlap_matrices.empty());
    EXPECT_TRUE(actual.neighbor_directions.empty());
}

TEST(Wannier90Importer, RefusesAGridThatDoesNotMatchTheFiles) {
    const std::string directory = fresh_directory();
    Wannier90Exporter().export_to_directory(exportable_record(), directory, "si");
    EXPECT_THROW(import_wannier90_files(directory + "/si", {2, 2, 1}), std::runtime_error);
    EXPECT_THROW(import_wannier90_files(directory + "/si", {0, 1, 1}), std::invalid_argument);
    std::filesystem::remove_all(directory);
}

TEST(Wannier90Importer, AnImportedRecordExportsWithoutAnEmptyOverlapFile) {
    const std::string directory = fresh_directory();
    Wannier90Exporter().export_to_directory(exportable_record(), directory, "si");
    const InterpolationRecord imported = import_wannier90_files(directory + "/si", {2, 1, 1});

    const std::string again = fresh_directory() + "_again";
    std::filesystem::create_directories(again);
    Wannier90Exporter().export_to_directory(imported, again, "si");
    EXPECT_TRUE(std::filesystem::exists(again + "/si.amn"));
    EXPECT_TRUE(std::filesystem::exists(again + "/si.eig"));
    EXPECT_FALSE(std::filesystem::exists(again + "/si.nnkp"))
        << "an imported record has no neighbour table; a .nnkp with an empty nnkpts block would "
           "pass for a real one";
    EXPECT_FALSE(std::filesystem::exists(again + "/si.mmn"))
        << "an imported record has no overlaps; writing an empty .mmn would pass for a real one";
    std::filesystem::remove_all(directory);
    std::filesystem::remove_all(again);
}
