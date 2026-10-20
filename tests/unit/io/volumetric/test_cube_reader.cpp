#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "io/volumetric/cube_reader.hpp"
#include "layer_0_core/types/constants.hpp"

using namespace q3;

namespace {

std::filesystem::path write_file(const std::string& name, const std::string& contents) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream stream(path);
    stream << contents;
    stream.close();
    return path;
}

/// A 2x2x2 cube over an orthorhombic cell, in bohr, with eight distinct values.
const char* const bohr_cube = R"(Q3 charge density
units: bohr
    1    0.000000    0.000000    0.000000
    2    1.000000    0.000000    0.000000
    2    0.000000    2.000000    0.000000
    2    0.000000    0.000000    3.000000
   14    4.000000    0.500000    1.000000    1.500000
 0.0 1.0
 2.0 3.0
 4.0 5.0
 6.0 7.0
)";

} // namespace

TEST(CubeVolumetricFieldReader, ReadsTheGridCellAndAtoms) {
    const auto path = write_file("q3_test_reader_bohr.cube", bohr_cube);
    const VolumetricField field = CubeVolumetricFieldReader().read(path.string());

    EXPECT_EQ(field.grid_dimensions[0], 2);
    EXPECT_EQ(field.grid_dimensions[1], 2);
    EXPECT_EQ(field.grid_dimensions[2], 2);
    /// The lattice vector is the voxel vector times the voxel count.
    EXPECT_NEAR(field.lattice_vectors_bohr[0][0], 2.0, 1e-12);
    EXPECT_NEAR(field.lattice_vectors_bohr[1][1], 4.0, 1e-12);
    EXPECT_NEAR(field.lattice_vectors_bohr[2][2], 6.0, 1e-12);
    EXPECT_NEAR(field.cell_volume_bohr3(), 48.0, 1e-12);

    ASSERT_EQ(field.number_of_atoms(), 1);
    EXPECT_EQ(field.atomic_number(0), 14);
    EXPECT_NEAR(field.species_valence_charges[0], 4.0, 1e-12);
    EXPECT_NEAR(field.atom_positions_crystal[0].x, 0.25, 1e-12);
    EXPECT_NEAR(field.atom_positions_crystal[0].y, 0.25, 1e-12);
    EXPECT_NEAR(field.atom_positions_crystal[0].z, 0.25, 1e-12);

    ASSERT_EQ(field.values.size(), 8u);
    EXPECT_DOUBLE_EQ(field.value(0, 0, 0), 0.0);
    EXPECT_DOUBLE_EQ(field.value(0, 0, 1), 1.0);
    EXPECT_DOUBLE_EQ(field.value(0, 1, 0), 2.0);
    EXPECT_DOUBLE_EQ(field.value(1, 0, 0), 4.0);
    EXPECT_DOUBLE_EQ(field.value(1, 1, 1), 7.0);

    std::filesystem::remove(path);
}

TEST(CubeVolumetricFieldReader, NegativeVoxelCountsMeanAngstromAndAreConverted) {
    std::string angstrom_cube = bohr_cube;
    const std::string::size_type first = angstrom_cube.find("    2    1.000000");
    ASSERT_NE(first, std::string::npos);
    angstrom_cube.replace(first, std::string("    2").size(), "   -2");

    const auto path = write_file("q3_test_reader_angstrom.cube", angstrom_cube);
    const VolumetricField field = CubeVolumetricFieldReader().read(path.string());

    EXPECT_EQ(field.grid_dimensions[0], 2);
    EXPECT_NEAR(field.lattice_vectors_bohr[0][0], 2.0 * constants::ANGSTROM_TO_BOHR, 1e-12);
    /// The other two axes kept their positive counts, so they are still bohr.
    EXPECT_NEAR(field.lattice_vectors_bohr[1][1], 4.0, 1e-12);

    std::filesystem::remove(path);
}

TEST(CubeVolumetricFieldReader, RefusesAMultiDatasetCube) {
    std::string orbital_cube = bohr_cube;
    const std::string::size_type atoms = orbital_cube.find("    1    0.000000");
    ASSERT_NE(atoms, std::string::npos);
    orbital_cube.replace(atoms, std::string("    1").size(), "   -1");

    const auto path = write_file("q3_test_reader_orbital.cube", orbital_cube);
    EXPECT_THROW(CubeVolumetricFieldReader().read(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(CubeVolumetricFieldReader, RefusesAFileWithTooFewValues) {
    std::string truncated = bohr_cube;
    const std::string::size_type last = truncated.find(" 6.0 7.0\n");
    ASSERT_NE(last, std::string::npos);
    truncated.erase(last);

    const auto path = write_file("q3_test_reader_truncated.cube", truncated);
    EXPECT_THROW(CubeVolumetricFieldReader().read(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(CubeVolumetricFieldReader, RefusesAMissingFile) {
    const auto path = std::filesystem::temp_directory_path() / "q3_test_reader_absent.cube";
    std::filesystem::remove(path);
    EXPECT_THROW(CubeVolumetricFieldReader().read(path.string()), std::runtime_error);
}
