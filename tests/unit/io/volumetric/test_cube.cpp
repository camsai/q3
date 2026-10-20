#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "io/volumetric/cube.hpp"
#include "io/volumetric/cube_reader.hpp"

using namespace q3;

namespace {

constexpr Real electron_count = 8.0;

/**
 * Silicon's cell, a 3x4x5 grid, and a constant density holding exactly eight
 * electrons.
 *
 * Constant on purpose: the integral is then `rho * Omega` with no quadrature
 * error at all, so a test of it fails only if the file's units, its volume or
 * its point count are wrong -- which are the three things a cube writer can
 * get wrong while still producing a file a viewer opens.
 *
 * The lattice constant is silicon's real one rather than a round number, also
 * on purpose. Divided by the grid counts it is a voxel vector no short decimal
 * represents, so the *header's* precision enters the integral -- and it does:
 * at the format's conventional six decimals this same check reads 7.99994
 * electrons back out of the file, because the cell volume it reconstructs is
 * wrong by 7.5e-6. A round cell would have hidden that.
 */
VolumetricField make_constant_density() {
    const Real half_lattice_constant = 5.1672017;
    VolumetricField field;
    field.grid_dimensions = {3, 4, 5};
    field.lattice_vectors_bohr[0] = {0.0, half_lattice_constant, half_lattice_constant};
    field.lattice_vectors_bohr[1] = {half_lattice_constant, 0.0, half_lattice_constant};
    field.lattice_vectors_bohr[2] = {half_lattice_constant, half_lattice_constant, 0.0};
    field.species_symbols = {"Si"};
    field.species_valence_charges = {4.0};
    field.atom_species_indices = {0, 0};
    field.atom_positions_crystal = {{0.875, 0.875, 0.875}, {0.125, 0.125, 0.125}};
    field.description = "charge density";
    field.unit = "e/bohr^3";
    field.is_a_density = true;
    field.values.assign(3 * 4 * 5, 0.0);
    const Real volume = field.cell_volume_bohr3();
    for (Real& value : field.values)
        value = electron_count / volume;
    return field;
}

/// The same cell and grid, with each sample carrying its own grid index.
VolumetricField make_indexed_field() {
    VolumetricField field = make_constant_density();
    for (int index_1 = 0; index_1 < 3; index_1++)
        for (int index_2 = 0; index_2 < 4; index_2++)
            for (int index_3 = 0; index_3 < 5; index_3++)
                field.values[(index_1 * 4 + index_2) * 5 + index_3] =
                    100.0 * index_1 + 10.0 * index_2 + index_3;
    return field;
}

std::filesystem::path temporary_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}

/// Every whitespace-separated token of a file, in order.
std::vector<std::string> read_tokens(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token)
        tokens.push_back(token);
    return tokens;
}

} // namespace

TEST(CubeVolumetricFieldWriter, HeaderDescribesTheGridInBohr) {
    const VolumetricField field = make_constant_density();
    const auto path = temporary_path("q3_test_density.cube");
    CubeVolumetricFieldWriter().write(path.string(), field);

    std::ifstream stream(path);
    std::string first_comment;
    std::string second_comment;
    std::getline(stream, first_comment);
    std::getline(stream, second_comment);

    int number_of_atoms = 0;
    Real origin_x = 0.0;
    Real origin_y = 0.0;
    Real origin_z = 0.0;
    stream >> number_of_atoms >> origin_x >> origin_y >> origin_z;
    EXPECT_EQ(number_of_atoms, 2);
    EXPECT_DOUBLE_EQ(origin_x, 0.0);

    /**
     * Positive voxel counts are what says the lengths are bohr, and each voxel
     * vector is the lattice vector divided by its own count.
     */
    for (int axis = 0; axis < 3; axis++) {
        int voxel_count = 0;
        Vector3D voxel;
        stream >> voxel_count >> voxel.x >> voxel.y >> voxel.z;
        EXPECT_GT(voxel_count, 0);
        EXPECT_EQ(voxel_count, field.grid_dimensions[axis]);
        for (int component = 0; component < 3; component++) {
            EXPECT_NEAR(
                voxel[component], field.lattice_vectors_bohr[axis][component] / voxel_count, 1e-11);
        }
    }

    int atomic_number = 0;
    Real charge = 0.0;
    Vector3D position;
    stream >> atomic_number >> charge >> position.x >> position.y >> position.z;
    EXPECT_EQ(atomic_number, 14);
    EXPECT_NEAR(charge, 4.0, 1e-9);
    const Vector3D expected = field.cartesian_position_bohr(0);
    EXPECT_NEAR(position.x, expected.x, 1e-11);
    EXPECT_NEAR(position.y, expected.y, 1e-11);
    EXPECT_NEAR(position.z, expected.z, 1e-11);

    std::filesystem::remove(path);
}

TEST(CubeVolumetricFieldWriter, DataIsWrittenWithTheThirdAxisFastest) {
    const VolumetricField field = make_indexed_field();
    const auto path = temporary_path("q3_test_indexed.cube");
    CubeVolumetricFieldWriter().write(path.string(), field);

    const std::vector<std::string> tokens = read_tokens(path);
    /// Two comment lines are not tokens we can count on, so skip past the header by shape.
    std::ifstream stream(path);
    std::string line;
    for (int header_line = 0; header_line < 6 + field.number_of_atoms(); header_line++)
        std::getline(stream, line);

    std::vector<Real> values;
    Real value = 0.0;
    while (stream >> value)
        values.push_back(value);
    ASSERT_EQ(values.size(), static_cast<std::size_t>(field.number_of_grid_points()));

    /**
     * The cube format's own ordering: i1 outermost, i3 innermost. Written in
     * any other order the file still opens and still looks like a crystal.
     */
    std::size_t flat_index = 0;
    for (int index_1 = 0; index_1 < field.grid_dimensions[0]; index_1++)
        for (int index_2 = 0; index_2 < field.grid_dimensions[1]; index_2++)
            for (int index_3 = 0; index_3 < field.grid_dimensions[2]; index_3++)
                EXPECT_DOUBLE_EQ(values[flat_index++], field.value(index_1, index_2, index_3));

    EXPECT_FALSE(tokens.empty());
    std::filesystem::remove(path);
}

TEST(CubeVolumetricFieldWriter, IntegralOfTheWrittenFileIsTheElectronCount) {
    const VolumetricField field = make_constant_density();
    const auto path = temporary_path("q3_test_integral.cube");
    CubeVolumetricFieldWriter().write(path.string(), field);

    /**
     * Read back rather than reusing the array that was written: that is the
     * whole point of the check. The integral is computed out of the file's own
     * cell, its own point count and its own samples.
     */
    const VolumetricField read_back = CubeVolumetricFieldReader().read(path.string());
    EXPECT_NEAR(read_back.integral(), electron_count, 1e-8);
    EXPECT_NEAR(read_back.cell_volume_bohr3(), field.cell_volume_bohr3(), 1e-8);

    std::filesystem::remove(path);
}

TEST(CubeVolumetricFieldWriter, RoundTripPreservesTheGridAndTheValues) {
    const VolumetricField field = make_indexed_field();
    const auto path = temporary_path("q3_test_roundtrip.cube");
    CubeVolumetricFieldWriter().write(path.string(), field);
    const VolumetricField read_back = CubeVolumetricFieldReader().read(path.string());

    ASSERT_EQ(read_back.grid_dimensions, field.grid_dimensions);
    ASSERT_EQ(read_back.number_of_atoms(), field.number_of_atoms());
    for (int axis = 0; axis < 3; axis++)
        for (int component = 0; component < 3; component++)
            EXPECT_NEAR(read_back.lattice_vectors_bohr[axis][component],
                        field.lattice_vectors_bohr[axis][component],
                        1e-10);
    for (int atom_index = 0; atom_index < field.number_of_atoms(); atom_index++) {
        EXPECT_EQ(read_back.atomic_number(atom_index), field.atomic_number(atom_index));
        const Vector3D expected = field.atom_positions_crystal[atom_index];
        const Vector3D actual = read_back.atom_positions_crystal[atom_index];
        EXPECT_NEAR(actual.x, expected.x, 1e-11);
        EXPECT_NEAR(actual.y, expected.y, 1e-11);
        EXPECT_NEAR(actual.z, expected.z, 1e-11);
    }
    ASSERT_EQ(read_back.values.size(), field.values.size());
    for (std::size_t index = 0; index < field.values.size(); index++)
        EXPECT_NEAR(read_back.values[index], field.values[index], 1e-6);

    std::filesystem::remove(path);
}
