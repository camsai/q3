#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "io/volumetric/xsf.hpp"
#include "layer_0_core/types/constants.hpp"

using namespace q3;

namespace {

VolumetricField make_indexed_field() {
    VolumetricField field;
    field.grid_dimensions = {3, 4, 5};
    field.lattice_vectors_bohr[0] = {0.0, 5.1, 5.1};
    field.lattice_vectors_bohr[1] = {5.1, 0.0, 5.1};
    field.lattice_vectors_bohr[2] = {5.1, 5.1, 0.0};
    field.species_symbols = {"Si"};
    field.species_valence_charges = {4.0};
    field.atom_species_indices = {0, 0};
    field.atom_positions_crystal = {{0.875, 0.875, 0.875}, {0.125, 0.125, 0.125}};
    field.description = "charge density";
    field.unit = "e/bohr^3";
    field.is_a_density = true;
    field.values.assign(3 * 4 * 5, 0.0);
    for (int index_1 = 0; index_1 < 3; index_1++)
        for (int index_2 = 0; index_2 < 4; index_2++)
            for (int index_3 = 0; index_3 < 5; index_3++)
                field.values[(index_1 * 4 + index_2) * 5 + index_3] =
                    100.0 * index_1 + 10.0 * index_2 + index_3;
    return field;
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(stream, line))
        lines.push_back(line);
    return lines;
}

std::size_t index_of_line_containing(const std::vector<std::string>& lines,
                                     const std::string& needle) {
    for (std::size_t index = 0; index < lines.size(); index++) {
        if (lines[index].find(needle) != std::string::npos)
            return index;
    }
    return lines.size();
}

} // namespace

TEST(XsfVolumetricFieldWriter, StructureBlockIsInAngstrom) {
    const VolumetricField field = make_indexed_field();
    const auto path = std::filesystem::temp_directory_path() / "q3_test_structure.xsf";
    XsfVolumetricFieldWriter().write(path.string(), field);
    const std::vector<std::string> lines = read_lines(path);

    ASSERT_LT(index_of_line_containing(lines, "CRYSTAL"), lines.size());
    const std::size_t primvec = index_of_line_containing(lines, "PRIMVEC");
    ASSERT_LT(primvec, lines.size());
    for (int axis = 0; axis < 3; axis++) {
        std::istringstream stream(lines[primvec + 1 + axis]);
        Vector3D vector;
        stream >> vector.x >> vector.y >> vector.z;
        for (int component = 0; component < 3; component++) {
            EXPECT_NEAR(vector[component],
                        field.lattice_vectors_bohr[axis][component] *
                            constants::BOHR_RADIUS_ANGSTROM,
                        1e-8);
        }
    }

    const std::size_t primcoord = index_of_line_containing(lines, "PRIMCOORD");
    ASSERT_LT(primcoord, lines.size());
    std::istringstream counts(lines[primcoord + 1]);
    int number_of_atoms = 0;
    int one = 0;
    counts >> number_of_atoms >> one;
    EXPECT_EQ(number_of_atoms, 2);
    EXPECT_EQ(one, 1);

    std::istringstream first_atom(lines[primcoord + 2]);
    int atomic_number = 0;
    Vector3D position;
    first_atom >> atomic_number >> position.x >> position.y >> position.z;
    EXPECT_EQ(atomic_number, 14);
    const Vector3D expected = field.cartesian_position_bohr(0);
    EXPECT_NEAR(position.x, expected.x * constants::BOHR_RADIUS_ANGSTROM, 1e-8);

    std::filesystem::remove(path);
}

TEST(XsfVolumetricFieldWriter, DatagridDuplicatesTheBoundaryAndRunsTheFirstAxisFastest) {
    const VolumetricField field = make_indexed_field();
    const auto path = std::filesystem::temp_directory_path() / "q3_test_datagrid.xsf";
    XsfVolumetricFieldWriter().write(path.string(), field);

    std::ifstream stream(path);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("BEGIN_DATAGRID_3D") != std::string::npos)
            break;
    }
    ASSERT_NE(line.find("BEGIN_DATAGRID_3D"), std::string::npos);

    int dimension_1 = 0;
    int dimension_2 = 0;
    int dimension_3 = 0;
    stream >> dimension_1 >> dimension_2 >> dimension_3;
    /**
     * An XSF general datagrid spans its vectors including the far end, so a
     * periodic n-point grid is written as n+1 points with the last a copy of
     * the first. Anything else leaves a gap at the cell boundary.
     */
    EXPECT_EQ(dimension_1, field.grid_dimensions[0] + 1);
    EXPECT_EQ(dimension_2, field.grid_dimensions[1] + 1);
    EXPECT_EQ(dimension_3, field.grid_dimensions[2] + 1);

    Vector3D origin;
    stream >> origin.x >> origin.y >> origin.z;
    EXPECT_DOUBLE_EQ(origin.x, 0.0);
    for (int axis = 0; axis < 3; axis++) {
        Vector3D span;
        stream >> span.x >> span.y >> span.z;
        EXPECT_NEAR(
            span.x, field.lattice_vectors_bohr[axis][0] * constants::BOHR_RADIUS_ANGSTROM, 1e-8);
    }

    std::vector<Real> values;
    values.reserve(static_cast<std::size_t>(dimension_1) * dimension_2 * dimension_3);
    Real value = 0.0;
    while (static_cast<int>(values.size()) < dimension_1 * dimension_2 * dimension_3 &&
           stream >> value) {
        values.push_back(value);
    }
    ASSERT_EQ(static_cast<int>(values.size()), dimension_1 * dimension_2 * dimension_3);

    std::size_t flat_index = 0;
    for (int index_3 = 0; index_3 < dimension_3; index_3++)
        for (int index_2 = 0; index_2 < dimension_2; index_2++)
            for (int index_1 = 0; index_1 < dimension_1; index_1++)
                EXPECT_DOUBLE_EQ(values[flat_index++],
                                 field.value(index_1 % field.grid_dimensions[0],
                                             index_2 % field.grid_dimensions[1],
                                             index_3 % field.grid_dimensions[2]));

    std::filesystem::remove(path);
}
