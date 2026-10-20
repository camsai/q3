#pragma once

/**
 * CubeVolumetricFieldReader — read a Gaussian cube file back into a
 * VolumetricField.
 *
 * The other half of cube.hpp. It exists so that what a writer claims can be
 * checked against what a reader sees rather than against the array the writer
 * was handed: `tests/unit/io/volumetric/test_cube.cpp` integrates the density
 * **out of the written file** and compares it against the electron count, and
 * asserts the grid came back in the same axis order it went out in. A writer
 * whose only check is its own in-memory input cannot catch a transposed grid
 * or a missed unit, which are exactly the two defects that produce a file
 * looking correct in a viewer.
 *
 * It also makes a cube from elsewhere readable: the format's two signed fields
 * are honoured, so a file written in Ångström (negative voxel counts) comes
 * back in bohr like any other.
 *
 * Not honoured, and refused rather than misread: a negative atom count, which
 * marks a file carrying several datasets (molecular orbitals) with an extra
 * index record this reader would otherwise swallow as field values.
 */

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/volumetric/field.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_1_structure/crystal/periodic_table.hpp"

namespace q3 {

class CubeVolumetricFieldReader {
  public:
    [[nodiscard]] VolumetricField read(const std::string& path) const {
        std::ifstream stream(path);
        if (!stream)
            throw std::runtime_error("Cannot read cube file: " + path);

        VolumetricField field;
        std::string first_comment;
        std::string second_comment;
        std::getline(stream, first_comment);
        std::getline(stream, second_comment);
        field.description = first_comment;

        int number_of_atoms = 0;
        Vector3D origin;
        read_line_values(stream, path, number_of_atoms, origin);
        if (number_of_atoms < 0) {
            throw std::runtime_error(
                "Cube file '" + path +
                "' declares a negative atom count, which marks a multi-dataset (orbital) cube. "
                "Q3 reads single-field cubes only.");
        }

        for (int axis = 0; axis < 3; axis++) {
            int voxel_count = 0;
            Vector3D voxel;
            read_line_values(stream, path, voxel_count, voxel);
            /**
             * A negative voxel count is the format's way of saying the lengths
             * on this file's header lines are Ångström rather than bohr.
             */
            const Real to_bohr = voxel_count < 0 ? constants::ANGSTROM_TO_BOHR : 1.0;
            const int dimension = voxel_count < 0 ? -voxel_count : voxel_count;
            if (dimension <= 0)
                throw std::runtime_error("Cube file '" + path + "' has a zero grid dimension.");
            field.grid_dimensions[axis] = dimension;
            for (int component = 0; component < 3; component++)
                field.lattice_vectors_bohr[axis][component] =
                    voxel[component] * to_bohr * dimension;
        }

        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            std::istringstream line = next_content_line(stream, path);
            int atomic_number = 0;
            Real charge = 0.0;
            Vector3D position;
            line >> atomic_number >> charge >> position.x >> position.y >> position.z;
            if (!line)
                throw std::runtime_error("Cube file '" + path + "' ends inside its atom list.");
            field.atom_species_indices.push_back(species_index_for(field, atomic_number, charge));
            field.atom_positions_crystal.push_back(
                to_crystal_coordinates(field.lattice_vectors_bohr, position));
        }

        const int number_of_grid_points = field.number_of_grid_points();
        field.values.reserve(number_of_grid_points);
        Real value = 0.0;
        while (static_cast<int>(field.values.size()) < number_of_grid_points && stream >> value)
            field.values.push_back(value);
        if (static_cast<int>(field.values.size()) != number_of_grid_points) {
            throw std::runtime_error("Cube file '" + path + "' holds " +
                                     std::to_string(field.values.size()) + " values but declares " +
                                     std::to_string(number_of_grid_points) + ".");
        }
        return field;
    }

  private:
    /// Read one line as an integer followed by three reals.
    static void
    read_line_values(std::ifstream& stream, const std::string& path, int& count, Vector3D& vector) {
        std::istringstream line = next_content_line(stream, path);
        line >> count >> vector.x >> vector.y >> vector.z;
        if (!line)
            throw std::runtime_error("Cube file '" + path + "' has a malformed header line.");
    }

    static std::istringstream next_content_line(std::ifstream& stream, const std::string& path) {
        std::string text;
        while (std::getline(stream, text)) {
            if (text.find_first_not_of(" \t\r") != std::string::npos)
                return std::istringstream(text);
        }
        throw std::runtime_error("Cube file '" + path + "' ends before its header does.");
    }

    /// Find or append the species an atom line names, keeping species order stable.
    static int species_index_for(VolumetricField& field, int atomic_number, Real valence_charge) {
        const std::string& symbol = PeriodicTable::element_symbol_by_atomic_number(atomic_number);
        for (std::size_t index = 0; index < field.species_symbols.size(); index++) {
            if (field.species_symbols[index] == symbol)
                return static_cast<int>(index);
        }
        field.species_symbols.push_back(symbol);
        field.species_valence_charges.push_back(valence_charge);
        return static_cast<int>(field.species_symbols.size()) - 1;
    }

    static Vector3D to_crystal_coordinates(const Matrix3x3& lattice_vectors_bohr,
                                           const Vector3D& cartesian) {
        /**
         * Rows of the lattice matrix are a1, a2, a3, so the fractional
         * coordinates are `r * A^-1` with r a row vector.
         */
        const Matrix3x3 inverse = lattice_vectors_bohr.inverse();
        Vector3D fractional;
        for (int axis = 0; axis < 3; axis++) {
            fractional[axis] = cartesian.x * inverse[0][axis] + cartesian.y * inverse[1][axis] +
                               cartesian.z * inverse[2][axis];
        }
        return fractional;
    }
};

} // namespace q3
