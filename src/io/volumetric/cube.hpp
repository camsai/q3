#pragma once

/**
 * CubeVolumetricFieldWriter — the Gaussian cube format.
 *
 * The format VESTA, XCrySDen, VMD, ASE and pymatgen all read, which is why it
 * is the first one Q3 writes. Implemented from the published description of
 * the format (Gaussian's "cubegen" documentation and the many public
 * restatements of it); no reference implementation was consulted.
 *
 * Layout, one line each:
 *
 *     comment                                  (two free-text lines)
 *     natoms   origin_x origin_y origin_z
 *     n1       a1/n1  (three Cartesian components)
 *     n2       a2/n2
 *     n3       a3/n3
 *     Z  charge  x y z                         (natoms of these)
 *     values, six per line
 *
 * **Conventions this writer chose, and the reasons.**
 *
 *   - *Bohr, and electrons/bohr^3.* A cube's lengths are bohr when the three
 *     voxel counts are positive and Ångström when they are negative; this
 *     writer always writes them positive and always writes bohr, which is the
 *     near-universal convention and also the unit the engine already works in,
 *     so nothing is converted and nothing can be converted twice. A density's
 *     samples are then electrons/bohr^3 and
 *     `Omega / (n1 n2 n3) * sum(values)` is the electron count —
 *     `tests/unit/io/volumetric/test_cube.cpp` asserts exactly that, reading
 *     it back out of the written file.
 *   - *The third axis varies fastest.* The value loop is `i1` outermost, `i3`
 *     innermost, which is the cube format's own requirement and, as it
 *     happens, the order VolumetricField already stores. A file written in any
 *     other order still opens, still looks like a crystal, and is wrong;
 *     `test_cube.cpp` pins the axis order with an asymmetric field whose value
 *     identifies its own grid point.
 *   - *No duplicated boundary.* The grid is periodic and each point appears
 *     once; the point at fractional coordinate 1 is the point at 0. (XSF
 *     differs — see xsf.hpp.)
 *   - *The atom "charge" column carries the valence charge*, not Z, because
 *     that is what the density in the file actually integrates to for a
 *     pseudopotential calculation. The atomic-number column is Z as usual.
 *
 * **Precision is wider than the conventional field widths**, deliberately, and
 * the reason is the electron count. The traditional layout writes the header
 * in `%12.6f` and the data in `%13.5E`; both are free-format in the format's
 * own description, and every reader in use splits on whitespace, so widening
 * them costs nothing and buys back two errors that are otherwise invisible.
 * Six decimals on a voxel vector round silicon's 0.1435333800 bohr to
 * 0.143533, which is a 7.5e-6 relative error in the *cell volume* and
 * therefore in every quantity integrated out of the file: a density holding
 * exactly eight electrons reads back as 7.99994. Five significant digits on
 * the samples would add a comparable error of its own. The header is written
 * with twelve decimals and the data with ten significant digits, which puts
 * both below 1e-9 — measured by `tests/regression/test_volumetric_export.py`,
 * which integrates a real silicon density out of the written file.
 */

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "io/volumetric/field.hpp"
#include "io/volumetric/writer.hpp"

namespace q3 {

class CubeVolumetricFieldWriter : public VolumetricFieldWriter {
  public:
    void write(const std::string& path, const VolumetricField& field) const override {
        std::ofstream stream(path);
        if (!stream)
            throw std::runtime_error("Cannot write cube file: " + path);

        const auto& dimensions = field.grid_dimensions;
        stream << "Q3 " << field.description << "\n";
        stream << "units: bohr, " << field.unit << "; grid " << dimensions[0] << "x"
               << dimensions[1] << "x" << dimensions[2] << ", third axis fastest\n";

        stream << std::fixed << std::setprecision(header_decimals);
        stream << std::setw(5) << field.number_of_atoms() << std::setw(header_width) << 0.0
               << std::setw(header_width) << 0.0 << std::setw(header_width) << 0.0 << "\n";
        for (int axis = 0; axis < 3; axis++) {
            stream << std::setw(5) << dimensions[axis];
            for (int component = 0; component < 3; component++) {
                stream << std::setw(header_width)
                       << field.lattice_vectors_bohr[axis][component] / dimensions[axis];
            }
            stream << "\n";
        }
        for (int atom_index = 0; atom_index < field.number_of_atoms(); atom_index++) {
            const Vector3D position = field.cartesian_position_bohr(atom_index);
            const int species_index = field.atom_species_indices[atom_index];
            stream << std::setw(5) << field.atomic_number(atom_index) << std::setw(header_width)
                   << field.species_valence_charges[species_index] << std::setw(header_width)
                   << position.x << std::setw(header_width) << position.y << std::setw(header_width)
                   << position.z << "\n";
        }

        stream << std::scientific << std::setprecision(10);
        int values_on_line = 0;
        for (int index_1 = 0; index_1 < dimensions[0]; index_1++) {
            for (int index_2 = 0; index_2 < dimensions[1]; index_2++) {
                for (int index_3 = 0; index_3 < dimensions[2]; index_3++) {
                    stream << std::setw(18) << field.value(index_1, index_2, index_3);
                    if (++values_on_line == values_per_line) {
                        stream << "\n";
                        values_on_line = 0;
                    }
                }
                /**
                 * A line break after each run along the fastest axis is what
                 * every cube writer does, and what makes a file readable by
                 * eye as a stack of z-columns.
                 */
                if (values_on_line != 0) {
                    stream << "\n";
                    values_on_line = 0;
                }
            }
        }

        stream.flush();
        if (!stream)
            throw std::runtime_error("Failed while writing cube file: " + path);
    }

    [[nodiscard]] std::string default_file_name(const std::string& stem) const override {
        return stem + ".cube";
    }

  private:
    static constexpr int values_per_line = 6;
    /// See the header: twelve decimals keep the cell volume exact to 1e-11.
    static constexpr int header_decimals = 12;
    static constexpr int header_width = 20;
};

} // namespace q3
