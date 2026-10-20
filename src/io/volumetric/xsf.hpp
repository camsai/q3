#pragma once

/**
 * XsfVolumetricFieldWriter — XCrySDen's XSF format, structure plus one 3D
 * datagrid.
 *
 * Written from the published XSF specification (the XCrySDen documentation's
 * "XSF (XCrySDen Structure File)" chapter); no reference implementation was
 * consulted. The file is a `CRYSTAL` block — `PRIMVEC` and `PRIMCOORD` — and
 * then a `BEGIN_BLOCK_DATAGRID_3D` carrying one `BEGIN_DATAGRID_3D_*` grid.
 *
 * **Conventions this writer chose, and the reasons.**
 *
 *   - *Ångström.* XSF's lengths are Ångström by definition, unlike the cube
 *     format's bohr, so the cell and the atoms are converted here and only
 *     here. The datagrid *values* are not converted — they stay in the field's
 *     own unit, which the datagrid's name states, because XCrySDen only ever
 *     draws isosurfaces of them and a silently rescaled density is worse than
 *     a labelled one.
 *   - *The first axis varies fastest*, which is the opposite of the cube
 *     format and is what the specification requires: the grid is written as
 *     Fortran would write `(((v(i,j,k), i=1,nx), j=1,ny), k=1,nz)`.
 *     `tests/unit/io/volumetric/test_xsf.cpp` pins it against a field whose
 *     value identifies its own grid point.
 *   - *The boundary plane is duplicated.* An XSF "general" datagrid spans the
 *     three vectors it names including their far end, so a periodic n-point
 *     grid is written as n+1 points per axis with the last a copy of the
 *     first. Writing n points instead leaves a one-voxel gap at the cell
 *     boundary that XCrySDen renders as a hole through every isosurface.
 */

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "io/volumetric/field.hpp"
#include "io/volumetric/writer.hpp"
#include "layer_0_core/types/constants.hpp"

namespace q3 {

class XsfVolumetricFieldWriter : public VolumetricFieldWriter {
  public:
    void write(const std::string& path, const VolumetricField& field) const override {
        std::ofstream stream(path);
        if (!stream)
            throw std::runtime_error("Cannot write xsf file: " + path);

        const Real bohr_to_angstrom = constants::BOHR_RADIUS_ANGSTROM;
        stream << "# Q3 " << field.description << " (" << field.unit << ")\n";
        stream << "CRYSTAL\n";
        stream << "PRIMVEC\n";
        stream << std::fixed << std::setprecision(10);
        for (int axis = 0; axis < 3; axis++) {
            for (int component = 0; component < 3; component++)
                stream << std::setw(20)
                       << field.lattice_vectors_bohr[axis][component] * bohr_to_angstrom;
            stream << "\n";
        }
        stream << "PRIMCOORD\n";
        stream << field.number_of_atoms() << " 1\n";
        for (int atom_index = 0; atom_index < field.number_of_atoms(); atom_index++) {
            const Vector3D position = field.cartesian_position_bohr(atom_index);
            stream << std::setw(4) << field.atomic_number(atom_index);
            for (int component = 0; component < 3; component++)
                stream << std::setw(20) << position[component] * bohr_to_angstrom;
            stream << "\n";
        }

        const auto& dimensions = field.grid_dimensions;
        stream << "BEGIN_BLOCK_DATAGRID_3D\n";
        stream << " Q3_" << datagrid_name(field) << "\n";
        stream << " BEGIN_DATAGRID_3D_Q3_" << datagrid_name(field) << "\n";
        stream << "  " << dimensions[0] + 1 << " " << dimensions[1] + 1 << " " << dimensions[2] + 1
               << "\n";
        stream << "  " << std::setw(20) << 0.0 << std::setw(20) << 0.0 << std::setw(20) << 0.0
               << "\n";
        for (int axis = 0; axis < 3; axis++) {
            stream << "  ";
            for (int component = 0; component < 3; component++)
                stream << std::setw(20)
                       << field.lattice_vectors_bohr[axis][component] * bohr_to_angstrom;
            stream << "\n";
        }

        stream << std::scientific << std::setprecision(10);
        int values_on_line = 0;
        for (int index_3 = 0; index_3 <= dimensions[2]; index_3++) {
            for (int index_2 = 0; index_2 <= dimensions[1]; index_2++) {
                for (int index_1 = 0; index_1 <= dimensions[0]; index_1++) {
                    stream << std::setw(18)
                           << field.value(index_1 % dimensions[0],
                                          index_2 % dimensions[1],
                                          index_3 % dimensions[2]);
                    if (++values_on_line == values_per_line) {
                        stream << "\n";
                        values_on_line = 0;
                    }
                }
            }
        }
        if (values_on_line != 0)
            stream << "\n";

        stream << " END_DATAGRID_3D\n";
        stream << "END_BLOCK_DATAGRID_3D\n";

        stream.flush();
        if (!stream)
            throw std::runtime_error("Failed while writing xsf file: " + path);
    }

    [[nodiscard]] std::string default_file_name(const std::string& stem) const override {
        return stem + ".xsf";
    }

  private:
    static constexpr int values_per_line = 6;

    /**
     * A datagrid identifier XCrySDen shows in its field chooser: no spaces, and
     * carrying the unit, since the format has nowhere else to put it.
     */
    [[nodiscard]] static std::string datagrid_name(const VolumetricField& field) {
        std::string name = field.description + "_" + field.unit;
        for (char& character : name) {
            const bool is_plain = (character >= 'a' && character <= 'z') ||
                                  (character >= 'A' && character <= 'Z') ||
                                  (character >= '0' && character <= '9');
            if (!is_plain)
                character = '_';
        }
        return name;
    }
};

} // namespace q3
