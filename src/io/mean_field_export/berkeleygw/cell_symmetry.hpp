#pragma once

/**
 * BerkeleyGW's `cell_symmetry` header field: 1 for a lattice of the hexagonal
 * family, 0 for every other.
 *
 * The BerkeleyGW specification defines it as 0 for cubic and 1 for hexagonal
 * symmetry (Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012), Appendix
 * 11.1; see header.hpp). pw2bgw.x takes it from QE's `ibrav` input (in
 * write_wfng and write_rhog, QE 7.5): 4 (hexagonal) and 5
 * (trigonal R) give 1, and the cubic, tetragonal, orthorhombic, monoclinic
 * and triclinic indices give 0. Q3 records the same index when the input names
 * a Bravais lattice. Every other cell is a free lattice, index 0: a YAML
 * lattice outside the cubic family, YAML `lattice.vectors`, and QE's
 * `CELL_PARAMETERS`, which is how Materials Project cells arrive. The export
 * used to refuse those, because pw2bgw.x settles that case from an input Q3 has
 * no counterpart of, which shut out every non-cubic crystal.
 *
 * The index only names the shape of the lattice, so the shape is read off the
 * vectors instead:
 *
 *   - hexagonal (as index 4): two vectors of equal length at 120 or 60
 *     degrees, and the third perpendicular to both;
 *   - trigonal R (as index 5): three vectors of equal length at one common
 *     angle that is none of the three a cubic lattice's primitive cell has,
 *     90 (simple), 60 (face-centred) and 109.47 degrees (body-centred);
 *   - anything else is in the cubic family's bracket, 0.
 */

#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "layer_0_core/types/matrix_3x3.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

namespace berkeleygw_cell_symmetry_detail {

/**
 * Relative agreement two lengths, or two angle cosines, need to count as
 * equal. Loose enough for a cell written to the six or seven decimals
 * structure databases carry (a "120 degree" angle stored as 120.000006 is
 * 1e-7 away in cosine), and far tighter than any lattice that is genuinely
 * not hexagonal.
 */
constexpr Real metric_tolerance = 1e-5;

/// The dot product of rows `first` and `second` of `vectors`.
inline Real dot(const Matrix3x3& vectors, int first, int second) {
    return vectors[first].dot(vectors[second]);
}

/// Whether two dimensionless quantities -- length ratios or cosines -- agree to `metric_tolerance`.
inline bool close(Real first, Real second) {
    return std::abs(first - second) <= metric_tolerance;
}

} // namespace berkeleygw_cell_symmetry_detail

/// Whether the rows of `lattice_vectors` span a hexagonal or trigonal-R lattice.
inline bool lattice_is_in_the_hexagonal_family(const Matrix3x3& lattice_vectors) {
    using namespace berkeleygw_cell_symmetry_detail;
    const std::array<Real, 3> length = {std::sqrt(dot(lattice_vectors, 0, 0)),
                                        std::sqrt(dot(lattice_vectors, 1, 1)),
                                        std::sqrt(dot(lattice_vectors, 2, 2))};
    if (length[0] <= 0.0 || length[1] <= 0.0 || length[2] <= 0.0)
        return false;
    auto cosine = [&](int first, int second) {
        return dot(lattice_vectors, first, second) / (length[first] * length[second]);
    };
    auto same_length = [&](int first, int second) {
        return close(length[first] / length[second], 1.0);
    };

    for (int unique = 0; unique < 3; unique++) {
        const int first = (unique + 1) % 3;
        const int second = (unique + 2) % 3;
        if (same_length(first, second) && close(std::abs(cosine(first, second)), 0.5) &&
            close(cosine(unique, first), 0.0) && close(cosine(unique, second), 0.0))
            return true;
    }

    if (same_length(0, 1) && same_length(1, 2)) {
        const Real common = cosine(0, 1);
        if (close(cosine(1, 2), common) && close(cosine(0, 2), common)) {
            const bool cubic_primitive =
                close(common, 0.0) || close(common, 0.5) || close(common, -1.0 / 3.0);
            return !cubic_primitive;
        }
    }
    return false;
}

/**
 * The `cell_symmetry` value for a cell: from the Bravais index when the input
 * named one, from the lattice vectors when it is a free lattice (index 0).
 */
inline int berkeleygw_cell_symmetry(int bravais_lattice_index, const Matrix3x3& lattice_vectors) {
    const int absolute_index = std::abs(bravais_lattice_index);
    if (bravais_lattice_index == 0)
        return lattice_is_in_the_hexagonal_family(lattice_vectors) ? 1 : 0;
    if (absolute_index >= 1 && absolute_index <= 3)
        return 0; // simple, face- and body-centred cubic
    if (absolute_index >= 4 && absolute_index <= 5)
        return 1; // hexagonal, trigonal R
    if (absolute_index >= 6 && absolute_index <= 14)
        return 0; // tetragonal, orthorhombic, monoclinic, triclinic
    throw std::runtime_error("BerkeleyGW cell_symmetry: unrecognized bravais_lattice_index " +
                             std::to_string(bravais_lattice_index));
}

} // namespace q3
