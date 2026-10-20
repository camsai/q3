#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "io/mean_field_export/berkeleygw/cell_symmetry.hpp"

using namespace q3;

namespace {

const Matrix3x3 unused_vectors{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

/// Lattice vectors from cell lengths and angles in degrees, a along x and b in the xy plane.
Matrix3x3 cell(Real a, Real b, Real c, Real alpha, Real beta, Real gamma) {
    const Real degree = M_PI / 180.0;
    const Real cos_alpha = std::cos(alpha * degree);
    const Real cos_beta = std::cos(beta * degree);
    const Real cos_gamma = std::cos(gamma * degree);
    const Real sin_gamma = std::sin(gamma * degree);
    const Real c_x = c * cos_beta;
    const Real c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    const Real c_z = std::sqrt(c * c - c_x * c_x - c_y * c_y);
    return {{a, 0.0, 0.0}, {b * cos_gamma, b * sin_gamma, 0.0}, {c_x, c_y, c_z}};
}

int free_lattice(const Matrix3x3& vectors) {
    return berkeleygw_cell_symmetry(0, vectors);
}

} // namespace

/// pw2bgw's bracketing of ibrav: 4 and 5 are the hexagonal family, the rest cubic.
TEST(BerkeleyGWCellSymmetry, NamedBravaisIndicesFollowPw2bgw) {
    for (int index : {1, 2, 3, -3, 6, 7, 8, 9, -9, 10, 11, 12, -12, 13, -13, 14})
        EXPECT_EQ(berkeleygw_cell_symmetry(index, unused_vectors), 0) << "ibrav " << index;
    for (int index : {4, 5, -5})
        EXPECT_EQ(berkeleygw_cell_symmetry(index, unused_vectors), 1) << "ibrav " << index;
    EXPECT_THROW(berkeleygw_cell_symmetry(99, unused_vectors), std::runtime_error);
}

/**
 * The wurtzite ZnO cell of camsai/q3#21 (mp-2133), a YAML HEX lattice that
 * arrives as free vectors, plus the same lattice in the 60-degree setting and
 * with the unique axis listed first.
 */
TEST(BerkeleyGWCellSymmetry, FreeHexagonalLatticesInAnySettingAreHexagonal) {
    EXPECT_EQ(free_lattice(cell(3.270985246, 3.270985246, 5.287937131, 90, 90, 120)), 1);
    EXPECT_EQ(free_lattice(cell(3.27, 3.27, 5.29, 90, 90, 60)), 1);
    EXPECT_EQ(free_lattice(cell(5.29, 3.27, 3.27, 120, 90, 90)), 1);
}

/// A structure database's 120.000006 degrees is still hexagonal.
TEST(BerkeleyGWCellSymmetry, DatabaseRoundingOfTheAngleIsTolerated) {
    EXPECT_EQ(free_lattice(cell(3.11, 3.11, 4.98, 90, 90, 120.000006)), 1);
    EXPECT_EQ(free_lattice(cell(3.11, 3.11, 4.98, 90, 90, 120.5)), 0);
}

/// Trigonal R is index 5's family; the cubic primitive cells are not.
TEST(BerkeleyGWCellSymmetry, RhombohedralIsHexagonalButCubicPrimitiveCellsAreNot) {
    EXPECT_EQ(free_lattice(cell(5.0, 5.0, 5.0, 80, 80, 80)), 1);
    EXPECT_EQ(free_lattice(cell(5.0, 5.0, 5.0, 90, 90, 90)), 0);
    EXPECT_EQ(free_lattice(cell(5.0, 5.0, 5.0, 60, 60, 60)), 0);
    const Real body_centred = std::acos(-1.0 / 3.0) * 180.0 / M_PI;
    EXPECT_EQ(free_lattice(cell(5.0, 5.0, 5.0, body_centred, body_centred, body_centred)), 0);
}

TEST(BerkeleyGWCellSymmetry, OtherFreeLatticesAreTheCubicFamily) {
    EXPECT_EQ(free_lattice(cell(4.0, 4.0, 6.0, 90, 90, 90)), 0);  // tetragonal
    EXPECT_EQ(free_lattice(cell(4.0, 5.0, 6.0, 90, 90, 90)), 0);  // orthorhombic
    EXPECT_EQ(free_lattice(cell(4.0, 5.0, 6.0, 90, 100, 90)), 0); // monoclinic
    EXPECT_EQ(free_lattice(cell(4.0, 5.0, 6.0, 80, 95, 110)), 0); // triclinic
    EXPECT_EQ(free_lattice(cell(4.0, 4.0, 6.0, 80, 90, 120)), 0); // c not perpendicular
}
