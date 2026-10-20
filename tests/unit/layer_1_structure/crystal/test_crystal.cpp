#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"

using namespace q3;

static Crystal make_si_fcc(Real cell_dimension_1 = 10.263048) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, cell_dimension_1);
    AtomSpecies silicon{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

// ── bravais_lattice_index=2 lattice vectors ─────────────────────────────────────────
TEST(Crystal, FCC_LatticeVectors) {
    Crystal crystal_structure = make_si_fcc();
    const Matrix3x3& at = crystal_structure.lattice_vectors();

    // bravais_lattice_index=2 (FCC): a1=(-1/2,0,1/2), a2=(0,1/2,1/2), a3=(-1/2,1/2,0)
    EXPECT_NEAR(at[0][0], -0.5, 1e-12);
    EXPECT_NEAR(at[0][1], 0.0, 1e-12);
    EXPECT_NEAR(at[0][2], 0.5, 1e-12);

    EXPECT_NEAR(at[1][0], 0.0, 1e-12);
    EXPECT_NEAR(at[1][1], 0.5, 1e-12);
    EXPECT_NEAR(at[1][2], 0.5, 1e-12);

    EXPECT_NEAR(at[2][0], -0.5, 1e-12);
    EXPECT_NEAR(at[2][1], 0.5, 1e-12);
    EXPECT_NEAR(at[2][2], 0.0, 1e-12);
}

// ── Cell volume ─────────────────────────────────────────────────────
TEST(Crystal, FCC_Volume) {
    Real cell_dimension_1 = 10.263048;
    Crystal crystal_structure = make_si_fcc(cell_dimension_1);

    // FCC volume = a³/4
    Real expected_omega = std::pow(cell_dimension_1, 3) / 4.0;
    EXPECT_NEAR(crystal_structure.cell_volume(), expected_omega, 1e-4);
}

// ── Reciprocal lattice orthogonality: a_i · b_j = δ_ij ─────────────
TEST(Crystal, ReciprocalLattice_Orthogonality) {
    Crystal crystal_structure = make_si_fcc();
    const Matrix3x3& at = crystal_structure.lattice_vectors();
    const Matrix3x3& bg = crystal_structure.reciprocal_lattice_vectors();

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Real dot = at[i][0] * bg[j][0] + at[i][1] * bg[j][1] + at[i][2] * bg[j][2];
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(dot, expected, 1e-12)
                << "a" << i << " · b" << j << " should be " << expected;
        }
    }
}

// ── tpiba = 2π/alat ────────────────────────────────────────────────
TEST(Crystal, Tpiba) {
    Real cell_dimension_1 = 10.263048;
    Crystal crystal_structure = make_si_fcc(cell_dimension_1);
    EXPECT_NEAR(
        crystal_structure.two_pi_over_lattice_parameter(), 2.0 * M_PI / cell_dimension_1, 1e-10);
    EXPECT_NEAR(crystal_structure.two_pi_over_lattice_parameter_squared(),
                crystal_structure.two_pi_over_lattice_parameter() *
                    crystal_structure.two_pi_over_lattice_parameter(),
                1e-10);
}

// ── Atom positions ──────────────────────────────────────────────────
TEST(Crystal, AtomPositions) {
    Crystal crystal_structure = make_si_fcc();

    EXPECT_EQ(crystal_structure.number_of_atoms(), 2);
    EXPECT_EQ(crystal_structure.number_of_species(), 1);

    // First atom at origin
    Vector3D t0 = crystal_structure.fractional_position(0);
    EXPECT_NEAR(t0.x, 0.0, 1e-12);
    EXPECT_NEAR(t0.y, 0.0, 1e-12);
    EXPECT_NEAR(t0.z, 0.0, 1e-12);

    // Second atom at (1/4, 1/4, 1/4)
    Vector3D t1 = crystal_structure.fractional_position(1);
    EXPECT_NEAR(t1.x, 0.25, 1e-12);
    EXPECT_NEAR(t1.y, 0.25, 1e-12);
    EXPECT_NEAR(t1.z, 0.25, 1e-12);
}

// ── Unsupported bravais_lattice_index throws ────────────────────────────────────────
TEST(Crystal, UnsupportedBravaisLatticeIndex_Throws) {
    Crystal crystal_structure;
    EXPECT_THROW(crystal_structure.set_from_lattice_type(99, 10.0), std::runtime_error);
}

// ── Bravais Lattices Validation ──────────────────────────────────────
TEST(Crystal, Ibrav1_SC) {
    Crystal c;
    c.set_from_lattice_type(1, 10.0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 1000.0, 1e-6);
    EXPECT_NEAR(c.lattice_vectors()[0].x, 1.0, 1e-10);
    EXPECT_NEAR(c.lattice_vectors()[1].y, 1.0, 1e-10);
    EXPECT_NEAR(c.lattice_vectors()[2].z, 1.0, 1e-10);
}

TEST(Crystal, Ibrav3_BCC) {
    Crystal c;
    c.set_from_lattice_type(3, 10.0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 500.0, 1e-6);
}

TEST(Crystal, Ibrav3Neg_BCC) {
    Crystal c;
    c.set_from_lattice_type(-3, 10.0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 500.0, 1e-6);
}

TEST(Crystal, Ibrav4_Hex) {
    Crystal c;
    c.set_from_lattice_type(4, 10.0, 0, 1.5, 0, 0, 0);
    c.compute_derived_quantities();
    Real expected = 1000.0 * std::sqrt(3.0) / 2.0 * 1.5;
    EXPECT_NEAR(c.cell_volume(), expected, 1e-4);
}

TEST(Crystal, Ibrav5_TrigR) {
    Crystal c;
    c.set_from_lattice_type(5, 10.0, 0, 0, 0.5, 0, 0);
    c.compute_derived_quantities();
    Real cosalpha = 0.5;
    Real vol_factor =
        std::sqrt(1.0 - 3.0 * cosalpha * cosalpha + 2.0 * cosalpha * cosalpha * cosalpha);
    EXPECT_NEAR(c.cell_volume(), 1000.0 * vol_factor, 1e-4);
}

TEST(Crystal, Ibrav5Neg_TrigR111) {
    Crystal c;
    c.set_from_lattice_type(-5, 10.0, 0, 0, 0.5, 0, 0);
    c.compute_derived_quantities();
    Crystal c2;
    c2.set_from_lattice_type(5, 10.0, 0, 0, 0.5, 0, 0);
    c2.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), c2.cell_volume(), 1e-4);
}

TEST(Crystal, Ibrav6_Tet) {
    Crystal c;
    c.set_from_lattice_type(6, 10.0, 0, 2.0, 0, 0, 0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 2000.0, 1e-6);
}

TEST(Crystal, Ibrav7_BCT) {
    Crystal c;
    c.set_from_lattice_type(7, 10.0, 0, 2.0, 0, 0, 0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 1000.0, 1e-4);
}

TEST(Crystal, Ibrav8_Ort) {
    Crystal c;
    c.set_from_lattice_type(8, 10.0, 1.5, 2.0, 0, 0, 0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 3000.0, 1e-6);
}

TEST(Crystal, Ibrav9_BaseCentOrt) {
    Crystal c;
    c.set_from_lattice_type(9, 10.0, 1.5, 2.0, 0, 0, 0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 1500.0, 1e-4);
}

TEST(Crystal, Ibrav10_FaceCentOrt) {
    Crystal c;
    c.set_from_lattice_type(10, 10.0, 1.5, 2.0, 0, 0, 0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 750.0, 1e-4);
}

TEST(Crystal, Ibrav11_BodyCentOrt) {
    Crystal c;
    c.set_from_lattice_type(11, 10.0, 1.5, 2.0, 0, 0, 0);
    c.compute_derived_quantities();
    EXPECT_NEAR(c.cell_volume(), 1500.0, 1e-4);
}

TEST(Crystal, Ibrav12_MonoP) {
    Crystal c;
    c.set_from_lattice_type(12, 10.0, 1.2, 1.5, 0.3, 0, 0);
    c.compute_derived_quantities();
    Real cosab = 0.3;
    Real sinab = std::sqrt(1.0 - cosab * cosab);
    EXPECT_NEAR(c.cell_volume(), 1000.0 * 1.2 * 1.5 * sinab, 1e-4);
}

// ── Pair separations in the home cell ───────────────────────────────

TEST(Crystal, HomeCellSeparationBringsCrystalCoordinatesToAtMostOneHalf) {
    /**
     * Silicon's second atom written 3, -2 and 1 cells away from (1/4, 1/4, 1/4). Its separation
     * from the first comes back as the in-cell pair's: (1/4, 1/4, 1/4) in crystal coordinates,
     * which is 1/4 (a1 + a2 + a3) = (-1/4, 1/4, 1/4) alat for ibrav = 2. The reverse
     * separation is its negative, and an atom's separation from itself is zero.
     */
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    crystal_structure.add_species({"Si", 28.08, 4.0, "Si.pz-vbc.UPF"});
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {3.25, -1.75, 1.25}});
    crystal_structure.compute_derived_quantities();

    const Vector3D expected{-0.25, 0.25, 0.25};
    const Vector3D forward = crystal_structure.home_cell_separation(1, 0);
    const Vector3D backward = crystal_structure.home_cell_separation(0, 1);
    const Vector3D self = crystal_structure.home_cell_separation(1, 1);
    for (int axis = 0; axis < 3; axis++) {
        EXPECT_NEAR(forward[axis], expected[axis], 1e-12) << "axis " << axis;
        EXPECT_NEAR(backward[axis], -expected[axis], 1e-12) << "axis " << axis;
        EXPECT_NEAR(self[axis], 0.0, 1e-12) << "axis " << axis;
    }
}

TEST(Crystal, Ibrav14_Triclinic) {
    Crystal c;
    c.set_from_lattice_type(14, 10.0, 1.1, 1.3, 0.1, 0.2, 0.15);
    c.compute_derived_quantities();
    EXPECT_GT(c.cell_volume(), 0.0);
}

namespace {
/// Two same-species atoms on the body-centred sites of a simple-cubic cell.
Crystal make_two_site_iron_cell() {
    Crystal crystal;
    crystal.set_from_lattice_type(1, 5.42);
    crystal.add_species({"Fe", 55.845, 8.0, "Fe.UPF"});
    crystal.add_atom({0, {0.0, 0.0, 0.0}});
    crystal.add_atom({0, {0.5, 0.5, 0.5}});
    crystal.compute_derived_quantities();
    return crystal;
}
} // namespace

TEST(CrystalSiteSymmetryClasses, DefaultIsTheSpeciesIndex) {
    Crystal crystal = make_two_site_iron_cell();
    EXPECT_FALSE(crystal.has_site_symmetry_classes());
    EXPECT_EQ(crystal.site_symmetry_class(0), 0);
    EXPECT_EQ(crystal.site_symmetry_class(1), 0);
}

TEST(CrystalSiteSymmetryClasses, AnInstalledPartitionSeparatesSameSpeciesSites) {
    Crystal crystal = make_two_site_iron_cell();
    crystal.set_site_symmetry_classes({0, 1});
    EXPECT_TRUE(crystal.has_site_symmetry_classes());
    EXPECT_EQ(crystal.site_symmetry_class(0), 0);
    EXPECT_EQ(crystal.site_symmetry_class(1), 1);
    EXPECT_EQ(crystal.species_type_index(1), 0) << "the species itself is untouched";

    crystal.set_site_symmetry_classes({});
    EXPECT_FALSE(crystal.has_site_symmetry_classes());
}

TEST(CrystalSiteSymmetryClasses, InvalidPartitionsAreRefused) {
    Crystal crystal = make_two_site_iron_cell();
    EXPECT_THROW(crystal.set_site_symmetry_classes({0}), std::runtime_error);
    EXPECT_THROW(crystal.set_site_symmetry_classes({0, -1}), std::runtime_error);

    Crystal mixed;
    mixed.set_from_lattice_type(1, 5.42);
    mixed.add_species({"Fe", 55.845, 8.0, "Fe.UPF"});
    mixed.add_species({"Co", 58.933, 9.0, "Co.UPF"});
    mixed.add_atom({0, {0.0, 0.0, 0.0}});
    mixed.add_atom({1, {0.5, 0.5, 0.5}});
    EXPECT_THROW(mixed.set_site_symmetry_classes({0, 0}), std::runtime_error)
        << "a partition may refine species, never merge them";
}

TEST(CrystalSiteSymmetryClasses, AddingAnAtomResetsThePartition) {
    Crystal crystal = make_two_site_iron_cell();
    crystal.set_site_symmetry_classes({0, 1});
    crystal.add_atom({0, {0.25, 0.25, 0.25}});
    EXPECT_FALSE(crystal.has_site_symmetry_classes());
    EXPECT_EQ(crystal.site_symmetry_class(2), 0);
}
