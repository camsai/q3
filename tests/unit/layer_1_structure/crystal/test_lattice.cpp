#include <gtest/gtest.h>

#include <cmath>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/lattice.hpp"

using namespace q3;

/**
 * Helper: compute cell volume from lattice vectors (units of alat^3).
 */
static Real compute_volume(const Matrix3x3& lattice_vectors) {
    return std::abs(lattice_vectors.determinant());
}

// ── bravais_lattice_index = 1 (Simple Cubic) ──────────────────────────────────────────
TEST(LatticeVectors, Ibrav1_SimpleCubic) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(1, 10.0);
    EXPECT_NEAR(lattice_vectors[0].x, 1.0, 1e-12);
    EXPECT_NEAR(lattice_vectors[1].y, 1.0, 1e-12);
    EXPECT_NEAR(lattice_vectors[2].z, 1.0, 1e-12);
    EXPECT_NEAR(compute_volume(lattice_vectors), 1.0, 1e-12);
}

// ── bravais_lattice_index = 2 (FCC) ──────────────────────────────────────────────────
TEST(LatticeVectors, Ibrav2_FCC) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(2, 10.0);
    EXPECT_NEAR(lattice_vectors[0].x, -0.5, 1e-12);
    EXPECT_NEAR(lattice_vectors[0].z, 0.5, 1e-12);
    EXPECT_NEAR(compute_volume(lattice_vectors), 0.25, 1e-12);
}

// ── bravais_lattice_index = 3 (BCC) ──────────────────────────────────────────────────
TEST(LatticeVectors, Ibrav3_BCC) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(3, 10.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), 0.5, 1e-12);
}

// ── bravais_lattice_index = -3 (BCC alt) ─────────────────────────────────────────────
TEST(LatticeVectors, Ibrav3Neg_BCC_Alt) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(-3, 10.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), 0.5, 1e-12);
    // Should give same volume as bravais_lattice_index=3
    auto lattice_vectors_3 = get_lattice_vectors_from_bravais_index(3, 10.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), compute_volume(lattice_vectors_3), 1e-12);
}

// ── bravais_lattice_index = 4 (Hexagonal) ────────────────────────────────────────────
TEST(LatticeVectors, Ibrav4_Hexagonal) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(4, 10.0, 0, 1.5);
    Real expected_volume = std::sqrt(3.0) / 2.0 * 1.5;
    EXPECT_NEAR(compute_volume(lattice_vectors), expected_volume, 1e-10);
}

// ── bravais_lattice_index = 5 (Trigonal R) ───────────────────────────────────────────
TEST(LatticeVectors, Ibrav5_TrigonalR) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(5, 10.0, 0, 0, 0.5);
    Real cos_alpha = 0.5;
    Real expected_volume =
        std::sqrt(1.0 - 3.0 * cos_alpha * cos_alpha + 2.0 * cos_alpha * cos_alpha * cos_alpha);
    EXPECT_NEAR(compute_volume(lattice_vectors), expected_volume, 1e-10);
}

// ── bravais_lattice_index = -5 (Trigonal R 111) ──────────────────────────────────────
TEST(LatticeVectors, Ibrav5Neg_TrigonalR111) {
    auto lattice_vectors_5 = get_lattice_vectors_from_bravais_index(5, 10.0, 0, 0, 0.5);
    auto lattice_vectors_neg5 = get_lattice_vectors_from_bravais_index(-5, 10.0, 0, 0, 0.5);
    EXPECT_NEAR(compute_volume(lattice_vectors_5), compute_volume(lattice_vectors_neg5), 1e-10);
}

// ── bravais_lattice_index = 6 (Tetragonal) ───────────────────────────────────────────
TEST(LatticeVectors, Ibrav6_Tetragonal) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(6, 10.0, 0, 2.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), 2.0, 1e-12);
}

// ── bravais_lattice_index = 7 (Body-centered tetragonal) ─────────────────────────────
TEST(LatticeVectors, Ibrav7_BodyCenteredTetragonal) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(7, 10.0, 0, 2.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), 1.0, 1e-10);
}

// ── bravais_lattice_index = 8 (Orthorhombic) ─────────────────────────────────────────
TEST(LatticeVectors, Ibrav8_Orthorhombic) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(8, 10.0, 1.5, 2.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), 3.0, 1e-12);
}

// ── bravais_lattice_index = 9 (Base-centered orthorhombic) ───────────────────────────
TEST(LatticeVectors, Ibrav9_BaseCenteredOrthorhombic) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(9, 10.0, 1.5, 2.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), 1.5, 1e-10);
}

// ── bravais_lattice_index = 10 (Face-centered orthorhombic) ──────────────────────────
TEST(LatticeVectors, Ibrav10_FaceCenteredOrthorhombic) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(10, 10.0, 1.5, 2.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), 0.75, 1e-10);
}

// ── bravais_lattice_index = 11 (Body-centered orthorhombic) ──────────────────────────
TEST(LatticeVectors, Ibrav11_BodyCenteredOrthorhombic) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(11, 10.0, 1.5, 2.0);
    EXPECT_NEAR(compute_volume(lattice_vectors), 1.5, 1e-10);
}

// ── bravais_lattice_index = 12 (Monoclinic P) ────────────────────────────────────────
TEST(LatticeVectors, Ibrav12_MonoclinicP) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(12, 10.0, 1.2, 1.5, 0.3);
    Real cos_ab = 0.3;
    Real sin_ab = std::sqrt(1.0 - cos_ab * cos_ab);
    Real expected_volume = 1.2 * 1.5 * sin_ab;
    EXPECT_NEAR(compute_volume(lattice_vectors), expected_volume, 1e-10);
}

// ── bravais_lattice_index = 14 (Triclinic) ───────────────────────────────────────────
TEST(LatticeVectors, Ibrav14_Triclinic) {
    auto lattice_vectors =
        get_lattice_vectors_from_bravais_index(14, 10.0, 1.1, 1.3, 0.1, 0.2, 0.15);
    EXPECT_GT(compute_volume(lattice_vectors), 0.0);
}

// ── bravais_lattice_index = 0 (Free lattice) ─────────────────────────────────────────
TEST(LatticeVectors, Ibrav0_FreeLattice) {
    auto lattice_vectors = get_lattice_vectors_from_bravais_index(0, 10.0);
    // Returns identity as placeholder
    EXPECT_NEAR(lattice_vectors[0].x, 1.0, 1e-12);
    EXPECT_NEAR(lattice_vectors[1].y, 1.0, 1e-12);
    EXPECT_NEAR(lattice_vectors[2].z, 1.0, 1e-12);
}

// ── Unsupported bravais_lattice_index throws ─────────────────────────────────────────
TEST(LatticeVectors, UnsupportedBravaisLatticeIndex_Throws) {
    EXPECT_THROW((void)get_lattice_vectors_from_bravais_index(99, 10.0), std::runtime_error);
    EXPECT_THROW((void)get_lattice_vectors_from_bravais_index(13, 10.0), std::runtime_error);
    EXPECT_THROW((void)get_lattice_vectors_from_bravais_index(-1, 10.0), std::runtime_error);
}
