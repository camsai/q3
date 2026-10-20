#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <vector>

#include "layer_1_structure/reciprocal_space/wigner_seitz_supercell.hpp"

using namespace q3;

namespace {

/// Simple-cubic primitive vectors, lattice parameter `a`.
Matrix3x3 simple_cubic_lattice(Real a) {
    Matrix3x3 lattice;
    lattice[0] = Vector3D(a, 0.0, 0.0);
    lattice[1] = Vector3D(0.0, a, 0.0);
    lattice[2] = Vector3D(0.0, 0.0, a);
    return lattice;
}

/// FCC primitive vectors, conventional lattice parameter `a`.
Matrix3x3 fcc_primitive_lattice(Real a) {
    Matrix3x3 lattice;
    lattice[0] = Vector3D(0.0, 0.5 * a, 0.5 * a);
    lattice[1] = Vector3D(0.5 * a, 0.0, 0.5 * a);
    lattice[2] = Vector3D(0.5 * a, 0.5 * a, 0.0);
    return lattice;
}

/// Hexagonal primitive vectors with in-plane `a` and axis `c`.
Matrix3x3 hexagonal_lattice(Real a, Real c) {
    Matrix3x3 lattice;
    lattice[0] = Vector3D(a, 0.0, 0.0);
    lattice[1] = Vector3D(-0.5 * a, 0.5 * std::sqrt(3.0) * a, 0.0);
    lattice[2] = Vector3D(0.0, 0.0, c);
    return lattice;
}

/// A mildly oblique triclinic cell with no symmetry to hide behind.
Matrix3x3 triclinic_lattice() {
    Matrix3x3 lattice;
    lattice[0] = Vector3D(1.0, 0.0, 0.0);
    lattice[1] = Vector3D(0.3, 1.1, 0.0);
    lattice[2] = Vector3D(0.2, 0.35, 0.9);
    return lattice;
}

/// A retained vector as (translation, degeneracy), comparable across constructions.
struct RetainedVector {
    int x, y, z, degeneracy;
    bool operator<(const RetainedVector& other) const {
        return std::tie(x, y, z, degeneracy) <
               std::tie(other.x, other.y, other.z, other.degeneracy);
    }
    bool operator==(const RetainedVector& other) const {
        return x == other.x && y == other.y && z == other.z && degeneracy == other.degeneracy;
    }
};

std::vector<RetainedVector> retained_set(const WignerSeitzSupercell& supercell) {
    std::vector<RetainedVector> result;
    for (const WignerSeitzLatticeVector& entry : supercell.lattice_vectors()) {
        result.push_back({entry.lattice_translation.x,
                          entry.lattice_translation.y,
                          entry.lattice_translation.z,
                          entry.degeneracy});
    }
    std::sort(result.begin(), result.end());
    return result;
}

/**
 * The same Wigner-Seitz criterion evaluated with deliberately oversized
 * boxes -- candidates out to three supercells, images out to five -- so it
 * does not rely on any bound the class derives. Slow, but on these meshes
 * it is the reference the derived bounds must reproduce exactly.
 */
std::vector<RetainedVector>
brute_force_wigner_seitz(const Matrix3x3& lattice, int mesh_x, int mesh_y, int mesh_z) {
    const std::array<int, 3> mesh = {mesh_x, mesh_y, mesh_z};
    std::vector<Vector3D> translations;
    for (int image_x = -5; image_x <= 5; image_x++) {
        for (int image_y = -5; image_y <= 5; image_y++) {
            for (int image_z = -5; image_z <= 5; image_z++) {
                translations.push_back(lattice[0] * static_cast<Real>(image_x * mesh[0]) +
                                       lattice[1] * static_cast<Real>(image_y * mesh[1]) +
                                       lattice[2] * static_cast<Real>(image_z * mesh[2]));
            }
        }
    }
    Real shortest = 0.0;
    for (int axis = 0; axis < 3; axis++) {
        Real length = (lattice[axis] * static_cast<Real>(mesh[axis])).norm();
        shortest = (shortest == 0.0) ? length : std::min(shortest, length);
    }
    const Real tolerance = 1e-6 * shortest;

    std::vector<RetainedVector> result;
    for (int x = -3 * mesh[0] - 3; x <= 3 * mesh[0] + 3; x++) {
        for (int y = -3 * mesh[1] - 3; y <= 3 * mesh[1] + 3; y++) {
            for (int z = -3 * mesh[2] - 3; z <= 3 * mesh[2] + 3; z++) {
                Vector3D candidate = lattice[0] * static_cast<Real>(x) +
                                     lattice[1] * static_cast<Real>(y) +
                                     lattice[2] * static_cast<Real>(z);
                Real minimum = candidate.norm();
                for (const Vector3D& translation : translations) {
                    minimum = std::min(minimum, (candidate - translation).norm());
                }
                if (candidate.norm() > minimum + tolerance) {
                    continue;
                }
                int degeneracy = 0;
                for (const Vector3D& translation : translations) {
                    if (std::abs((candidate - translation).norm() - minimum) <= tolerance) {
                        degeneracy++;
                    }
                }
                result.push_back({x, y, z, degeneracy});
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace

TEST(WignerSeitzSupercell, TotalWeightEqualsTheKPointCount) {
    /**
     * The cheapest correctness check on the degeneracy bookkeeping: each of
     * the N_k cells of the supercell must be covered exactly once, with
     * boundary points sharing their coverage among the images they tie with.
     * A missed boundary point or a wrong degeneracy breaks this immediately.
     */
    for (int mesh : {2, 3, 4}) {
        WignerSeitzSupercell supercell;
        supercell.compute(simple_cubic_lattice(4.0), mesh, mesh, mesh);
        EXPECT_NEAR(supercell.total_weight(), static_cast<Real>(mesh * mesh * mesh), 1e-10)
            << "mesh " << mesh;
    }
}

TEST(WignerSeitzSupercell, IncludesTheOriginWithUnitDegeneracy) {
    WignerSeitzSupercell supercell;
    supercell.compute(simple_cubic_lattice(4.0), 3, 3, 3);
    bool found_origin = false;
    for (const WignerSeitzLatticeVector& entry : supercell.lattice_vectors()) {
        if (entry.lattice_translation.x == 0 && entry.lattice_translation.y == 0 &&
            entry.lattice_translation.z == 0) {
            found_origin = true;
            EXPECT_EQ(entry.degeneracy, 1);
        }
    }
    EXPECT_TRUE(found_origin);
}

TEST(WignerSeitzSupercell, IsSymmetricUnderInversion) {
    /**
     * R and -R are equidistant from the origin, so both must be retained
     * with the same degeneracy; an asymmetric set would make H(R) non-Hermitian.
     */
    WignerSeitzSupercell supercell;
    supercell.compute(simple_cubic_lattice(4.0), 3, 3, 3);
    const auto& entries = supercell.lattice_vectors();
    for (const WignerSeitzLatticeVector& entry : entries) {
        bool found_inverse = false;
        for (const WignerSeitzLatticeVector& candidate : entries) {
            if (candidate.lattice_translation.x == -entry.lattice_translation.x &&
                candidate.lattice_translation.y == -entry.lattice_translation.y &&
                candidate.lattice_translation.z == -entry.lattice_translation.z) {
                found_inverse = true;
                EXPECT_EQ(candidate.degeneracy, entry.degeneracy);
            }
        }
        EXPECT_TRUE(found_inverse);
    }
}

TEST(WignerSeitzSupercell, RejectsANonPositiveGridDimension) {
    WignerSeitzSupercell supercell;
    EXPECT_THROW(supercell.compute(simple_cubic_lattice(4.0), 2, 0, 2), std::invalid_argument);
}

TEST(WignerSeitzSupercell, RejectsLinearlyDependentLatticeVectors) {
    Matrix3x3 degenerate;
    degenerate[0] = Vector3D(1.0, 0.0, 0.0);
    degenerate[1] = Vector3D(2.0, 0.0, 0.0);
    degenerate[2] = Vector3D(0.0, 0.0, 1.0);
    WignerSeitzSupercell supercell;
    EXPECT_THROW(supercell.compute(degenerate, 2, 2, 2), std::invalid_argument);
}

TEST(WignerSeitzSupercell, TotalWeightEqualsTheKPointCountOnAnisotropicMeshes) {
    /**
     * Anisotropic meshes on non-orthogonal cells are where a fixed search
     * box fails: the superlattice is far from reduced, and the nearest image
     * of a candidate can lie outside a +-1 box. Every one of these came out
     * wrong (32.33 for FCC 4x4x2, 128.13 for 8x4x4) before the bounds were
     * derived from the lattice.
     */
    struct Case {
        const char* label;
        Matrix3x3 lattice;
        int mesh_x, mesh_y, mesh_z;
    };
    const std::vector<Case> cases = {
        {"fcc 4x4x2", fcc_primitive_lattice(10.2), 4, 4, 2},
        {"fcc 8x4x4", fcc_primitive_lattice(10.2), 8, 4, 4},
        {"fcc 8x8x1", fcc_primitive_lattice(10.2), 8, 8, 1},
        {"fcc 6x6x2", fcc_primitive_lattice(10.2), 6, 6, 2},
        {"hexagonal 12x6x4", hexagonal_lattice(4.6, 12.6), 12, 6, 4},
        {"hexagonal 9x9x3", hexagonal_lattice(4.6, 12.6), 9, 9, 3},
        {"triclinic 3x5x7", triclinic_lattice(), 3, 5, 7},
        {"triclinic 6x1x2", triclinic_lattice(), 6, 1, 2},
    };
    for (const Case& test_case : cases) {
        WignerSeitzSupercell supercell;
        ASSERT_NO_THROW(supercell.compute(
            test_case.lattice, test_case.mesh_x, test_case.mesh_y, test_case.mesh_z))
            << test_case.label;
        EXPECT_NEAR(supercell.total_weight(),
                    static_cast<Real>(test_case.mesh_x * test_case.mesh_y * test_case.mesh_z),
                    1e-10)
            << test_case.label;
    }
}

TEST(WignerSeitzSupercell, MatchesABruteForceSearchOnObliqueCells) {
    /**
     * The weight identity is necessary but not sufficient: an 8x8x1 FCC mesh
     * once kept the weight exactly right while missing six boundary vectors,
     * because dropped vectors can cancel in the sum. Comparing the retained
     * set itself against an oversized brute-force search is the check that
     * the derived bounds actually cover the Wigner-Seitz cell.
     */
    struct Case {
        const char* label;
        Matrix3x3 lattice;
        int mesh_x, mesh_y, mesh_z;
    };
    const std::vector<Case> cases = {
        {"fcc 4x4x2", fcc_primitive_lattice(1.0), 4, 4, 2},
        {"fcc 8x8x1", fcc_primitive_lattice(1.0), 8, 8, 1},
        {"hexagonal 6x3x2", hexagonal_lattice(1.0, 2.7), 6, 3, 2},
        {"triclinic 3x5x7", triclinic_lattice(), 3, 5, 7},
        {"triclinic 4x1x1", triclinic_lattice(), 4, 1, 1},
    };
    for (const Case& test_case : cases) {
        WignerSeitzSupercell supercell;
        supercell.compute(test_case.lattice, test_case.mesh_x, test_case.mesh_y, test_case.mesh_z);
        std::vector<RetainedVector> derived = retained_set(supercell);
        std::vector<RetainedVector> reference = brute_force_wigner_seitz(
            test_case.lattice, test_case.mesh_x, test_case.mesh_y, test_case.mesh_z);
        EXPECT_EQ(derived, reference) << test_case.label << ": derived " << derived.size()
                                      << " vectors, brute force " << reference.size();
    }
}

TEST(WignerSeitzSupercell, ReproducesTheValidatedIsotropicSets) {
    /**
     * The silicon example's FCC 4x4x4 and 8x8x8 sets were validated against
     * Quantum ESPRESSO end to end; the vector counts pin that the derived
     * bounds change nothing where the old fixed boxes were already right.
     */
    WignerSeitzSupercell coarse;
    coarse.compute(fcc_primitive_lattice(10.3344), 4, 4, 4);
    EXPECT_EQ(coarse.number_of_lattice_vectors(), 93);
    EXPECT_NEAR(coarse.total_weight(), 64.0, 1e-10);

    WignerSeitzSupercell fine;
    fine.compute(fcc_primitive_lattice(10.3344), 8, 8, 8);
    EXPECT_EQ(fine.number_of_lattice_vectors(), 617);
    EXPECT_NEAR(fine.total_weight(), 512.0, 1e-10);
}
