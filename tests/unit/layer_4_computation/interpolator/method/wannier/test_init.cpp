#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/local_orbital.hpp"

using namespace q3;

namespace {

Matrix3x3 simple_cubic_lattice(Real a) {
    Matrix3x3 lattice;
    lattice[0] = Vector3D(a, 0.0, 0.0);
    lattice[1] = Vector3D(0.0, a, 0.0);
    lattice[2] = Vector3D(0.0, 0.0, a);
    return lattice;
}

Real analytic_cubic_band(const Vector3D& kpoint, Real onsite, Real hopping) {
    return onsite +
           2.0 * hopping *
               (std::cos(constants::two_pi * kpoint.x) + std::cos(constants::two_pi * kpoint.y) +
                std::cos(constants::two_pi * kpoint.z));
}

/**
 * A two-band record on a coarse cubic mesh. `projection` is the (square) A
 * block used at every k-point; any full-rank choice gives a legitimate
 * unitary gauge, and the round trip at the mesh points must not care which.
 */
WannierInputs two_band_record(int mesh, const std::vector<Complex>& projection) {
    WannierInputs record;
    record.kpoint_grid_dimensions = {mesh, mesh, mesh};
    record.number_of_bands = 2;
    record.number_of_trial_orbitals = 2;
    for (int x = 0; x < mesh; x++) {
        for (int y = 0; y < mesh; y++) {
            for (int z = 0; z < mesh; z++) {
                Vector3D kpoint(static_cast<Real>(x) / mesh,
                                static_cast<Real>(y) / mesh,
                                static_cast<Real>(z) / mesh);
                record.kpoint_coordinates.push_back(kpoint);
                record.eigenvalues_rydberg.push_back({analytic_cubic_band(kpoint, -1.0, 0.3),
                                                      analytic_cubic_band(kpoint, 2.0, -0.15)});
                record.projection_matrices.push_back(projection);
            }
        }
    }
    return record;
}

const std::vector<Complex> identity_projection = {
    Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};
const std::vector<Complex> skewed_projection = {
    Complex(1.0, 0.0), Complex(0.2, -0.1), Complex(0.0, 0.3), Complex(0.8, 0.0)};

} // namespace

/**
 * Localisation on a mesh with a two-point axis is refused, not attempted. On
 * such an axis k+b and k-b are the same point, so the finite-difference Berry
 * phase the centres are built from has no meaning, and a minimiser run there
 * reports a spread bounded by the mesh rather than by the physics -- a number
 * that looks like an answer.
 */
TEST(BuildWannierHamiltonian, RefusesLocalizationOnAMeshWithATwoPointAxis) {
    WannierInputs record = two_band_record(2, identity_projection);
    KPointNeighborShells shells;
    Matrix3x3 reciprocal;
    reciprocal[0] = Vector3D(1.0, 0.0, 0.0);
    reciprocal[1] = Vector3D(0.0, 1.0, 0.0);
    reciprocal[2] = Vector3D(0.0, 0.0, 1.0);
    shells.compute(reciprocal, 2, 2, 2);
    record.neighbor_directions = shells.neighbor_directions();
    for (std::size_t kpoint = 0; kpoint < record.kpoint_coordinates.size(); kpoint++) {
        record.overlap_matrices.push_back(std::vector<std::vector<Complex>>(
            record.neighbor_directions.size(), identity_projection));
    }

    WannierConstructionSettings settings;
    settings.localization = LocalizationPolicy::Required;
    EXPECT_THROW(WannierHamiltonianBuilder(settings).build(record, simple_cubic_lattice(10.2)),
                 WannierDiagnosticError);
}

TEST(BuildWannierHamiltonian, ReproducesTheRecordEigenvaluesAtTheMeshPoints) {
    const int mesh = 3;
    WannierInputs record = two_band_record(mesh, identity_projection);
    LocalOrbitalQuantumStateInterpolator interpolator(
        WannierHamiltonianBuilder().build(record, simple_cubic_lattice(10.2)).hamiltonian);
    EXPECT_EQ(interpolator.number_of_bands(), 2);

    for (std::size_t index = 0; index < record.kpoint_coordinates.size(); index++) {
        std::vector<Real> interpolated =
            interpolator.eigenvalues_at(record.kpoint_coordinates[index]);
        std::vector<Real> expected = record.eigenvalues_rydberg[index];
        std::sort(expected.begin(), expected.end());
        ASSERT_EQ(interpolated.size(), 2u);
        EXPECT_NEAR(interpolated[0], expected[0], 1e-10) << "k-point " << index;
        EXPECT_NEAR(interpolated[1], expected[1], 1e-10) << "k-point " << index;
    }
}

TEST(BuildWannierHamiltonian, ANonTrivialGaugeLeavesTheMeshPointsExact) {
    /**
     * The projection gauge from a skewed A is a unitary that mixes the two
     * bands; the k -> R -> k round trip is an identity for ANY unitary, so
     * the mesh-point eigenvalues must still come back exactly. Off the mesh
     * the two gauges interpolate differently, which is the physics and not
     * tested here.
     */
    const int mesh = 3;
    WannierInputs record = two_band_record(mesh, skewed_projection);
    LocalOrbitalQuantumStateInterpolator interpolator(
        WannierHamiltonianBuilder().build(record, simple_cubic_lattice(10.2)).hamiltonian);
    for (std::size_t index = 0; index < record.kpoint_coordinates.size(); index++) {
        std::vector<Real> interpolated =
            interpolator.eigenvalues_at(record.kpoint_coordinates[index]);
        std::vector<Real> expected = record.eigenvalues_rydberg[index];
        std::sort(expected.begin(), expected.end());
        EXPECT_NEAR(interpolated[0], expected[0], 1e-10) << "k-point " << index;
        EXPECT_NEAR(interpolated[1], expected[1], 1e-10) << "k-point " << index;
    }
}

TEST(BuildWannierHamiltonian, RefusesMoreBandsThanWannierFunctions) {
    /// An entangled window needs the disentangler, which does not exist; refuse, do not guess.
    WannierInputs record = two_band_record(2, identity_projection);
    record.number_of_trial_orbitals = 1;
    EXPECT_THROW(WannierHamiltonianBuilder().build(record, simple_cubic_lattice(10.2)).hamiltonian,
                 std::invalid_argument);
}

TEST(BuildWannierHamiltonian, RefusesARecordWithoutItsGrid) {
    WannierInputs record = two_band_record(2, identity_projection);
    record.kpoint_grid_dimensions = {0, 0, 0};
    EXPECT_THROW(WannierHamiltonianBuilder().build(record, simple_cubic_lattice(10.2)).hamiltonian,
                 std::invalid_argument);
}

TEST(BuildWannierHamiltonian, RefusesAnEmptyRecord) {
    WannierInputs record;
    EXPECT_THROW(WannierHamiltonianBuilder().build(record, simple_cubic_lattice(10.2)).hamiltonian,
                 std::invalid_argument);
}

namespace {

Matrix3x3 fcc_primitive_lattice(Real a) {
    Matrix3x3 lattice;
    lattice[0] = Vector3D(0.0, 0.5 * a, 0.5 * a);
    lattice[1] = Vector3D(0.5 * a, 0.0, 0.5 * a);
    lattice[2] = Vector3D(0.5 * a, 0.5 * a, 0.0);
    return lattice;
}

Matrix3x3 triclinic_lattice() {
    Matrix3x3 lattice;
    lattice[0] = Vector3D(1.0, 0.0, 0.0);
    lattice[1] = Vector3D(0.3, 1.1, 0.0);
    lattice[2] = Vector3D(0.2, 0.35, 0.9);
    return lattice;
}

/// Two bands on an arbitrary Gamma-centred mesh, identity gauge.
WannierInputs two_band_inputs_on(int mesh_x, int mesh_y, int mesh_z) {
    WannierInputs inputs;
    inputs.kpoint_grid_dimensions = {mesh_x, mesh_y, mesh_z};
    inputs.number_of_bands = 2;
    inputs.number_of_trial_orbitals = 2;
    for (int x = 0; x < mesh_x; x++) {
        for (int y = 0; y < mesh_y; y++) {
            for (int z = 0; z < mesh_z; z++) {
                const Vector3D kpoint(static_cast<Real>(x) / mesh_x,
                                      static_cast<Real>(y) / mesh_y,
                                      static_cast<Real>(z) / mesh_z);
                inputs.kpoint_coordinates.push_back(kpoint);
                inputs.eigenvalues_rydberg.push_back({analytic_cubic_band(kpoint, -1.0, 0.3),
                                                      analytic_cubic_band(kpoint, 2.0, -0.15)});
                inputs.projection_matrices.push_back(identity_projection);
            }
        }
    }
    return inputs;
}

} // namespace

TEST(BuildWannierHamiltonian, RoundTripIsExactOnAnisotropicMeshesOverObliqueCells) {
    /**
     * The identity at the mesh points holds for any correct Wigner-Seitz set,
     * so this is the physics-level detector for a wrong one: with the old
     * fixed search boxes an FCC 4x4x2 mesh reproduced its own input to only
     * 6.5e-2 of the band scale, and 8x8x1 to 0.48. An isotropic mesh over a
     * cubic cell cannot see it.
     */
    struct Case {
        const char* label;
        Matrix3x3 lattice;
        int mesh_x, mesh_y, mesh_z;
    };
    const std::vector<Case> cases = {
        {"fcc 4x4x2", fcc_primitive_lattice(1.0), 4, 4, 2},
        {"fcc 8x8x1", fcc_primitive_lattice(1.0), 8, 8, 1},
        {"triclinic 3x5x2", triclinic_lattice(), 3, 5, 2},
    };
    for (const Case& test_case : cases) {
        WannierInputs inputs =
            two_band_inputs_on(test_case.mesh_x, test_case.mesh_y, test_case.mesh_z);
        LocalOrbitalQuantumStateInterpolator interpolator(
            WannierHamiltonianBuilder().build(inputs, test_case.lattice).hamiltonian);

        for (std::size_t index = 0; index < inputs.kpoint_coordinates.size(); index++) {
            std::vector<Real> interpolated =
                interpolator.eigenvalues_at(inputs.kpoint_coordinates[index]);
            std::vector<Real> expected = inputs.eigenvalues_rydberg[index];
            std::sort(expected.begin(), expected.end());
            for (int band = 0; band < 2; band++) {
                EXPECT_NEAR(interpolated[band], expected[band], 1e-10)
                    << test_case.label << " k-point " << index << " band " << band;
            }
        }
    }
}

TEST(BuildWannierHamiltonian, HamiltonianIsHermitianInRealSpace) {
    /// H(-R) = H(R)^dagger; a violation would give complex "eigenvalues".
    const int mesh = 3;
    WannierInputs inputs = two_band_inputs_on(mesh, mesh, mesh);
    for (std::vector<Complex>& projection : inputs.projection_matrices) {
        projection = {Complex(0.6, 0.8), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.8, -0.6)};
    }
    const LocalOrbitalHamiltonian hamiltonian =
        WannierHamiltonianBuilder().build(inputs, simple_cubic_lattice(4.0)).hamiltonian;

    const auto& lattice_vectors = hamiltonian.lattice_vectors();
    const auto& blocks = hamiltonian.hamiltonian_blocks();
    for (std::size_t index = 0; index < lattice_vectors.size(); index++) {
        const IntegerVector3D& translation = lattice_vectors[index].translation;
        std::size_t inverse_index = lattice_vectors.size();
        for (std::size_t candidate = 0; candidate < lattice_vectors.size(); candidate++) {
            const IntegerVector3D& other = lattice_vectors[candidate].translation;
            if (other.x == -translation.x && other.y == -translation.y &&
                other.z == -translation.z) {
                inverse_index = candidate;
                break;
            }
        }
        ASSERT_LT(inverse_index, lattice_vectors.size());
        EXPECT_NEAR(lattice_vectors[index].bloch_sum_weight,
                    lattice_vectors[inverse_index].bloch_sum_weight,
                    0.0);
        for (int left = 0; left < 2; left++) {
            for (int right = 0; right < 2; right++) {
                const Complex forward = blocks[index][static_cast<std::size_t>(right) * 2 + left];
                const Complex backward =
                    blocks[inverse_index][static_cast<std::size_t>(left) * 2 + right];
                EXPECT_NEAR(forward.real(), backward.real(), 1e-12);
                EXPECT_NEAR(forward.imag(), -backward.imag(), 1e-12);
            }
        }
    }
}

/**
 * The report says what was built. With no overlaps in the record there is
 * no spread to evaluate, and the description claims exactly that much: the
 * projection gauge, and nothing more.
 */
TEST(BuildWannierHamiltonian, TheReportDescribesAnIsolatedManifoldInTheProjectionGauge) {
    const WannierInputs record = two_band_record(3, skewed_projection);
    const WannierConstruction construction =
        WannierHamiltonianBuilder().build(record, simple_cubic_lattice(10.2));
    EXPECT_FALSE(construction.report.disentanglement.performed);
    EXPECT_FALSE(construction.report.spread.evaluated);
    EXPECT_FALSE(construction.report.spread.localized);
    EXPECT_EQ(construction.report.description(), "wannier (projection gauge)");
    EXPECT_GT(construction.report.hamiltonian_decay.outermost_shell_size, 0);
}

/**
 * A preferred localisation is a preference: a record without overlaps
 * builds in the projection gauge and says so, where a required one is an
 * error with the way out named.
 */
TEST(BuildWannierHamiltonian, APreferredLocalizationYieldsToARecordWithoutOverlaps) {
    const WannierInputs record = two_band_record(3, identity_projection);
    WannierConstructionSettings settings;
    settings.localization = LocalizationPolicy::Preferred;
    const WannierConstruction construction =
        WannierHamiltonianBuilder(settings).build(record, simple_cubic_lattice(10.2));
    EXPECT_FALSE(construction.report.spread.localized);
    EXPECT_EQ(construction.report.description(), "wannier (projection gauge)");

    settings.localization = LocalizationPolicy::Required;
    EXPECT_THROW(WannierHamiltonianBuilder(settings).build(record, simple_cubic_lattice(10.2)),
                 WannierDiagnosticError);
}
