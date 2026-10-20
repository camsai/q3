#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/lattice_fourier_transform.hpp"

using namespace q3;
namespace transform = q3::lattice_fourier_transform;

namespace {

/**
 * A three-point mesh along the first axis with the symmetric lattice set
 * {-1, 0, 1}. For an odd mesh this pair is exactly orthogonal -- the phases
 * sum to a Kronecker delta modulo three -- so the round trip below is an
 * identity, not an approximation.
 */
std::vector<Vector3D> three_point_mesh() {
    return {Vector3D(0.0, 0.0, 0.0), Vector3D(1.0 / 3.0, 0.0, 0.0), Vector3D(2.0 / 3.0, 0.0, 0.0)};
}

std::vector<IntegerVector3D> symmetric_translations() {
    return {IntegerVector3D(-1, 0, 0), IntegerVector3D(0, 0, 0), IntegerVector3D(1, 0, 0)};
}

} // namespace

TEST(LatticeFourierTransform, RoundTripAtMeshPointsIsAnIdentity) {
    const std::vector<Vector3D> kpoints = three_point_mesh();
    const std::vector<IntegerVector3D> translations = symmetric_translations();
    const std::vector<Real> weights(translations.size(), 1.0);

    /// A 2x2 block per k-point, Hermitian in the way an H(k) would be.
    const std::vector<std::vector<Complex>> values_at_kpoints = {
        {Complex(1.0, 0.0), Complex(0.2, 0.3), Complex(0.2, -0.3), Complex(-0.5, 0.0)},
        {Complex(0.4, 0.0), Complex(-0.1, 0.7), Complex(-0.1, -0.7), Complex(0.9, 0.0)},
        {Complex(-0.8, 0.0), Complex(0.6, -0.2), Complex(0.6, 0.2), Complex(0.3, 0.0)}};

    const auto values_at_lattice_vectors =
        transform::to_real_space(kpoints, values_at_kpoints, translations);
    ASSERT_EQ(values_at_lattice_vectors.size(), translations.size());

    for (std::size_t kpoint_index = 0; kpoint_index < kpoints.size(); kpoint_index++) {
        const std::vector<Complex> recovered = transform::to_kpoint(
            kpoints[kpoint_index], values_at_lattice_vectors, translations, weights);
        ASSERT_EQ(recovered.size(), values_at_kpoints[kpoint_index].size());
        for (std::size_t element = 0; element < recovered.size(); element++) {
            EXPECT_NEAR(
                recovered[element].real(), values_at_kpoints[kpoint_index][element].real(), 1e-12)
                << "k-point " << kpoint_index << ", element " << element;
            EXPECT_NEAR(
                recovered[element].imag(), values_at_kpoints[kpoint_index][element].imag(), 1e-12);
        }
    }
}

TEST(LatticeFourierTransform, AScalarIsABlockOfOne) {
    /**
     * The Fourier star-function method fits a single band energy per k-point.
     * It must go through the same code as a matrix, so a block of size one is
     * a supported shape rather than a special case.
     */
    const std::vector<Vector3D> kpoints = three_point_mesh();
    const std::vector<IntegerVector3D> translations = symmetric_translations();
    const std::vector<Real> weights(translations.size(), 1.0);
    const std::vector<std::vector<Complex>> band_energy = {
        {Complex(-1.0, 0.0)}, {Complex(0.5, 0.0)}, {Complex(0.5, 0.0)}};

    const auto coefficients = transform::to_real_space(kpoints, band_energy, translations);
    for (std::size_t kpoint_index = 0; kpoint_index < kpoints.size(); kpoint_index++) {
        const std::vector<Complex> recovered =
            transform::to_kpoint(kpoints[kpoint_index], coefficients, translations, weights);
        ASSERT_EQ(recovered.size(), 1u);
        EXPECT_NEAR(recovered[0].real(), band_energy[kpoint_index][0].real(), 1e-12);
    }
}

TEST(LatticeFourierTransform, EvaluatesBetweenMeshPointsWithoutRingingOnAConstant) {
    /**
     * A constant function transforms to a single nonzero coefficient at R = 0,
     * so every off-mesh k must return that same constant. This is the one
     * off-mesh value that has an exact answer, which makes it the check that
     * the inverse's phase convention and weights are right.
     */
    const std::vector<Vector3D> kpoints = three_point_mesh();
    const std::vector<IntegerVector3D> translations = symmetric_translations();
    const std::vector<Real> weights(translations.size(), 1.0);
    const std::vector<std::vector<Complex>> constant_value(3, {Complex(2.5, 0.0)});

    const auto coefficients = transform::to_real_space(kpoints, constant_value, translations);
    EXPECT_NEAR(coefficients[0][0].real(), 0.0, 1e-12) << "R = -1 must carry nothing";
    EXPECT_NEAR(coefficients[1][0].real(), 2.5, 1e-12) << "R = 0 carries the constant";
    EXPECT_NEAR(coefficients[2][0].real(), 0.0, 1e-12) << "R = +1 must carry nothing";

    for (const Real fraction : {0.1, 0.25, 0.5}) {
        const std::vector<Complex> off_mesh =
            transform::to_kpoint(Vector3D(fraction, 0.0, 0.0), coefficients, translations, weights);
        EXPECT_NEAR(off_mesh[0].real(), 2.5, 1e-12) << "at k = " << fraction;
        EXPECT_NEAR(off_mesh[0].imag(), 0.0, 1e-12);
    }
}

TEST(LatticeFourierTransform, DegeneracyWeightsScaleTheInverseSum) {
    /**
     * A Wigner-Seitz boundary vector is shared by its periodic images and
     * enters the Bloch sum with 1/degeneracy. Halving one weight must halve
     * that vector's contribution and nothing else.
     */
    const std::vector<IntegerVector3D> translations = symmetric_translations();
    const std::vector<std::vector<Complex>> blocks = {
        {Complex(1.0, 0.0)}, {Complex(0.0, 0.0)}, {Complex(0.0, 0.0)}};

    const std::vector<Complex> full =
        transform::to_kpoint(Vector3D(0.0, 0.0, 0.0), blocks, translations, {1.0, 1.0, 1.0});
    const std::vector<Complex> halved =
        transform::to_kpoint(Vector3D(0.0, 0.0, 0.0), blocks, translations, {0.5, 1.0, 1.0});

    EXPECT_NEAR(full[0].real(), 1.0, 1e-12);
    EXPECT_NEAR(halved[0].real(), 0.5, 1e-12);
}

TEST(LatticeFourierTransform, MismatchedShapesAreRefused) {
    const std::vector<Vector3D> kpoints = three_point_mesh();
    const std::vector<IntegerVector3D> translations = symmetric_translations();

    EXPECT_THROW(transform::to_real_space(kpoints, {{Complex(1.0, 0.0)}}, translations),
                 std::invalid_argument)
        << "one block for three k-points";

    const std::vector<std::vector<Complex>> ragged = {
        {Complex(1.0, 0.0)}, {Complex(1.0, 0.0), Complex(2.0, 0.0)}, {Complex(1.0, 0.0)}};
    EXPECT_THROW(transform::to_real_space(kpoints, ragged, translations), std::invalid_argument)
        << "blocks of differing size";

    EXPECT_THROW(transform::to_kpoint(
                     Vector3D(0.0, 0.0, 0.0), {{Complex(1.0, 0.0)}}, translations, {1.0, 1.0, 1.0}),
                 std::invalid_argument)
        << "one block for three lattice vectors";

    EXPECT_THROW(transform::to_real_space({}, {}, translations), std::invalid_argument)
        << "no k-points";
}
