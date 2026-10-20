#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "layer_0_core/math/spherical/coefficients.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/vector_3d.hpp"

using namespace q3;

// ─── Spherical Harmonics Tests ──────────────────────────────────────────────

TEST(SphericalHarmonics, Y00IsConstantEverywhere) {
    // Y_00 = 1 / (2 sqrt(pi)) = 1 / sqrt(4pi)
    Real expected_y00 = 1.0 / std::sqrt(constants::four_pi);

    // Test at several directions
    Vector3D z_axis(0.0, 0.0, 1.0);
    Vector3D x_axis(1.0, 0.0, 0.0);
    Vector3D y_axis(0.0, 1.0, 0.0);
    Vector3D diagonal(1.0, 1.0, 1.0);
    Vector3D negative_direction(-1.0, -2.0, 3.0);

    EXPECT_NEAR(compute_real_spherical_harmonic(0, 0, z_axis), expected_y00, 1e-14);
    EXPECT_NEAR(compute_real_spherical_harmonic(0, 0, x_axis), expected_y00, 1e-14);
    EXPECT_NEAR(compute_real_spherical_harmonic(0, 0, y_axis), expected_y00, 1e-14);
    EXPECT_NEAR(compute_real_spherical_harmonic(0, 0, diagonal), expected_y00, 1e-14);
    EXPECT_NEAR(compute_real_spherical_harmonic(0, 0, negative_direction), expected_y00, 1e-14);
}

TEST(SphericalHarmonics, Y00AtZeroVector) {
    // At zero vector, Y_00 should still be 1/sqrt(4pi) by convention
    Real expected_y00 = 1.0 / std::sqrt(constants::four_pi);
    Vector3D zero_vector(0.0, 0.0, 0.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(0, 0, zero_vector), expected_y00, 1e-14);
}

TEST(SphericalHarmonics, Y10ProportionalToCosTheta) {
    /**
     * Y_10 = sqrt(3/4pi) * cos(theta)
     * Along z-axis: cos(theta) = 1
     */
    Real normalization = std::sqrt(3.0 / constants::four_pi);

    Vector3D z_axis(0.0, 0.0, 1.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, 0, z_axis), normalization * 1.0, 1e-14);

    // Along negative z-axis: cos(theta) = -1
    Vector3D negative_z(0.0, 0.0, -1.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, 0, negative_z), normalization * (-1.0), 1e-14);

    // In the xy-plane: cos(theta) = 0
    Vector3D x_axis(1.0, 0.0, 0.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, 0, x_axis), 0.0, 1e-14);
}

TEST(SphericalHarmonics, Y11ProportionalToSinThetaCosPhi) {
    /**
     * Real Y_11 = -sqrt(3/4pi) * sin(theta) * cos(phi) = -sqrt(3/4pi) * x/r
     * The Condon-Shortley phase gives the negative sign
     */
    Real normalization = std::sqrt(3.0 / constants::four_pi);

    // Along x-axis: x/r = 1, so Y_11 = -normalization
    Vector3D x_axis(1.0, 0.0, 0.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, 1, x_axis), -normalization, 1e-14);

    // Along z-axis: sin(theta) = 0, so Y_11 = 0
    Vector3D z_axis(0.0, 0.0, 1.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, 1, z_axis), 0.0, 1e-14);

    // Along y-axis: cos(phi) = 0, so Y_11 = 0
    Vector3D y_axis(0.0, 1.0, 0.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, 1, y_axis), 0.0, 1e-14);
}

TEST(SphericalHarmonics, Y1Minus1ProportionalToSinThetaSinPhi) {
    // Real Y_1,-1 = -sqrt(3/4pi) * sin(theta) * sin(phi) = -sqrt(3/4pi) * y/r
    Real normalization = std::sqrt(3.0 / constants::four_pi);

    // Along y-axis: y/r = 1, so Y_1,-1 = -normalization
    Vector3D y_axis(0.0, 1.0, 0.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, -1, y_axis), -normalization, 1e-14);

    // Along z-axis: sin(theta) = 0
    Vector3D z_axis(0.0, 0.0, 1.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, -1, z_axis), 0.0, 1e-14);

    // Along x-axis: sin(phi) = 0
    Vector3D x_axis(1.0, 0.0, 0.0);
    EXPECT_NEAR(compute_real_spherical_harmonic(1, -1, x_axis), 0.0, 1e-14);
}

TEST(SphericalHarmonics, HigherLVanishAtZeroVector) {
    // For l > 0, Y_lm at zero vector should return 0
    Vector3D zero_vector(0.0, 0.0, 0.0);
    for (int degree = 1; degree <= 4; degree++) {
        for (int order : harmonic_order_sequence(degree)) {
            EXPECT_NEAR(compute_real_spherical_harmonic(degree, order, zero_vector), 0.0, 1e-14)
                << "Failed for degree=" << degree << ", order=" << order;
        }
    }
}

TEST(SphericalHarmonics, NormalizationByNumericalIntegration) {
    /**
     * Verify normalization: integral of |Y_lm|^2 over the sphere = 1
     * Use the batch function evaluated at many points on the sphere
     * and sum (which approximates the integral with equal-area quadrature)
     */
    int number_of_points = 5000;
    auto directions = generate_deterministic_unit_vectors(number_of_points);

    // Test for l=0,1,2
    for (int degree = 0; degree <= 2; degree++) {
        for (int order : harmonic_order_sequence(degree)) {
            Real sum_squared = 0.0;
            for (int point_index = 0; point_index < number_of_points; point_index++) {
                Real value =
                    compute_real_spherical_harmonic(degree, order, directions[point_index]);
                sum_squared += value * value;
            }
            // The equal-area quadrature weight is 4*pi / N for each point
            Real integral = sum_squared * constants::four_pi / number_of_points;
            EXPECT_NEAR(integral, 1.0, 0.01)
                << "Normalization failed for degree=" << degree << ", order=" << order;
        }
    }
}

TEST(SphericalHarmonics, OrthogonalityByNumericalIntegration) {
    // Verify orthogonality: integral of Y_lm * Y_l'm' over sphere = 0 for (l,m) != (l',m')
    int number_of_points = 5000;
    auto directions = generate_deterministic_unit_vectors(number_of_points);

    // Test orthogonality between Y_00 and Y_10
    Real sum_product = 0.0;
    for (int point_index = 0; point_index < number_of_points; point_index++) {
        Real y00 = compute_real_spherical_harmonic(0, 0, directions[point_index]);
        Real y10 = compute_real_spherical_harmonic(1, 0, directions[point_index]);
        sum_product += y00 * y10;
    }
    Real integral = sum_product * constants::four_pi / number_of_points;
    EXPECT_NEAR(integral, 0.0, 0.01);

    // Test orthogonality between Y_10 and Y_11
    sum_product = 0.0;
    for (int point_index = 0; point_index < number_of_points; point_index++) {
        Real y10 = compute_real_spherical_harmonic(1, 0, directions[point_index]);
        Real y11 = compute_real_spherical_harmonic(1, 1, directions[point_index]);
        sum_product += y10 * y11;
    }
    integral = sum_product * constants::four_pi / number_of_points;
    EXPECT_NEAR(integral, 0.0, 0.01);
}

TEST(SphericalHarmonics, HarmonicOrderSequenceL0) {
    auto sequence = harmonic_order_sequence(0);
    ASSERT_EQ(sequence.size(), 1u);
    EXPECT_EQ(sequence[0], 0);
}

TEST(SphericalHarmonics, HarmonicOrderSequenceL1) {
    auto sequence = harmonic_order_sequence(1);
    ASSERT_EQ(sequence.size(), 3u);
    EXPECT_EQ(sequence[0], 0);
    EXPECT_EQ(sequence[1], 1);
    EXPECT_EQ(sequence[2], -1);
}

TEST(SphericalHarmonics, HarmonicOrderSequenceL2) {
    auto sequence = harmonic_order_sequence(2);
    ASSERT_EQ(sequence.size(), 5u);
    EXPECT_EQ(sequence[0], 0);
    EXPECT_EQ(sequence[1], 1);
    EXPECT_EQ(sequence[2], -1);
    EXPECT_EQ(sequence[3], 2);
    EXPECT_EQ(sequence[4], -2);
}

TEST(SphericalHarmonics, CombinedDegreeOrderIndex) {
    // l=0, m=0 -> 0
    EXPECT_EQ(harmonic_combined_degree_order_index(0, 0), 0);
    // l=1, m=0 -> 1
    EXPECT_EQ(harmonic_combined_degree_order_index(1, 0), 1);
    // l=1, m=1 -> 2
    EXPECT_EQ(harmonic_combined_degree_order_index(1, 1), 2);
    // l=1, m=-1 -> 3
    EXPECT_EQ(harmonic_combined_degree_order_index(1, -1), 3);
    // l=2, m=0 -> 4
    EXPECT_EQ(harmonic_combined_degree_order_index(2, 0), 4);
}

// ─── Gaunt Coefficient Tests ────────────────────────────────────────────────

// ─── Batch Y_lm ─────────────────────────────────────────────────────────────

/**
 * The batch routine had no coverage of its own: every gate on it was
 * end-to-end, so any disagreement with the closed form showed up only as a
 * slightly wrong total energy. These three pin it directly.
 */

TEST(SphericalHarmonicsBatch, AgreesWithTheClosedFormInEveryComponent) {
    /**
     * The batch builds Y_lm by upward recursion; compute_real_spherical_harmonic
     * evaluates the closed forms. They are independent routes to the same
     * numbers, so they should agree to rounding for every (l,m) and direction.
     */
    const int maximum_degree = 4;
    const int number_of_lm_components = (maximum_degree + 1) * (maximum_degree + 1);

    std::vector<std::array<Real, 3>> g_vectors = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {-1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0},
        {0.0, 0.0, -1.0},
        {1.0, 1.0, 1.0},
        {-1.0, -1.0, -1.0},
        {3.0, -4.0, 5.0},
        {-3.0, -4.0, -5.0},
        {0.0, 2.0, -3.0},
        {-2.0, 0.0, 3.0},
        {0.5, -0.5, 2.0},
        {7.0, 0.0, -1.0},
    };
    std::vector<Real> g_squared;
    for (const auto& g : g_vectors)
        g_squared.push_back(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);

    std::vector<std::vector<Real>> batch;
    compute_real_spherical_harmonics_batch(
        number_of_lm_components, static_cast<int>(g_vectors.size()), g_vectors, g_squared, batch);

    for (size_t g_index = 0; g_index < g_vectors.size(); g_index++) {
        Vector3D direction(g_vectors[g_index][0], g_vectors[g_index][1], g_vectors[g_index][2]);
        for (int degree = 0; degree <= maximum_degree; degree++) {
            for (int order = -degree; order <= degree; order++) {
                int lm = harmonic_combined_degree_order_index(degree, order);
                EXPECT_NEAR(batch[g_index][lm],
                            compute_real_spherical_harmonic(degree, order, direction),
                            1e-13)
                    << "l=" << degree << " m=" << order << " g=" << g_index;
            }
        }
    }
}

TEST(SphericalHarmonicsBatch, IsOrthonormalOverTheSphere) {
    /**
     * Integrating Y_lm * Y_l'm' over a Lebedev-free product grid gives the
     * identity if and only if both the normalization and the (l,m) slot
     * assignment are right — which is what a recursion plus an index map can
     * get subtly wrong without changing any single value's magnitude.
     *
     * The polar rule has to be Gauss-Legendre in cos(theta) for that argument
     * to hold. A product Y_lm Y_l'm' is a polynomial of degree at most
     * 2 * maximum_degree in cos(theta) once the azimuth is integrated out, so
     * eight nodes are exact through degree 15 and leave the harmonics' own
     * error as the only residual. A uniform midpoint rule in theta is merely
     * second-order and does not: at 60 points it leaves 1.1e-4 in
     * <Y_00, Y_00> and 8.0e-4 in <Y_30, Y_30>, all of it the grid's error and
     * none of it the routine's, which buries the deviation this test exists to
     * detect under a hundred times its size. The azimuthal rule stays a
     * uniform midpoint sum: it is spectrally exact for the cos(m phi) and
     * sin(m phi) modes, whose highest harmonic here is 2 * maximum_degree,
     * well under its sixteen-point Nyquist limit.
     */
    const int maximum_degree = 3;
    const int number_of_lm_components = (maximum_degree + 1) * (maximum_degree + 1);
    const int phi_points = 16;

    /**
     * Gauss-Legendre nodes and weights on [-1, 1] for eight points, tabulated
     * rather than iterated for: a rule this test depends on should be readable
     * against any standard table, and the sum-to-4pi check below pins them.
     */
    const std::array<Real, 8> cosines = {-0.9602898564975362,
                                         -0.7966664774136267,
                                         -0.5255324099163290,
                                         -0.1834346424956498,
                                         0.1834346424956498,
                                         0.5255324099163290,
                                         0.7966664774136267,
                                         0.9602898564975362};
    const std::array<Real, 8> cosine_weights = {0.1012285362903763,
                                                0.2223810344533745,
                                                0.3137066458778873,
                                                0.3626837833783620,
                                                0.3626837833783620,
                                                0.3137066458778873,
                                                0.2223810344533745,
                                                0.1012285362903763};

    std::vector<std::array<Real, 3>> g_vectors;
    std::vector<Real> weights;
    for (size_t i = 0; i < cosines.size(); i++) {
        Real cosine_theta = cosines[i];
        Real sine_theta = std::sqrt(1.0 - cosine_theta * cosine_theta);
        for (int j = 0; j < phi_points; j++) {
            Real phi = 2.0 * constants::pi * (j + 0.5) / phi_points;
            g_vectors.push_back(
                {sine_theta * std::cos(phi), sine_theta * std::sin(phi), cosine_theta});
            weights.push_back(cosine_weights[i] * (2.0 * constants::pi / phi_points));
        }
    }
    std::vector<Real> g_squared(g_vectors.size(), 1.0);

    /**
     * The rule integrates the constant 1 over the sphere, so a mistyped node or
     * weight fails here rather than silently loosening every overlap below.
     */
    Real solid_angle = 0.0;
    for (Real weight : weights)
        solid_angle += weight;
    EXPECT_NEAR(solid_angle, constants::four_pi, 1e-13);

    std::vector<std::vector<Real>> batch;
    compute_real_spherical_harmonics_batch(
        number_of_lm_components, static_cast<int>(g_vectors.size()), g_vectors, g_squared, batch);

    for (int a = 0; a < number_of_lm_components; a++) {
        for (int b = a; b < number_of_lm_components; b++) {
            Real overlap = 0.0;
            for (size_t g_index = 0; g_index < g_vectors.size(); g_index++)
                overlap += weights[g_index] * batch[g_index][a] * batch[g_index][b];
            EXPECT_NEAR(overlap, (a == b) ? 1.0 : 0.0, 1e-12) << "lm " << a << "," << b;
        }
    }
}

TEST(SphericalHarmonicsBatch, ZeroVectorUsesTheDocumentedDegenerateConvention) {
    /**
     * G = 0 has no direction, but a plane-wave batch always contains it and the
     * routine must return something. The convention is the +y meridian
     * (theta = pi/2, phi = +pi/2); it is arbitrary, which is exactly why it is
     * pinned here rather than left to fall out of whichever inverse tangent the
     * implementation happens to call.
     */
    const int maximum_degree = 2;
    const int number_of_lm_components = (maximum_degree + 1) * (maximum_degree + 1);
    std::vector<std::array<Real, 3>> g_vectors = {{0.0, 0.0, 0.0}};
    std::vector<Real> g_squared = {0.0};

    std::vector<std::vector<Real>> batch;
    compute_real_spherical_harmonics_batch(number_of_lm_components, 1, g_vectors, g_squared, batch);

    // Y_00 is direction-independent.
    EXPECT_NEAR(batch[0][harmonic_combined_degree_order_index(0, 0)],
                1.0 / std::sqrt(constants::four_pi),
                1e-14);
    // theta = pi/2 puts cos(theta) = 0, so Y_10 vanishes.
    EXPECT_NEAR(batch[0][harmonic_combined_degree_order_index(1, 0)], 0.0, 1e-14);
    // phi = +pi/2 puts the weight on the sin(phi) partner, not the cos(phi) one.
    EXPECT_NEAR(batch[0][harmonic_combined_degree_order_index(1, 1)], 0.0, 1e-14);
    EXPECT_LT(batch[0][harmonic_combined_degree_order_index(1, -1)], -1e-3);
}

TEST(GauntCoefficient, SymmetryInFirstTwoIndices) {
    /**
     * Gaunt coefficients have symmetry: ap(lp, lm1, lm2) = ap(lp, lm2, lm1)
     * because the integral of Y_lp * Y_lm1 * Y_lm2 is symmetric in lm1 <-> lm2
     */
    int max_degree = 2;
    auto table = generate_gaunt_coefficients(max_degree);

    int nlm_projector = (max_degree + 1) * (max_degree + 1);
    int nlm_product = (2 * max_degree + 1) * (2 * max_degree + 1);

    for (int lp = 0; lp < nlm_product; lp++) {
        for (int lm1 = 0; lm1 < nlm_projector; lm1++) {
            for (int lm2 = lm1; lm2 < nlm_projector; lm2++) {
                EXPECT_NEAR(
                    table.coefficients[lp][lm1][lm2], table.coefficients[lp][lm2][lm1], 1e-10)
                    << "Symmetry failed at lp=" << lp << ", lm1=" << lm1 << ", lm2=" << lm2;
            }
        }
    }
}

TEST(GauntCoefficient, Y00CouplingIsKroneckerDelta) {
    // ap(0, lm1, lm2) = integral Y_00 * Y_lm1 * Y_lm2 = (1/sqrt(4pi)) * delta_{lm1,lm2}
    int max_degree = 2;
    auto table = generate_gaunt_coefficients(max_degree);

    int nlm_projector = (max_degree + 1) * (max_degree + 1);
    Real y00_value = 1.0 / std::sqrt(constants::four_pi);

    for (int lm1 = 0; lm1 < nlm_projector; lm1++) {
        for (int lm2 = 0; lm2 < nlm_projector; lm2++) {
            Real expected = (lm1 == lm2) ? y00_value : 0.0;
            EXPECT_NEAR(table.coefficients[0][lm1][lm2], expected, 1e-8)
                << "Y00 coupling failed at lm1=" << lm1 << ", lm2=" << lm2;
        }
    }
}

TEST(GauntCoefficient, SparsityTableConsistency) {
    // Verify that the sparsity tables are consistent with the coefficient values
    int max_degree = 2;
    auto table = generate_gaunt_coefficients(max_degree);

    int nlm_projector = (max_degree + 1) * (max_degree + 1);

    for (int lm1 = 0; lm1 < nlm_projector; lm1++) {
        for (int lm2 = 0; lm2 < nlm_projector; lm2++) {
            int reported_count = table.number_of_nonzero_product_indices[lm1][lm2];
            int actual_indices_count =
                static_cast<int>(table.nonzero_product_indices[lm1][lm2].size());
            EXPECT_EQ(reported_count, actual_indices_count)
                << "Count mismatch at lm1=" << lm1 << ", lm2=" << lm2;

            // Each listed index should have a non-zero coefficient
            for (int entry_index = 0; entry_index < actual_indices_count; entry_index++) {
                int lp = table.nonzero_product_indices[lm1][lm2][entry_index];
                EXPECT_GT(std::abs(table.coefficients[lp][lm1][lm2]),
                          math::tolerances::gaunt_coefficient_threshold * 0.1)
                    << "Listed non-zero index has zero coefficient at lp=" << lp << ", lm1=" << lm1
                    << ", lm2=" << lm2;
            }
        }
    }
}

TEST(GauntCoefficient, NonZeroCountForSameIndices) {
    // For ap(lp, lm, lm), there should always be at least one non-zero entry (lp=0 at minimum)
    int max_degree = 2;
    auto table = generate_gaunt_coefficients(max_degree);

    int nlm_projector = (max_degree + 1) * (max_degree + 1);
    for (int lm = 0; lm < nlm_projector; lm++) {
        EXPECT_GE(table.number_of_nonzero_product_indices[lm][lm], 1)
            << "Expected at least one non-zero Gaunt coefficient for diagonal lm=" << lm;
    }
}

// ─── The closed forms stop at degree 4, and say so ──────────────────────────

TEST(SphericalHarmonics, ADegreeAboveFourIsRefusedRatherThanReturnedAsZero) {
    /**
     * Zero is a legal value of every harmonic, so returning it for a degree
     * with no closed form was indistinguishable from an answer: a caller got
     * a number, used it, and produced a result with no error anywhere in the
     * chain. The Gaunt collocation is the case that made this concrete.
     */
    const Vector3D direction{0.3, -0.4, 0.5};
    EXPECT_THROW(compute_real_spherical_harmonic(5, 0, direction), std::invalid_argument);
    EXPECT_THROW(compute_real_spherical_harmonic(6, -3, direction), std::invalid_argument);
}

TEST(SphericalHarmonics, EveryImplementedDegreeStillAnswers) {
    /**
     * The other side of the boundary: degrees 0 through 4 are what the
     * refusal must not touch, at every order they define.
     */
    const Vector3D direction{0.3, -0.4, 0.5};
    for (int degree = 0; degree <= 4; degree++) {
        for (int order : harmonic_order_sequence(degree)) {
            EXPECT_NO_THROW({
                volatile Real value = compute_real_spherical_harmonic(degree, order, direction);
                (void)value;
            }) << "degree "
               << degree << ", order " << order;
        }
    }
}

TEST(SphericalHarmonics, TheZeroDirectionIsStillAnsweredAtEveryImplementedDegree) {
    /**
     * The degenerate-direction branch returns before the degree dispatch, so
     * it must keep its old behaviour rather than inherit the refusal: a zero
     * vector has no direction, and the convention callers were given is
     * Y_00 for degree 0 and 0 elsewhere.
     */
    const Vector3D origin{0.0, 0.0, 0.0};
    EXPECT_NEAR(
        compute_real_spherical_harmonic(0, 0, origin), 1.0 / std::sqrt(4.0 * constants::pi), 1e-14);
    EXPECT_NO_THROW(compute_real_spherical_harmonic(4, -2, origin));
}
