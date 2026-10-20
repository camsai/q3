#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "layer_0_core/math/spherical/coefficients.hpp"

using namespace q3;

// ─── invert_matrix tests ────────────────────────────────────────────────────

TEST(InvertMatrix, IdentityMatrix) {
    int dimension = 3;
    std::vector<std::vector<Real>> matrix = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    std::vector<std::vector<Real>> inverse;
    invert_matrix(matrix, inverse, dimension);
    for (int row = 0; row < dimension; row++) {
        for (int column = 0; column < dimension; column++) {
            Real expected = (row == column) ? 1.0 : 0.0;
            EXPECT_NEAR(inverse[row][column], expected, 1e-14);
        }
    }
}

TEST(InvertMatrix, TwoByTwoMatrix) {
    // A = [[1, 2], [3, 4]], A^{-1} = [[-2, 1], [1.5, -0.5]]
    int dimension = 2;
    std::vector<std::vector<Real>> matrix = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<Real>> inverse;
    invert_matrix(matrix, inverse, dimension);
    EXPECT_NEAR(inverse[0][0], -2.0, 1e-14);
    EXPECT_NEAR(inverse[0][1], 1.0, 1e-14);
    EXPECT_NEAR(inverse[1][0], 1.5, 1e-14);
    EXPECT_NEAR(inverse[1][1], -0.5, 1e-14);
}

TEST(InvertMatrix, DiagonalMatrix) {
    int dimension = 3;
    std::vector<std::vector<Real>> matrix = {{2.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 5.0}};
    std::vector<std::vector<Real>> inverse;
    invert_matrix(matrix, inverse, dimension);
    EXPECT_NEAR(inverse[0][0], 0.5, 1e-14);
    EXPECT_NEAR(inverse[1][1], 0.25, 1e-14);
    EXPECT_NEAR(inverse[2][2], 0.2, 1e-14);
    EXPECT_NEAR(inverse[0][1], 0.0, 1e-14);
    EXPECT_NEAR(inverse[1][2], 0.0, 1e-14);
}

// ─── generate_deterministic_unit_vectors tests ──────────────────────────────

TEST(DeterministicUnitVectors, CorrectCount) {
    auto vectors = generate_deterministic_unit_vectors(10);
    ASSERT_EQ(static_cast<int>(vectors.size()), 10);
}

TEST(DeterministicUnitVectors, AllUnitLength) {
    auto vectors = generate_deterministic_unit_vectors(20);
    for (const auto& vector : vectors) {
        Real magnitude = vector.norm();
        EXPECT_NEAR(magnitude, 1.0, 1e-14);
    }
}

TEST(DeterministicUnitVectors, Deterministic) {
    auto first_set = generate_deterministic_unit_vectors(10);
    auto second_set = generate_deterministic_unit_vectors(10);
    for (int index = 0; index < 10; index++) {
        EXPECT_DOUBLE_EQ(first_set[index].x, second_set[index].x);
        EXPECT_DOUBLE_EQ(first_set[index].y, second_set[index].y);
        EXPECT_DOUBLE_EQ(first_set[index].z, second_set[index].z);
    }
}

TEST(DeterministicUnitVectors, SinglePoint) {
    auto vectors = generate_deterministic_unit_vectors(1);
    ASSERT_EQ(static_cast<int>(vectors.size()), 1);
    EXPECT_NEAR(vectors[0].norm(), 1.0, 1e-14);
}

// ─── generate_gaunt_coefficients tests ──────────────────────────────────────

TEST(GauntCoefficients, MaxDegreeZero) {
    // max_degree=0: nlm_projector=1, nlm_product=1
    auto table = generate_gaunt_coefficients(0);
    int nlm_projector = 1;
    int nlm_product = 1;
    ASSERT_EQ(static_cast<int>(table.coefficients.size()), nlm_product);
    ASSERT_EQ(static_cast<int>(table.coefficients[0].size()), nlm_projector);
    ASSERT_EQ(static_cast<int>(table.coefficients[0][0].size()), nlm_projector);
    // ap(0, 0, 0) = ∫ Y_00 * Y_00 * Y_00 dΩ = (1/√(4π))³ * 4π = 1/(4π)^{1/2}
    Real expected = 1.0 / std::sqrt(constants::four_pi);
    EXPECT_NEAR(table.coefficients[0][0][0], expected, 1e-6);
}

TEST(GauntCoefficients, TableDimensions) {
    int max_degree = 2;
    auto table = generate_gaunt_coefficients(max_degree);
    int nlm_projector = (max_degree + 1) * (max_degree + 1);       // 9
    int nlm_product = (2 * max_degree + 1) * (2 * max_degree + 1); // 25
    ASSERT_EQ(static_cast<int>(table.coefficients.size()), nlm_product);
    ASSERT_EQ(static_cast<int>(table.coefficients[0].size()), nlm_projector);
    ASSERT_EQ(static_cast<int>(table.number_of_nonzero_product_indices.size()), nlm_projector);
    ASSERT_EQ(static_cast<int>(table.nonzero_product_indices.size()), nlm_projector);
}

TEST(GauntCoefficients, SparsityTablesConsistent) {
    auto table = generate_gaunt_coefficients(1);
    int nlm_projector = 4;
    for (int lm1 = 0; lm1 < nlm_projector; lm1++) {
        for (int lm2 = 0; lm2 < nlm_projector; lm2++) {
            int count = table.number_of_nonzero_product_indices[lm1][lm2];
            int list_size = static_cast<int>(table.nonzero_product_indices[lm1][lm2].size());
            EXPECT_EQ(count, list_size);
        }
    }
}

TEST(GauntCoefficients, OrthogonalityY00WithY1m) {
    /**
     * ∫ Y_00 * Y_1m * Y_00 dΩ = Y_00² * ∫ Y_1m dΩ = 0 for m ≠ 0
     * This means ap(lm1=lm_for_Y10, lm2=lm_for_Y00, lp=lm_for_Y00) should be 0
     */
    auto table = generate_gaunt_coefficients(1);
    /**
     * Y_00 is at lm_index=0, Y_10 is at lm_index=1
     * ap(0, 1, 0) = ∫ Y_00 * Y_10 * Y_00 dΩ = (1/√4π) * ∫ Y_10 Y_00 dΩ = 0 by orthogonality
     */
    EXPECT_NEAR(table.coefficients[0][1][0], 0.0, 1e-6);
    EXPECT_NEAR(table.coefficients[0][0][1], 0.0, 1e-6);
}

// ─── The degree limit, stated rather than silently absorbed ─────────────────

TEST(GauntCoefficients, ADegreeAboveTheClosedFormsIsRefused) {
    /**
     * A projector degree of 3 (f channels) needs product harmonics to degree
     * 6, and compute_real_spherical_harmonic stops at 4. It used to answer
     * degrees 5 and 6 with 0.0, which left the collocation matrix singular in
     * exactly those rows; the inversion skipped the pivots and returned a
     * table whose surviving entries carried errors of order 0.7 against true
     * values of order 0.1, with nothing thrown and nothing logged.
     */
    EXPECT_THROW(generate_gaunt_coefficients(3), std::invalid_argument);
    EXPECT_THROW(generate_gaunt_coefficients(4), std::invalid_argument);
}

TEST(GauntCoefficients, DChannelsAreStillBuilt) {
    /**
     * The boundary is a projector degree of 2, which is where every dataset
     * Q3 ships sits. Refusing one degree lower would reject every ultrasoft
     * and PAW run, so this case pins the side of the boundary that has to
     * keep working.
     */
    EXPECT_NO_THROW(generate_gaunt_coefficients(2));
    auto table = generate_gaunt_coefficients(2);
    EXPECT_EQ(table.coefficients.size(), 25u) << "product harmonics: (2*2+1)^2";
    EXPECT_EQ(table.coefficients[0].size(), 9u) << "projector harmonics: (2+1)^2";
}

TEST(GauntCoefficients, TheSparsityPatternListsOnlyEntriesThatAreNonZero) {
    /**
     * The table is only ever read through this pattern, so a pattern that
     * over-reports costs every consumer the terms it names. Before the
     * argument at the call site was corrected, a d-channel dataset built a
     * table for product degree 8, and its pattern listed 5403 entries against
     * the 708 that are genuinely non-zero -- the surplus being rows the
     * missing harmonics should have filled.
     *
     * Asserting on the table itself is what was missing: the damaged entries
     * had no consumer, so no result-level test could observe them.
     */
    constexpr int max_degree = 2;
    constexpr Real non_zero_threshold = 1e-8;
    auto table = generate_gaunt_coefficients(max_degree);
    const int nlm_projector = (max_degree + 1) * (max_degree + 1);
    const int nlm_product = (2 * max_degree + 1) * (2 * max_degree + 1);

    int listed = 0;
    int actually_non_zero = 0;
    for (int lm1 = 0; lm1 < nlm_projector; lm1++) {
        for (int lm2 = 0; lm2 < nlm_projector; lm2++) {
            listed += table.number_of_nonzero_product_indices[lm1][lm2];
            for (int product_index = 0; product_index < nlm_product; product_index++) {
                if (std::abs(table.coefficients[product_index][lm1][lm2]) > non_zero_threshold) {
                    actually_non_zero++;
                }
            }
            for (int listed_index : table.nonzero_product_indices[lm1][lm2]) {
                ASSERT_GE(listed_index, 0);
                ASSERT_LT(listed_index, nlm_product);
                EXPECT_GT(std::abs(table.coefficients[listed_index][lm1][lm2]), non_zero_threshold)
                    << "the pattern names (" << listed_index << ", " << lm1 << ", " << lm2
                    << "), which is zero";
            }
        }
    }
    EXPECT_EQ(listed, actually_non_zero);
    EXPECT_GT(listed, 0);
}

TEST(GauntCoefficients, TheTableReproducesTheOrthonormalityOfTheProductHarmonics) {
    /**
     * With Y_00 = 1/sqrt(4 pi) constant, the integral of Y_LM Y_lm Y_00
     * collapses to Y_00 times the orthonormality relation, so the table's
     * value at (LM, lm, 0) is 1/sqrt(4 pi) exactly when LM names the same
     * harmonic as lm, and 0 otherwise. That is an exact statement about every
     * entry the collocation produces, not a sampled one.
     */
    constexpr int max_degree = 2;
    auto table = generate_gaunt_coefficients(max_degree);
    const Real y00 = 1.0 / std::sqrt(4.0 * constants::pi);
    const int nlm_projector = (max_degree + 1) * (max_degree + 1);
    const int nlm_product = (2 * max_degree + 1) * (2 * max_degree + 1);

    for (int lm = 0; lm < nlm_projector; lm++) {
        for (int product_index = 0; product_index < nlm_product; product_index++) {
            const Real expected = (product_index == lm) ? y00 : 0.0;
            EXPECT_NEAR(table.coefficients[product_index][lm][0], expected, 1e-10)
                << "product " << product_index << ", projector " << lm;
        }
    }
}
