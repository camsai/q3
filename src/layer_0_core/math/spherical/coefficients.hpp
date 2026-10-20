#pragma once

/**
 * Gaunt coefficients for real spherical harmonics.
 *
 * Gaunt coefficients are the integrals of three spherical harmonics:
 *   ap(lp, lm1, lm2) = ∫ Y_lp(Ω) Y_lm1(Ω) Y_lm2(Ω) dΩ
 *
 * They appear in the angular expansion of augmentation charges (USPP)
 * and are a pure mathematical quantity with no physics dependencies.
 *
 * The table is built by exact collocation. A product of two harmonics of
 * degree <= L is a polynomial on the sphere of degree <= 2L, so it lies
 * exactly in the span of the (2L+1)^2 harmonics of degree <= 2L. Sampling
 * that many directions therefore determines the expansion coefficients with
 * no quadrature error: evaluate the harmonics at the directions, invert the
 * resulting square matrix, and apply the inverse to the sampled products.
 *
 * Collocation is one of several exact routes -- closed forms via
 * Clebsch-Gordan coefficients, and quadrature rules that integrate a triple
 * product exactly, both work. Which route this file should use is an open
 * review item; the reason it uses this one is that it replaced a
 * 10,000-point quadrature and is ~75x faster than that quadrature was.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <vector>

#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Pre-computed Gaunt coefficient table.
 *
 * Stores the sparse Gaunt coefficient tensor for a given maximum
 * degree. The sparsity tables (lpx, lpl) allow efficient iteration
 * over non-zero entries.
 */
struct GauntCoefficientTable {
    /**
     * Gaunt coefficients: ap[lp][lm1][lm2] = ∫ Y_lp Y_lm1 Y_lm2 dΩ
     * Dimensions: [nlm_product][nlm_projector][nlm_projector]
     * where nlm_product = (2*lmax+1)^2 and nlm_projector = (lmax+1)^2
     */
    std::vector<std::vector<std::vector<Real>>> coefficients;

    /**
     * Number of non-zero lp entries for each (lm1, lm2) pair.
     * Dimensions: [nlm_projector][nlm_projector]
     */
    std::vector<std::vector<int>> number_of_nonzero_product_indices;

    /**
     * List of non-zero lp indices for each (lm1, lm2) pair.
     * Dimensions: [nlm_projector][nlm_projector][variable]
     */
    std::vector<std::vector<std::vector<int>>> nonzero_product_indices;
};

/**
 * Invert a square matrix in-place using Gauss-Jordan elimination.
 * Returns the inverse in 'inverse'. The input matrix 'matrix' is modified.
 *
 * @param matrix   [in/out] The matrix to invert (modified in-place)
 * @param inverse  [out] The inverse matrix
 * @param dimension The matrix dimension
 */
inline void invert_matrix(std::vector<std::vector<Real>>& matrix,
                          std::vector<std::vector<Real>>& inverse,
                          int dimension) {
    // Initialize inverse as identity
    inverse.resize(dimension, std::vector<Real>(dimension, 0.0));
    for (int row_index = 0; row_index < dimension; row_index++) {
        inverse[row_index][row_index] = 1.0;
    }

    // Gauss-Jordan elimination with partial pivoting
    for (int column_index = 0; column_index < dimension; column_index++) {
        // Find pivot
        int pivot_row = column_index;
        Real max_value = std::abs(matrix[column_index][column_index]);
        for (int row_index = column_index + 1; row_index < dimension; row_index++) {
            if (std::abs(matrix[row_index][column_index]) > max_value) {
                max_value = std::abs(matrix[row_index][column_index]);
                pivot_row = row_index;
            }
        }

        // Swap rows if needed
        if (pivot_row != column_index) {
            std::swap(matrix[column_index], matrix[pivot_row]);
            std::swap(inverse[column_index], inverse[pivot_row]);
        }

        Real pivot_value = matrix[column_index][column_index];
        if (std::abs(pivot_value) < 1e-15) {
            continue;
        }

        // Scale pivot row
        for (int element_index = 0; element_index < dimension; element_index++) {
            matrix[column_index][element_index] /= pivot_value;
            inverse[column_index][element_index] /= pivot_value;
        }

        // Eliminate column
        for (int row_index = 0; row_index < dimension; row_index++) {
            if (row_index == column_index)
                continue;
            Real factor = matrix[row_index][column_index];
            for (int element_index = 0; element_index < dimension; element_index++) {
                matrix[row_index][element_index] -= factor * matrix[column_index][element_index];
                inverse[row_index][element_index] -= factor * inverse[column_index][element_index];
            }
        }
    }
}

/**
 * Generate deterministic unit vectors on the sphere using a Fibonacci-like spiral.
 * Provides well-distributed points without any randomness.
 *
 * @param number_of_points  Number of unit vectors to generate
 * @return Vector of unit direction vectors
 */
inline std::vector<Vector3D> generate_deterministic_unit_vectors(int number_of_points) {
    std::vector<Vector3D> directions(number_of_points);
    // Use golden-ratio spiral for deterministic, well-distributed points
    Real golden_ratio = constants::pi * (1.0 + std::sqrt(5.0));
    for (int point_index = 0; point_index < number_of_points; point_index++) {
        Real theta = std::acos(1.0 - 2.0 * (point_index + 0.5) / number_of_points);
        Real phi = golden_ratio * point_index;
        directions[point_index] = Vector3D(
            std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
    }
    return directions;
}

/**
 * Generate Gaunt coefficients for spherical harmonics up to a given max degree.
 *
 * Exact collocation: evaluate the harmonics at as many sphere directions as
 * there are product harmonics, invert that square matrix, and apply the
 * inverse to the sampled products. Exact rather than approximate because the
 * product lies in the sampled span; O(n^3) in the product-space dimension
 * rather than O(N_quad * nlm^2 * nlm_prod) for a quadrature.
 *
 * LIMIT: this needs harmonics up to degree 2*max_degree, and
 * compute_real_spherical_harmonic implements degrees 0..4 and returns 0.0
 * above that. For max_degree > 2 the collocation matrix is therefore singular
 * by construction, the inverter skips those pivots, and entries whose two
 * projector degrees sum above 4 come out wrong with no error raised. Entries
 * at or below that sum stay exact. Callers must keep 2*max_degree <= 4 until
 * the harmonics reach l = 5, 6.
 *
 * @param max_degree  Maximum degree for projectors (nlm_proj = (max_degree+1)^2)
 * @return GauntCoefficientTable with coefficients up to product L = 2*max_degree
 */
inline GauntCoefficientTable generate_gaunt_coefficients(int max_degree) {
    GauntCoefficientTable table;

    int nlm_projector = (max_degree + 1) * (max_degree + 1);
    int max_product_degree = 2 * max_degree;
    int nlm_product = (max_product_degree + 1) * (max_product_degree + 1);

    /**
     * `max_degree` is the degree of the *projector* harmonics whose pairwise
     * products this table expands. The collocation below therefore evaluates
     * harmonics up to `2 * max_degree`, and compute_real_spherical_harmonic
     * has closed forms only to degree 4.
     *
     * Checking here rather than letting the harmonic call refuse is what
     * names the real bound to the caller: the limit on this table is a
     * projector degree of 2, not a harmonic degree of 4, and a message about
     * degree 5 would send a reader looking in the wrong place. A d-channel
     * dataset sits exactly at the boundary; an f-channel one is over it and
     * is refused rather than silently given a table whose sampling matrix is
     * singular in the rows the missing harmonics should have filled.
     */
    constexpr int highest_implemented_harmonic_degree = 4;
    if (max_product_degree > highest_implemented_harmonic_degree) {
        throw std::invalid_argument(
            "generate_gaunt_coefficients: a projector degree of " + std::to_string(max_degree) +
            " needs harmonics up to degree " + std::to_string(max_product_degree) +
            ", above the highest implemented (" +
            std::to_string(highest_implemented_harmonic_degree) +
            "). The collocation matrix would be singular in the missing rows and the "
            "table silently wrong, so no table is built. Q3 supports projector "
            "channels up to d.");
    }

    table.coefficients.resize(nlm_product);
    for (auto& ap_row : table.coefficients) {
        ap_row.resize(nlm_projector);
        for (auto& ap_element : ap_row)
            ap_element.resize(nlm_projector, 0.0);
    }
    table.number_of_nonzero_product_indices.resize(nlm_projector);
    table.nonzero_product_indices.resize(nlm_projector);
    for (int lm1 = 0; lm1 < nlm_projector; lm1++) {
        table.number_of_nonzero_product_indices[lm1].resize(nlm_projector, 0);
        table.nonzero_product_indices[lm1].resize(nlm_projector);
    }

    /**
     * One sample direction per product harmonic, and exactly that many.
     * Exact collocation of a function known to lie in an n-dimensional span
     * needs a square n x n sample matrix to invert, and n is nlm_product --
     * so equality is forced, not chosen. It is also the only value this code
     * admits: invert_matrix is called with dimension llx against a matrix
     * allocated nlm_product square.
     *
     * The directions are deterministic (a golden-ratio spiral), not drawn
     * from a generator; test_coefficients.cpp pins that.
     */
    int llx = nlm_product;
    auto directions = generate_deterministic_unit_vectors(llx);

    /**
     * Compute Y_lm for all random vectors: ylm[point][lm]
     * We need up to max_product_degree for the product Y_lm values
     */
    std::vector<std::vector<Real>> ylm_at_points(llx, std::vector<Real>(nlm_product, 0.0));
    for (int point_index = 0; point_index < llx; point_index++) {
        for (int degree = 0; degree <= max_product_degree; degree++) {
            for (int order : harmonic_order_sequence(degree)) {
                int lm_index = harmonic_combined_degree_order_index(degree, order);
                ylm_at_points[point_index][lm_index] =
                    compute_real_spherical_harmonic(degree, order, directions[point_index]);
            }
        }
    }

    /**
     * Copy the sampled harmonics into the square matrix to be inverted,
     * indexed [point][lm]. No transposition happens here or is needed: the
     * inverse of a matrix indexed [point][lm] is naturally indexed
     * [lm][point], which is how the contraction below reads it.
     */
    std::vector<std::vector<Real>> ylm_matrix(nlm_product, std::vector<Real>(nlm_product, 0.0));
    for (int point_index = 0; point_index < llx; point_index++) {
        for (int lm_index = 0; lm_index < nlm_product; lm_index++) {
            ylm_matrix[point_index][lm_index] = ylm_at_points[point_index][lm_index];
        }
    }

    /**
     * Invert the Y_lm matrix: mly = ylm^(-1)
     * mly[lm_product][point] = inverse of ylm[point][lm_product]
     */
    std::vector<std::vector<Real>> mly;
    invert_matrix(ylm_matrix, mly, llx);

    /**
     * Apply the inverse to the sampled products. Since the harmonic matrix
     * maps coefficients to sampled values, its inverse maps sampled values
     * back to coefficients, and the samples of Y_i * Y_j are just the
     * pointwise products of the two columns:
     *
     *   coefficients[LM][i][j] = sum over directions of
     *       inverse[LM][direction] * Y_i(direction) * Y_j(direction)
     *
     * i and j range over the projector harmonics, LM over the product ones.
     */
    for (int lm1 = 0; lm1 < nlm_projector; lm1++) {
        for (int lm2 = 0; lm2 < nlm_projector; lm2++) {
            for (int lp = 0; lp < nlm_product; lp++) {
                Real sum = 0.0;
                for (int point_index = 0; point_index < llx; point_index++) {
                    sum += mly[lp][point_index] * ylm_at_points[point_index][lm1] *
                           ylm_at_points[point_index][lm2];
                }
                table.coefficients[lp][lm1][lm2] = sum;
            }
        }
    }

    // Sparsify: record non-zero entries
    for (int lm1 = 0; lm1 < nlm_projector; lm1++) {
        for (int lm2 = 0; lm2 < nlm_projector; lm2++) {
            for (int lp = 0; lp < nlm_product; lp++) {
                if (std::abs(table.coefficients[lp][lm1][lm2]) >
                    math::tolerances::gaunt_coefficient_threshold) {
                    table.nonzero_product_indices[lm1][lm2].push_back(lp);
                    table.number_of_nonzero_product_indices[lm1][lm2]++;
                } else {
                    table.coefficients[lp][lm1][lm2] = 0.0;
                }
            }
        }
    }

    return table;
}

} // namespace q3
