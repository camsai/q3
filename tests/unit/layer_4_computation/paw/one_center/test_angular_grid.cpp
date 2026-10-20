/**
 * A2.2 gates — angular quadrature exactness.
 *
 * The one-center XC sums the lm-decomposed density onto real directions,
 * evaluates the functional there, and projects back onto Y_LM. Every step of
 * that round trip is only as good as the quadrature, and a grid that is merely
 * approximate shows up much later as a small systematic XC error that looks
 * like a functional difference. So the grid is pinned by exactness, not by a
 * tolerance on some test integrand.
 *
 * The cross-check against `GauntCoefficientTable` is deliberate: that table
 * and this grid are two independent carriers of the same Y_LM conventions
 * (Condon-Shortley phase, real-harmonic normalization), and this is the first
 * moment both exist and can be compared.
 */

#include "layer_0_core/math/spherical/coefficients.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_4_computation/paw/one_center/angular_grid.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace q3;
using namespace q3::paw_one_center;

namespace {

/// All real harmonics up to l_max at one direction, in Q3's flat lm ordering.
std::vector<Real> harmonics_at(const AngularPoint& point, int maximum_angular_momentum) {
    int count = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);
    std::vector<Real> values(count, 0.0);
    Vector3D direction{point.x, point.y, point.z};
    for (int degree = 0; degree <= maximum_angular_momentum; degree++) {
        for (int order : harmonic_order_sequence(degree)) {
            int index = harmonic_combined_degree_order_index(degree, order);
            values[index] = compute_real_spherical_harmonic(degree, order, direction);
        }
    }
    return values;
}

} // namespace

TEST(AngularGridTest, WeightsSumToFourPi) {
    for (int maximum_angular_momentum : {0, 1, 2, 3, 4}) {
        auto grid = build_angular_grid(maximum_angular_momentum);
        Real total = 0.0;
        for (const auto& point : grid)
            total += point.weight;
        EXPECT_NEAR(total, 4.0 * M_PI, 1e-12) << "l_max = " << maximum_angular_momentum;
    }
}

TEST(AngularGridTest, PointsLieOnTheUnitSphere) {
    auto grid = build_angular_grid(4);
    ASSERT_FALSE(grid.empty());
    for (const auto& point : grid) {
        Real norm = point.x * point.x + point.y * point.y + point.z * point.z;
        EXPECT_NEAR(norm, 1.0, 1e-14);
        EXPECT_GT(point.weight, 0.0);
    }
}

TEST(AngularGridTest, IntegratesSphericalHarmonicProductsExactly) {
    /**
     * The defining requirement: ∫ Y_LM Y_L'M' dΩ = δ. If this is only
     * approximate, every projection back from the angular grid carries a
     * leak between channels — the exact failure that would masquerade as a
     * physical coupling.
     */
    const int maximum_angular_momentum = 4;
    auto grid = build_angular_grid(maximum_angular_momentum);
    const int number_of_harmonics = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);

    // Tabulate all real harmonics at every grid point once.
    std::vector<std::vector<Real>> harmonics(grid.size());
    for (size_t point = 0; point < grid.size(); point++)
        harmonics[point] = harmonics_at(grid[point], maximum_angular_momentum);

    Real worst_diagonal = 0.0;
    Real worst_off_diagonal = 0.0;
    for (int first = 0; first < number_of_harmonics; first++) {
        for (int second = 0; second < number_of_harmonics; second++) {
            Real integral = 0.0;
            for (size_t point = 0; point < grid.size(); point++)
                integral += grid[point].weight * harmonics[point][first] * harmonics[point][second];
            if (first == second)
                worst_diagonal = std::max(worst_diagonal, std::abs(integral - 1.0));
            else
                worst_off_diagonal = std::max(worst_off_diagonal, std::abs(integral));
        }
    }
    EXPECT_LT(worst_diagonal, 1e-13);
    EXPECT_LT(worst_off_diagonal, 1e-13);
}

TEST(AngularGridTest, ReproducesTheGauntCoefficientTable) {
    /**
     * Two independent encodings of the same Y_LM conventions must agree: the
     * existing Gaunt table (used by the ultrasoft augmentation) against
     * numerical integration of triple products on this grid. A
     * Condon-Shortley phase or normalization mismatch between them is a silent
     * factor in every one-center density, and this is the cheapest place to
     * catch it.
     */
    const int maximum_angular_momentum = 2;
    auto grid = build_angular_grid(2 * maximum_angular_momentum);
    const int number_of_harmonics =
        (2 * maximum_angular_momentum + 1) * (2 * maximum_angular_momentum + 1);

    std::vector<std::vector<Real>> harmonics(grid.size());
    for (size_t point = 0; point < grid.size(); point++)
        harmonics[point] = harmonics_at(grid[point], 2 * maximum_angular_momentum);

    auto gaunt = generate_gaunt_coefficients(maximum_angular_momentum);

    int compared = 0;
    Real worst = 0.0;
    int first_count = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);
    for (int first = 0; first < first_count; first++) {
        for (int second = 0; second < first_count; second++) {
            for (int third = 0; third < number_of_harmonics; third++) {
                Real numerical = 0.0;
                for (size_t point = 0; point < grid.size(); point++) {
                    numerical += grid[point].weight * harmonics[point][first] *
                                 harmonics[point][second] * harmonics[point][third];
                }
                Real tabulated = gaunt.coefficients[third][first][second];
                worst = std::max(worst, std::abs(numerical - tabulated));
                compared++;
            }
        }
    }
    EXPECT_GT(compared, 0);
    EXPECT_LT(worst, 1e-12) << "compared " << compared << " triple products";
}
