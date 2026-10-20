#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "io/volumetric/inverse_transform.hpp"
#include "layer_0_core/types/constants.hpp"

using namespace q3;

namespace {

constexpr std::array<int, 3> grid = {4, 6, 8};

int flat_index(int index_1, int index_2, int index_3) {
    return (index_1 * grid[1] + index_2) * grid[2] + index_3;
}

} // namespace

TEST(VolumetricInverseTransform, ConstantComesFromTheOriginCoefficientAlone) {
    const std::vector<IntegerVector3D> millers = {{0, 0, 0}};
    const std::vector<Complex> coefficients = {Complex(2.5, 0.0)};

    const std::vector<Real> field =
        real_field_from_reciprocal_space(grid, millers, coefficients, /*holds_half_sphere=*/false);

    ASSERT_EQ(field.size(), static_cast<std::size_t>(grid[0] * grid[1] * grid[2]));
    for (Real value : field)
        EXPECT_NEAR(value, 2.5, 1e-12);
}

TEST(VolumetricInverseTransform, OnePlaneWaveVariesAlongItsOwnAxis) {
    /**
     * A coefficient at Miller (0, 1, 0) must produce exp(2*pi*i*j2/n2): it has
     * to vary along the SECOND grid axis and be constant along the other two.
     * An index-order slip inside the scatter shows up here and nowhere else,
     * because every field a real calculation produces is smooth in all three.
     */
    const std::vector<IntegerVector3D> millers = {{0, 1, 0}};
    const std::vector<Complex> coefficients = {Complex(1.0, 0.0)};

    const std::vector<Complex> field = complex_field_from_reciprocal_space(
        grid, millers, coefficients, /*holds_half_sphere=*/false);

    for (int index_1 = 0; index_1 < grid[0]; index_1++) {
        for (int index_2 = 0; index_2 < grid[1]; index_2++) {
            const Real phase = constants::two_pi * index_2 / grid[1];
            for (int index_3 = 0; index_3 < grid[2]; index_3++) {
                const Complex value = field[flat_index(index_1, index_2, index_3)];
                EXPECT_NEAR(value.real(), std::cos(phase), 1e-12);
                EXPECT_NEAR(value.imag(), std::sin(phase), 1e-12);
            }
        }
    }
}

TEST(VolumetricInverseTransform, HalfSphereFillsTheMissingConjugateAndGivesARealField) {
    const std::vector<IntegerVector3D> millers = {{0, 0, 0}, {1, 0, 0}};
    const std::vector<Complex> coefficients = {Complex(1.0, 0.0), Complex(0.5, 0.0)};

    const std::vector<Complex> field =
        complex_field_from_reciprocal_space(grid,
                                            millers,
                                            coefficients,
                                            /*holds_half_sphere=*/true);

    for (int index_1 = 0; index_1 < grid[0]; index_1++) {
        const Real expected = 1.0 + std::cos(constants::two_pi * index_1 / grid[0]);
        for (int index_2 = 0; index_2 < grid[1]; index_2++) {
            for (int index_3 = 0; index_3 < grid[2]; index_3++) {
                const Complex value = field[flat_index(index_1, index_2, index_3)];
                EXPECT_NEAR(value.real(), expected, 1e-12);
                EXPECT_NEAR(value.imag(), 0.0, 1e-12);
            }
        }
    }
}

TEST(VolumetricInverseTransform, ParsevalHoldsSoANormalizedBandIntegratesToOne) {
    /**
     * `sum_r |f(r)|^2 / N = sum_G |c_G|^2`, which is the identity the band
     * probability density's normalization rests on: with `sum_G |c_G|^2 = 1`
     * the field `|f|^2 / Omega` integrates over the cell to exactly 1.
     */
    const std::vector<IntegerVector3D> millers = {{0, 0, 0}, {1, 0, 0}, {0, -2, 1}, {1, 1, -1}};
    const std::vector<Complex> coefficients = {
        Complex(0.4, 0.1), Complex(-0.2, 0.5), Complex(0.3, -0.3), Complex(0.1, 0.2)};

    Real reciprocal_space_sum = 0.0;
    for (const Complex& coefficient : coefficients)
        reciprocal_space_sum += std::norm(coefficient);

    const std::vector<Complex> field = complex_field_from_reciprocal_space(
        grid, millers, coefficients, /*holds_half_sphere=*/false);
    Real real_space_sum = 0.0;
    for (const Complex& value : field)
        real_space_sum += std::norm(value);

    EXPECT_NEAR(real_space_sum / field.size(), reciprocal_space_sum, 1e-12);
}

TEST(VolumetricInverseTransform, CoefficientsOutsideTheGridAreDroppedRatherThanWrapped) {
    /**
     * Miller index 3 does not fit a four-point axis (only -2..2 do). Wrapping
     * it would alias a high frequency onto a low one and give a smooth, wrong
     * field; dropping it leaves the rest untouched.
     */
    const std::vector<IntegerVector3D> millers = {{0, 0, 0}, {3, 0, 0}};
    const std::vector<Complex> coefficients = {Complex(1.0, 0.0), Complex(7.0, 0.0)};

    const std::vector<Real> field =
        real_field_from_reciprocal_space(grid, millers, coefficients, /*holds_half_sphere=*/false);
    for (Real value : field)
        EXPECT_NEAR(value, 1.0, 1e-12);
}
