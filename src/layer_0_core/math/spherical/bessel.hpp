#pragma once

/**
 * Spherical Bessel functions j_l(x).
 *
 * Two branches, split at x = l + 1, because neither is accurate over the whole
 * range: the ascending series converges quickly for small x but suffers
 * cancellation once x grows past l, and the upward recurrence — along with the
 * closed forms it starts from — is stable for large x but subtracts nearly
 * equal quantities below x ≈ l, where it loses every significant digit and can
 * return the wrong sign outright.
 */

#include <cmath>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Spherical Bessel function j_l(x).
 * Ascending series for x < l+1, closed forms for l = 0..2 above that, and
 * upward recurrence for l > 2.
 *
 * @param degree    Order l of the Bessel function
 * @param argument  The argument x
 * @return j_l(x)
 */
inline Real spherical_bessel(int degree, Real argument) {
    if (std::abs(argument) < math::tolerances::bessel_argument_zero_threshold) {
        return (degree == 0) ? 1.0 : 0.0;
    }
    /**
     * Ascending series, summed to convergence.
     *
     *     j_l(x) = x^l/(2l+1)!! · Σ_k (−x²/2)^k / (k! (2l+3)(2l+5)···(2l+2k+1))
     *
     * Used whenever x < l + 1, which is not a matter of taste: the upward
     * recurrence below subtracts two nearly equal quantities once x drops under
     * l, and by l = 5, x = 0.06 it returns the wrong sign. The series has no
     * such cancellation there — its terms fall off like (x²/2)^k/k! — so this
     * branch is the only correct one in that region, and the recurrence keeps
     * the region where it is the accurate one.
     *
     * Summing to convergence rather than to a fixed number of terms is what
     * lets the branch extend that far; a four-term truncation is only good to
     * x ≈ 0.05.
     */
    if (std::abs(argument) < static_cast<Real>(degree) + 1.0) {
        Real argument_to_degree_power = (degree == 0) ? 1.0 : std::pow(argument, degree);
        double double_factorial_value = 1.0;
        for (int index = 2 * degree + 1; index >= 1; index -= 2) {
            double_factorial_value *= index;
        }
        Real half_argument_squared = 0.5 * argument * argument;
        Real term = 1.0;
        Real sum = 1.0;
        for (int index = 1; index <= 60; index++) {
            term *= -half_argument_squared / (index * (2.0 * degree + 2.0 * index + 1.0));
            sum += term;
            if (std::abs(term) < 1e-18 * std::abs(sum))
                break;
        }
        return (argument_to_degree_power / double_factorial_value) * sum;
    }
    switch (degree) {
        case 0:
            return std::sin(argument) / argument;
        case 1:
            return (std::sin(argument) / argument - std::cos(argument)) / argument;
        case 2:
            return ((3.0 / (argument * argument) - 1.0) * std::sin(argument) -
                    3.0 * std::cos(argument) / argument) /
                   argument;
        default: {
            Real bessel_minus_one = std::sin(argument) / argument;
            Real bessel_current = (std::sin(argument) / argument - std::cos(argument)) / argument;
            for (int index = 2; index <= degree; index++) {
                Real bessel_plus_one =
                    ((2 * index - 1) * bessel_current / argument) - bessel_minus_one;
                bessel_minus_one = bessel_current;
                bessel_current = bessel_plus_one;
            }
            return bessel_current;
        }
    }
}

/**
 * Batch-compute spherical Bessel function j_l(q * r[i]) for all radial grid points.
 * Whole-array because that is what the caller needs: the radial transform
 * multiplies j_l(q*r) into an integrand and hands the array to Simpson's
 * rule, which integrates arrays, not scalars. With l fixed across the mesh
 * the branch on l is loop-invariant, so it is hoisted out of the point loop
 * -- which is the whole difference between this routine and a loop calling
 * the scalar one, and the reason it exists.
 *
 * @param degree          Angular momentum l
 * @param q_value         The q-point value
 * @param radial_grid     Array of r values
 * @param mesh_size       Number of radial grid points to compute
 * @param result          [out] Pre-allocated array of size >= mesh_size to store j_l(q*r[i])
 */
inline void spherical_bessel_batch(
    int degree, Real q_value, const Real* radial_grid, int mesh_size, Real* result) {
    if (std::abs(q_value) < math::tolerances::bessel_argument_zero_threshold) {
        Real value = (degree == 0) ? 1.0 : 0.0;
        for (int index = 0; index < mesh_size; index++) {
            result[index] = value;
        }
        return;
    }

    /**
     * Compute q*r for all points, then evaluate j_l with the closed form for
     * l = 0..2 and upward recurrence from j_0, j_1 above that — each only in
     * the range where it is the accurate branch, deferring to the scalar
     * routine's series below.
     */

    switch (degree) {
        case 0:
            for (int index = 0; index < mesh_size; index++) {
                Real argument = q_value * radial_grid[index];
                if (std::abs(argument) < 1.0) {
                    result[index] = spherical_bessel(degree, argument);
                } else {
                    result[index] = std::sin(argument) / argument;
                }
            }
            break;

        case 1:
            for (int index = 0; index < mesh_size; index++) {
                Real argument = q_value * radial_grid[index];
                if (std::abs(argument) < 1.0) {
                    result[index] = spherical_bessel(degree, argument);
                } else {
                    result[index] = (std::sin(argument) / argument - std::cos(argument)) / argument;
                }
            }
            break;

        case 2:
            for (int index = 0; index < mesh_size; index++) {
                Real argument = q_value * radial_grid[index];
                /**
                 * (3/x² − 1)sin(x)/x − 3cos(x)/x² differences two quantities of
                 * order 3/x², so below x ≈ 1 it loses the digits the answer is
                 * made of; the scalar routine's series is exact there.
                 */
                if (std::abs(argument) < 1.0) {
                    result[index] = spherical_bessel(degree, argument);
                } else {
                    result[index] = ((3.0 / (argument * argument) - 1.0) * std::sin(argument) -
                                     3.0 * std::cos(argument) / argument) /
                                    argument;
                }
            }
            break;

        default:
            // General case: compute j_0 and j_1 for all points, then use recurrence
            for (int index = 0; index < mesh_size; index++) {
                Real argument = q_value * radial_grid[index];
                /**
                 * Below x = l + 1 the recurrence cancels catastrophically and
                 * the scalar routine's converged series is the accurate branch
                 * — the same crossover, for the same reason. Deferring to it
                 * keeps one definition of where that boundary sits rather than
                 * two that can drift apart.
                 */
                if (std::abs(argument) < static_cast<Real>(degree) + 1.0) {
                    result[index] = spherical_bessel(degree, argument);
                } else {
                    Real bessel_minus_one = std::sin(argument) / argument;
                    Real bessel_current =
                        (std::sin(argument) / argument - std::cos(argument)) / argument;
                    for (int recurrence_index = 2; recurrence_index <= degree; recurrence_index++) {
                        Real bessel_plus_one =
                            ((2 * recurrence_index - 1) * bessel_current / argument) -
                            bessel_minus_one;
                        bessel_minus_one = bessel_current;
                        bessel_current = bessel_plus_one;
                    }
                    result[index] = bessel_current;
                }
            }
            break;
    }
}

} // namespace q3
