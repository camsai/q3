/**
 * A2.1 gates — the radial Poisson solve against closed-form solutions.
 *
 * These need no PAW dataset and no SCF: every case here has an analytic
 * answer, which is the point. A one-center Hartree error is electron-volts in
 * size and still converges to a tidy number, so the solver is pinned against
 * mathematics before any density built from real partial waves reaches it.
 *
 * Rydberg units throughout (e² = 2): a point charge Z gives 2Z/r, not Z/r.
 */

#include "layer_4_computation/paw/one_center/radial_poisson.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace q3;
using namespace q3::paw_one_center;

namespace {

/// A logarithmic grid of the shape the datasets use: r_i = r_0 exp(i·h).
struct LogarithmicGrid {
    std::vector<Real> radius;
    std::vector<Real> measure; // dr/di = r·h
    int mesh_size{};

    LogarithmicGrid(Real first_radius, Real last_radius, int points) : mesh_size(points) {
        Real spacing = std::log(last_radius / first_radius) / static_cast<Real>(points - 1);
        radius.resize(points);
        measure.resize(points);
        for (int index = 0; index < points; index++) {
            radius[index] = first_radius * std::exp(spacing * index);
            measure[index] = radius[index] * spacing;
        }
    }
};

} // namespace

TEST(RadialPoissonTest, PointChargeGivesTwoZOverR) {
    /**
     * The Rydberg check. A compact spherical charge Z produces 2Z/r outside
     * itself; reading Z/r instead would mean the solver is in Hartree units,
     * a factor of two through every one-center energy.
     */
    LogarithmicGrid grid(1e-5, 40.0, 1501);
    const Real charge = 3.0;
    const Real width = 0.05;

    // A narrow normalized Gaussian: rho(r) = Z (1/(2 pi w^2))^{3/2} exp(-r^2/2w^2).
    std::vector<Real> density_r_squared(grid.mesh_size);
    Real normalization = charge / std::pow(2.0 * M_PI * width * width, 1.5);
    for (int index = 0; index < grid.mesh_size; index++) {
        Real radius = grid.radius[index];
        Real value = normalization * std::exp(-radius * radius / (2.0 * width * width));
        // rho_L=0 in the Y_LM convention carries a factor sqrt(4 pi).
        density_r_squared[index] = std::sqrt(4.0 * M_PI) * value * radius * radius;
    }

    std::vector<Real> potential;
    solve_radial_hartree(
        0, grid.radius, grid.measure, density_r_squared, grid.mesh_size, potential);

    // Compare V(r)·Y_00 against 2Z/r well outside the charge.
    const Real y_00 = 1.0 / std::sqrt(4.0 * M_PI);
    for (int index = 0; index < grid.mesh_size; index++) {
        Real radius = grid.radius[index];
        if (radius < 20.0 * width || radius > 20.0)
            continue;
        Real computed = potential[index] * y_00;
        Real expected = 2.0 * charge / radius;
        EXPECT_NEAR(computed, expected, 1e-9 * std::abs(expected)) << "at r = " << radius;
    }
}

TEST(RadialPoissonTest, GaussianChargeReproducesTheErrorFunctionSolution) {
    /**
     * Inside the charge the answer is not a pure Coulomb tail, so this checks
     * the inner integral as well: a normalized Gaussian of width w has
     * V(r) = 2Z·erf(r/(w√2))/r in Rydberg.
     */
    LogarithmicGrid grid(1e-5, 60.0, 2001);
    const Real charge = 1.0;
    const Real width = 0.8;

    std::vector<Real> density_r_squared(grid.mesh_size);
    Real normalization = charge / std::pow(2.0 * M_PI * width * width, 1.5);
    for (int index = 0; index < grid.mesh_size; index++) {
        Real radius = grid.radius[index];
        Real value = normalization * std::exp(-radius * radius / (2.0 * width * width));
        density_r_squared[index] = std::sqrt(4.0 * M_PI) * value * radius * radius;
    }

    std::vector<Real> potential;
    solve_radial_hartree(
        0, grid.radius, grid.measure, density_r_squared, grid.mesh_size, potential);

    const Real y_00 = 1.0 / std::sqrt(4.0 * M_PI);
    Real worst_relative = 0.0;
    for (int index = 0; index < grid.mesh_size; index++) {
        Real radius = grid.radius[index];
        if (radius < 1e-2 || radius > 30.0)
            continue;
        Real computed = potential[index] * y_00;
        Real expected = 2.0 * charge * std::erf(radius / (width * std::sqrt(2.0))) / radius;
        worst_relative =
            std::max(worst_relative, std::abs(computed - expected) / std::abs(expected));
    }
    EXPECT_LT(worst_relative, 1e-8);
}

TEST(RadialPoissonTest, HigherMultipolesMatchTheirClosedForm) {
    /**
     * L > 0 exercises both r^L prefactors and both integrals, and is where a
     * slip in the (2L+1) denominator or in either power of r would hide —
     * L = 0 is insensitive to all of it.
     *
     * For u_L(r) = r^(L+2) e^(-a r) both integrals are elementary:
     *
     *   inner(r) = ∫₀^r u_L r'^L dr'          = γ(2L+3, ar) / a^(2L+3)
     *   outer(r) = ∫_r^∞ u_L r'^(-(L+1)) dr'  = e^(-ar)(ar + 1) / a²
     *
     * with γ the lower incomplete gamma, a finite sum for integer first
     * argument. So the gate compares against mathematics rather than against
     * another discretization.
     */
    /**
     * The lower incomplete gamma, evaluated stably. The textbook identity
     * γ(n,x) = (n-1)!·[1 − e^(−x) Σ x^k/k!] is unusable here: for small x the
     * bracket is a difference of two nearly equal numbers whose true value is
     * ~x^n/n!, so at L = 3 it loses every significant digit and reports an
     * error of 800% against a solver that is in fact correct. Below x = n+1
     * the ascending series is used, which has no cancellation; above it the
     * complement is safe because γ approaches (n-1)! and the subtracted term
     * is small.
     */
    auto lower_incomplete_gamma_integer = [](int shape, Real argument) {
        Real factorial = 1.0;
        for (int step = 1; step <= shape - 1; step++)
            factorial *= step;

        if (argument < static_cast<Real>(shape) + 1.0) {
            Real denominator = static_cast<Real>(shape);
            Real term = 1.0 / denominator;
            Real sum = term;
            for (int iteration = 0; iteration < 500; iteration++) {
                denominator += 1.0;
                term *= argument / denominator;
                sum += term;
                if (std::abs(term) < std::abs(sum) * 1e-17)
                    break;
            }
            return sum * std::exp(-argument + shape * std::log(argument));
        }

        Real sum = 0.0;
        Real term = 1.0;
        for (int index = 0; index <= shape - 1; index++) {
            sum += term;
            term *= argument / static_cast<Real>(index + 1);
        }
        return factorial - factorial * std::exp(-argument) * sum;
    };

    for (int angular_momentum : {1, 2, 3, 4}) {
        LogarithmicGrid grid(1e-5, 50.0, 2001);
        const Real decay = 1.3;

        std::vector<Real> density_r_squared(grid.mesh_size);
        for (int index = 0; index < grid.mesh_size; index++) {
            Real radius = grid.radius[index];
            density_r_squared[index] =
                std::pow(radius, angular_momentum + 2) * std::exp(-decay * radius);
        }

        std::vector<Real> potential;
        solve_radial_hartree(angular_momentum,
                             grid.radius,
                             grid.measure,
                             density_r_squared,
                             grid.mesh_size,
                             potential);

        const Real prefactor = 8.0 * M_PI / static_cast<Real>(2 * angular_momentum + 1);
        Real worst_relative = 0.0;
        for (int index = 0; index < grid.mesh_size; index++) {
            Real radius = grid.radius[index];
            if (radius < 0.05 || radius > 20.0)
                continue;
            Real inner = lower_incomplete_gamma_integer(2 * angular_momentum + 3, decay * radius) /
                         std::pow(decay, 2 * angular_momentum + 3);
            Real outer = std::exp(-decay * radius) * (decay * radius + 1.0) / (decay * decay);
            Real expected = prefactor * (inner / std::pow(radius, angular_momentum + 1) +
                                         std::pow(radius, angular_momentum) * outer);
            worst_relative = std::max(worst_relative,
                                      std::abs(potential[index] - expected) / std::abs(expected));
        }
        /**
         * Measured relative error on this grid: 8e-9 (L=1), 3.0e-8 (L=2),
         * 8.0e-8 (L=3), 1.8e-7 (L=4) — quadrature error amplified by r^L, and
         * the reason the bound is not tighter. The order of the underlying
         * cumulative rule is pinned separately below.
         */
        EXPECT_LT(worst_relative, 1e-6) << "L = " << angular_momentum;
    }
}

TEST(RadialPoissonTest, HartreeEnergyMatchesTheGaussianSelfEnergy) {
    /**
     * Ties the potential to the energy expression, which is where the
     * double-counting one-half lives. A normalized Gaussian of width w has
     * self-energy E = 2Z²/(2w√π) in Rydberg.
     */
    LogarithmicGrid grid(1e-5, 60.0, 2001);
    const Real charge = 1.5;
    const Real width = 0.9;

    std::vector<Real> density_r_squared(grid.mesh_size);
    Real normalization = charge / std::pow(2.0 * M_PI * width * width, 1.5);
    for (int index = 0; index < grid.mesh_size; index++) {
        Real radius = grid.radius[index];
        Real value = normalization * std::exp(-radius * radius / (2.0 * width * width));
        density_r_squared[index] = std::sqrt(4.0 * M_PI) * value * radius * radius;
    }

    std::vector<Real> potential;
    solve_radial_hartree(
        0, grid.radius, grid.measure, density_r_squared, grid.mesh_size, potential);

    Real energy =
        radial_hartree_energy({potential}, {density_r_squared}, grid.measure, grid.mesh_size);
    Real expected = 2.0 * charge * charge / (2.0 * width * std::sqrt(M_PI));
    EXPECT_NEAR(energy, expected, 1e-7 * std::abs(expected));
}

TEST(RadialPoissonTest, CumulativeIntegralConvergesAtFourthOrder) {
    /**
     * The cumulative rule feeds every result above, and it uses a half-step
     * formula at odd points that a whole-interval Simpson test would never
     * exercise. Asserting the order catches a wrong half-step coefficient,
     * which a single tolerance on a smooth function would not.
     */
    auto worst_error_at = [](int points) {
        LogarithmicGrid grid(1e-4, 10.0, points);
        std::vector<Real> values(grid.mesh_size);
        for (int index = 0; index < grid.mesh_size; index++)
            values[index] = std::exp(-grid.radius[index]);
        auto cumulative = cumulative_integral(values, grid.measure, grid.mesh_size);

        Real worst = 0.0;
        for (int index = 0; index < grid.mesh_size; index++) {
            Real radius = grid.radius[index];
            // integral of exp(-r) from r_0 to r
            Real expected = std::exp(-grid.radius[0]) - std::exp(-radius);
            worst = std::max(worst, std::abs(cumulative[index] - expected));
        }
        return worst;
    };

    Real coarse = worst_error_at(401);
    Real fine = worst_error_at(801);
    EXPECT_LT(coarse, 1e-6);
    EXPECT_GT(coarse / fine, 8.0) << "coarse " << coarse << " fine " << fine;
}
