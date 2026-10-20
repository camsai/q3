#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "layer_4_computation/interpolator/target/quantum_state/local_orbital.hpp"
#include "layer_5_properties/density_of_states/interpolated/init.hpp"
#include "utils/cubic_tight_binding.hpp"

using namespace q3;

using namespace q3::test_support;

namespace {

/// Riemann sum over the uniform energy grid the calculator emits.
Real integrate(const InterpolatedDensityOfStates& density) {
    Real energy_step = density.energies_rydberg[1] - density.energies_rydberg[0];
    return std::accumulate(
               density.density_of_states.begin(), density.density_of_states.end(), 0.0) *
           energy_step;
}

} // namespace

TEST(InterpolatedDensityOfStatesCalculator, IntegratesToTheBandCount) {
    /**
     * The normalisation check that makes a DOS comparable to anything: with a
     * unit-area Gaussian per state and a 1/N_k prefactor, integrating over
     * energy must return the number of bands, independent of broadening and
     * mesh. If this drifts, every DOS this class produces is off by a scale
     * factor that no amount of eyeballing the shape would reveal.
     */
    LocalOrbitalQuantumStateInterpolator interpolator = build_cubic_interpolator(3, -0.25, 0.1);
    InterpolatedDensityOfStatesCalculator calculator(interpolator);
    for (Real broadening : {0.005, 0.02}) {
        InterpolatedDensityOfStates density = calculator.compute(8, 8, 8, broadening, 4000);
        EXPECT_NEAR(integrate(density), 1.0, 2e-3) << "broadening " << broadening;
    }
}

TEST(InterpolatedDensityOfStatesCalculator, IntegratesToTheBandCountAtLargeBroadening) {
    /**
     * With a fixed 0.05 Ry window margin, a broadening of 0.05 Ry clipped the
     * Gaussian tails of the band-edge states and the integral came out at
     * 0.9983 of the band count. The margin now grows with the broadening's
     * own cutoff, so the invariant holds at any broadening.
     */
    LocalOrbitalQuantumStateInterpolator interpolator = build_cubic_interpolator(3, -0.25, 0.1);
    InterpolatedDensityOfStatesCalculator calculator(interpolator);
    for (Real broadening : {0.05, 0.1}) {
        InterpolatedDensityOfStates density = calculator.compute(8, 8, 8, broadening, 4000);
        EXPECT_NEAR(integrate(density), 1.0, 2e-3) << "broadening " << broadening;
        const Real span = density.energies_rydberg.back() - density.energies_rydberg.front();
        EXPECT_GT(span, 2.0 * 8.0 * broadening) << "window narrower than the Gaussian cutoff";
    }
}

TEST(InterpolatedDensityOfStatesCalculator, SpansTheBandwidthAndIsNonNegative) {
    const Real onsite = -0.25;
    const Real hopping = 0.1;
    LocalOrbitalQuantumStateInterpolator interpolator =
        build_cubic_interpolator(3, onsite, hopping);
    InterpolatedDensityOfStatesCalculator calculator(interpolator);
    InterpolatedDensityOfStates density = calculator.compute(10, 10, 10, 0.01);

    // The cubic band runs over onsite +- 6|t|.
    const Real lower_edge = onsite - 6.0 * std::abs(hopping);
    const Real upper_edge = onsite + 6.0 * std::abs(hopping);
    EXPECT_LT(density.energies_rydberg.front(), lower_edge);
    EXPECT_GT(density.energies_rydberg.back(), upper_edge);
    for (Real value : density.density_of_states) {
        EXPECT_GE(value, 0.0);
    }

    // Weight must actually sit inside the band, not only in the tails.
    Real inside = 0.0;
    for (std::size_t index = 0; index < density.energies_rydberg.size(); index++) {
        Real energy = density.energies_rydberg[index];
        if (energy > lower_edge && energy < upper_edge) {
            inside += density.density_of_states[index];
        }
    }
    Real total =
        std::accumulate(density.density_of_states.begin(), density.density_of_states.end(), 0.0);
    EXPECT_GT(inside / total, 0.95);
}

TEST(InterpolatedDensityOfStatesCalculator, RejectsInvalidParameters) {
    LocalOrbitalQuantumStateInterpolator interpolator = build_cubic_interpolator(3, -0.25, 0.1);
    InterpolatedDensityOfStatesCalculator calculator(interpolator);
    EXPECT_THROW(calculator.compute(0, 4, 4, 0.01), std::invalid_argument);
    EXPECT_THROW(calculator.compute(4, 4, 4, 0.0), std::invalid_argument);
    EXPECT_THROW(calculator.compute(4, 4, 4, 0.01, 1), std::invalid_argument);
}
