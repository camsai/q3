#include <gtest/gtest.h>

#include <memory>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/factory.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/factory.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/gradient_corrected.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/local_density.hpp"

using namespace q3;

/**
 * The factory is the ONE place allowed to consult
 * requires_density_gradient(); these tests pin its selection so that a
 * regression cannot silently route a GGA through the local-density evaluation
 * (which would drop the gradient terms without any error). Construction only
 * stores references, so placeholder basis/crystal objects suffice — the
 * numerical behavior of both strategies is gated end-to-end by the
 * bit-identity comparison recorded in the decomposition commit and by every
 * PBE/PZ validation run.
 */

TEST(ExchangeCorrelationPotentialCalculatorFactory, LocalFunctionalResolvesToLocalDensity) {
    PlanewaveBasis planewave_basis;
    Crystal crystal_structure;
    auto functional = XCFunctionalFactory::create("PZ", 1);
    ASSERT_FALSE(functional->requires_density_gradient());

    auto calculator = create_exchange_correlation_potential_calculator(
        planewave_basis, crystal_structure, *functional);
    EXPECT_NE(dynamic_cast<LocalDensityExchangeCorrelationPotentialCalculator*>(calculator.get()),
              nullptr);
    EXPECT_EQ(
        dynamic_cast<GradientCorrectedExchangeCorrelationPotentialCalculator*>(calculator.get()),
        nullptr);
}

TEST(ExchangeCorrelationPotentialCalculatorFactory,
     GradientCorrectedFunctionalResolvesToGradientCorrected) {
    PlanewaveBasis planewave_basis;
    Crystal crystal_structure;
    auto functional = XCFunctionalFactory::create("PBE", 1);
    ASSERT_TRUE(functional->requires_density_gradient());

    auto calculator = create_exchange_correlation_potential_calculator(
        planewave_basis, crystal_structure, *functional);
    EXPECT_NE(
        dynamic_cast<GradientCorrectedExchangeCorrelationPotentialCalculator*>(calculator.get()),
        nullptr);
    EXPECT_EQ(dynamic_cast<LocalDensityExchangeCorrelationPotentialCalculator*>(calculator.get()),
              nullptr);
}

TEST(ExchangeCorrelationPotentialCalculatorFactory, SpinPolarizedSelectionMatchesUnpolarized) {
    /**
     * The selection depends on the functional family alone, never on the spin
     * count: an LSDA PBE run needs the gradient pipeline exactly as much as an
     * unpolarized one.
     */
    PlanewaveBasis planewave_basis;
    Crystal crystal_structure;
    auto polarized_gga = XCFunctionalFactory::create("PBE", 2);
    auto polarized_lda = XCFunctionalFactory::create("PZ", 2);

    EXPECT_NE(dynamic_cast<GradientCorrectedExchangeCorrelationPotentialCalculator*>(
                  create_exchange_correlation_potential_calculator(
                      planewave_basis, crystal_structure, *polarized_gga)
                      .get()),
              nullptr);
    EXPECT_NE(dynamic_cast<LocalDensityExchangeCorrelationPotentialCalculator*>(
                  create_exchange_correlation_potential_calculator(
                      planewave_basis, crystal_structure, *polarized_lda)
                      .get()),
              nullptr);
}
