#include <gtest/gtest.h>

#include <complex>
#include <memory>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/factory.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/factory.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/gradient_corrected.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/null.hpp"

using namespace q3;

/**
 * The factory is the stress path's ONE consumer of
 * requires_density_gradient(); these tests pin its selection so that a
 * regression cannot silently drop the Nielsen-Martin gradient term from a GGA
 * stress (the diagonal term alone looks plausible, which is exactly why the
 * omission would be quiet). Construction only stores references, so
 * placeholder objects suffice — the numerical behavior is gated by the
 * bit-identity comparison recorded in the decomposition commit.
 */

namespace {

/// Density source stub: the factory never calls it, and these tests never evaluate.
class StubDensitySource : public ExchangeCorrelationStressDensitySource {
  public:
    [[nodiscard]] std::vector<Complex> build_total_charge_density_g() const override {
        return {};
    }
    [[nodiscard]] std::vector<Complex>
    build_spin_charge_density_g(int /*spin_index*/) const override {
        return {};
    }
};

} // namespace

TEST(ExchangeCorrelationGradientStressCalculatorFactory, LocalFunctionalResolvesToNull) {
    Crystal crystal_structure;
    PlanewaveBasis planewave_basis;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    StubDensitySource density_source;
    StressTensorContributions result;
    auto functional = XCFunctionalFactory::create("PZ", 1);
    ASSERT_FALSE(functional->requires_density_gradient());

    auto calculator = create_exchange_correlation_gradient_stress_calculator(
        *functional, crystal_structure, planewave_basis, scf_state, density_source, result);
    EXPECT_NE(dynamic_cast<NullExchangeCorrelationGradientStressCalculator*>(calculator.get()),
              nullptr);

    /**
     * The Null object's whole contract: contributing nothing. A stray write
     * here would corrupt every LDA stress tensor.
     */
    result.exchange_correlation[0][0] = 1.25;
    calculator->add_gradient_stress();
    EXPECT_EQ(result.exchange_correlation[0][0], 1.25);
    EXPECT_EQ(result.exchange_correlation[1][2], 0.0);
}

TEST(ExchangeCorrelationGradientStressCalculatorFactory,
     GradientCorrectedFunctionalResolvesToGradientCorrected) {
    Crystal crystal_structure;
    PlanewaveBasis planewave_basis;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    StubDensitySource density_source;
    StressTensorContributions result;
    auto functional = XCFunctionalFactory::create("PBE", 1);
    ASSERT_TRUE(functional->requires_density_gradient());

    auto calculator = create_exchange_correlation_gradient_stress_calculator(
        *functional, crystal_structure, planewave_basis, scf_state, density_source, result);
    EXPECT_NE(dynamic_cast<GradientCorrectedExchangeCorrelationGradientStressCalculator*>(
                  calculator.get()),
              nullptr);
    EXPECT_EQ(dynamic_cast<NullExchangeCorrelationGradientStressCalculator*>(calculator.get()),
              nullptr);
}
