#include <gtest/gtest.h>

#include <stdexcept>

#include "layer_0_core/types/constants.hpp"
#include "layer_4_computation/interpolator/input_data/specification.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"

using namespace q3;

TEST(WannierConstructionSettings, TheDisentanglementBudgetAndToleranceReachTheIteration) {
    InterpolationSpecification specification;
    specification.method = "wannier";
    specification.disentanglement = "projectability";
    const WannierConstructionSettings defaults = WannierConstructionSettings::from(specification);
    ASSERT_TRUE(defaults.disentangles());
    EXPECT_EQ(defaults.disentanglement->iteration.maximum_iterations,
              DisentanglementSettings{}.maximum_iterations);
    EXPECT_DOUBLE_EQ(defaults.disentanglement->iteration.convergence_tolerance,
                     DisentanglementSettings{}.convergence_tolerance);

    specification.disentanglement_maximum_iterations = 16000;
    specification.disentanglement_convergence_tolerance = 1.0e-7;
    const WannierConstructionSettings raised = WannierConstructionSettings::from(specification);
    ASSERT_TRUE(raised.disentangles());
    EXPECT_EQ(raised.disentanglement->iteration.maximum_iterations, 16000);
    EXPECT_DOUBLE_EQ(raised.disentanglement->iteration.convergence_tolerance, 1.0e-7);
}

/**
 * The manifest's knobs are mapped in one place. The window is given in
 * electronvolts and the classifier reads rydberg; the localisation words map
 * onto the policy in the order none, auto, maximal; and a manifest that says
 * nothing at all asks for an isolated manifold, localised where possible.
 */
TEST(WannierConstructionSettings, TheManifestMapsOntoTheConstructionSettingsOnce) {
    InterpolationSpecification specification;
    specification.method = "wannier";
    const WannierConstructionSettings plain = WannierConstructionSettings::from(specification);
    EXPECT_FALSE(plain.disentangles());
    EXPECT_EQ(plain.localization, LocalizationPolicy::Preferred);
    EXPECT_EQ(plain.spread_minimization.maximum_iterations,
              SpreadMinimizationSettings{}.maximum_iterations);

    specification.localization = "none";
    EXPECT_EQ(WannierConstructionSettings::from(specification).localization,
              LocalizationPolicy::None);

    specification.localization = "maximal";
    specification.localization_maximum_iterations = 6000;
    const WannierConstructionSettings demanded = WannierConstructionSettings::from(specification);
    EXPECT_EQ(demanded.localization, LocalizationPolicy::Required);
    EXPECT_EQ(demanded.spread_minimization.maximum_iterations, 6000);

    specification.disentanglement = "projectability";
    specification.projectability_minimum = 0.02;
    specification.projectability_maximum = 0.9;
    specification.frozen_window_is_automatic = false;
    specification.frozen_window_above_fermi_ev = 2.0 * constants::RY_TO_EV;
    const WannierConstructionSettings disentangled =
        WannierConstructionSettings::from(specification);
    ASSERT_TRUE(disentangled.disentangles());
    const BandClassificationThresholds& thresholds = disentangled.disentanglement->thresholds;
    EXPECT_DOUBLE_EQ(thresholds.projectability_minimum, 0.02);
    EXPECT_DOUBLE_EQ(thresholds.projectability_maximum, 0.9);
    EXPECT_FALSE(thresholds.automatic_frozen_window);
    EXPECT_NEAR(thresholds.frozen_window_above_fermi_rydberg, 2.0, 1.0e-12);
}

/// The defaults the manifest documents are the settings structs' own, not a second copy of them.
TEST(WannierConstructionSettings, TheSpecificationDefaultsAreTheSettingsOwn) {
    const InterpolationSpecification specification;
    EXPECT_DOUBLE_EQ(specification.projectability_minimum,
                     BandClassificationThresholds{}.projectability_minimum);
    EXPECT_DOUBLE_EQ(specification.projectability_maximum,
                     BandClassificationThresholds{}.projectability_maximum);
    EXPECT_EQ(specification.localization_maximum_iterations,
              SpreadMinimizationSettings{}.maximum_iterations);
    EXPECT_EQ(specification.disentanglement_maximum_iterations,
              DisentanglementSettings{}.maximum_iterations);
    EXPECT_DOUBLE_EQ(specification.disentanglement_convergence_tolerance,
                     DisentanglementSettings{}.convergence_tolerance);
}

TEST(WannierConstructionSettings, AnUnknownLocalizationWordIsRefused) {
    EXPECT_EQ(localization_policy_from("none"), LocalizationPolicy::None);
    EXPECT_EQ(localization_policy_from("auto"), LocalizationPolicy::Preferred);
    EXPECT_EQ(localization_policy_from("maximal"), LocalizationPolicy::Required);
    EXPECT_THROW((void)localization_policy_from("always"), std::invalid_argument);
}
