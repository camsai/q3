#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/factory.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/isolated_manifold_selection.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/projectability_disentanglement_selection.hpp"

using namespace q3;

TEST(SubspaceSelectionFactory, TheDefaultSettingsTakeTheWindowAsItStands) {
    const std::unique_ptr<SubspaceSelection> selection =
        create_subspace_selection(WannierConstructionSettings{});
    EXPECT_NE(dynamic_cast<const IsolatedManifoldSelection*>(selection.get()), nullptr);
}

TEST(SubspaceSelectionFactory, AskingForDisentanglementSelectsTheClassifier) {
    WannierConstructionSettings settings;
    settings.disentanglement = ProjectabilityDisentanglementSettings{};
    const std::unique_ptr<SubspaceSelection> selection = create_subspace_selection(settings);
    EXPECT_NE(dynamic_cast<const ProjectabilityDisentanglementSelection*>(selection.get()),
              nullptr);
}

/**
 * Thresholds that describe no rule are refused when the strategy is made, not at the first
 * k-point.
 */
TEST(SubspaceSelectionFactory, InvertedThresholdsAreRefusedAtConstruction) {
    WannierConstructionSettings settings;
    ProjectabilityDisentanglementSettings disentanglement;
    disentanglement.thresholds.projectability_minimum = 0.9;
    disentanglement.thresholds.projectability_maximum = 0.1;
    settings.disentanglement = disentanglement;
    EXPECT_THROW(create_subspace_selection(settings), std::invalid_argument);
}
