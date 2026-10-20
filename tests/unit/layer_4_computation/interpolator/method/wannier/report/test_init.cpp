#include <gtest/gtest.h>

#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"

using namespace q3;

/**
 * The description is what the results file says the model is, so each of its
 * four outcomes is held here: a gauge that stalled must never read as
 * maximally localized, which is the one a driver acts on.
 */
TEST(WannierConstructionReport, TheProjectionGaugeOfAnIsolatedManifoldSaysSo) {
    const WannierConstructionReport report;
    EXPECT_EQ(report.description(), "wannier (projection gauge)");
}

TEST(WannierConstructionReport, ADisentangledSubspaceIsNamed) {
    WannierConstructionReport report;
    report.disentanglement.performed = true;
    EXPECT_EQ(report.description(), "wannier (projectability disentanglement)");
}

TEST(WannierConstructionReport, AStationaryGaugeIsCalledMaximallyLocalized) {
    WannierConstructionReport report;
    report.spread.localized = true;
    report.spread.localization_converged = true;
    EXPECT_EQ(report.description(), "wannier (projection gauge, maximally localized)");

    report.disentanglement.performed = true;
    EXPECT_EQ(report.description(),
              "wannier (projectability disentanglement, maximally localized)");
}

TEST(WannierConstructionReport, AGaugeThatStalledIsNeverCalledMaximallyLocalized) {
    WannierConstructionReport report;
    report.disentanglement.performed = true;
    report.spread.localized = true;
    report.spread.localization_converged = false;
    EXPECT_EQ(report.description(),
              "wannier (projectability disentanglement, localization stalled)");
}
