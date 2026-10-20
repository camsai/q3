#include <gtest/gtest.h>

#include <string>

#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"

using namespace q3;

/**
 * The names are the enumerators' own spelling: the utility prints them on
 * the `diagnostic:` line and the Python side's enum repeats them, so a
 * rename here without a rename there is what this pins.
 */
TEST(WannierDiagnostic, EveryCodeIsNamedByItsOwnSpelling) {
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::disentanglement_unconverged),
              "disentanglement_unconverged");
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::disentanglement_unsupported),
              "disentanglement_unsupported");
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::frozen_window_overfull),
              "frozen_window_overfull");
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::kept_bands_underfull),
              "kept_bands_underfull");
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::gauge_singular), "gauge_singular");
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::gauge_collapsed), "gauge_collapsed");
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::overlaps_not_orthonormal),
              "overlaps_not_orthonormal");
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::localization_unsupported),
              "localization_unsupported");
    EXPECT_EQ(wannier_diagnostic_name(WannierDiagnostic::two_point_axis), "two_point_axis");
}
