#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "layer_0_core/diagnostic_error.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"

using namespace q3;

TEST(WannierDiagnosticError, CarriesTheCodeBesideTheProseAndIsARuntimeError) {
    static_assert(std::is_base_of_v<std::runtime_error, WannierDiagnosticError>,
                  "a catch of std::runtime_error must still see the prose");
    const WannierDiagnosticError error(WannierDiagnostic::two_point_axis, "the k-mesh is 2x2x2");
    EXPECT_EQ(error.diagnostic(), WannierDiagnostic::two_point_axis);
    EXPECT_STREQ(error.what(), "the k-mesh is 2x2x2");
    EXPECT_TRUE(error.spreads_bohr_squared().empty());
    EXPECT_EQ(error.spreads_text(), "");
}

/// The line the utility prints after the prose uses the prose's own six-decimal form.
TEST(WannierDiagnosticError, AnUnconvergedDisentanglementCarriesItsLastSpreads) {
    const WannierDiagnosticError error(WannierDiagnostic::disentanglement_unconverged,
                                       "disentanglement did not converge in 2 iterations",
                                       {32.869717, 32.869701, 32.869684});
    ASSERT_EQ(error.spreads_bohr_squared().size(), 3u);
    EXPECT_DOUBLE_EQ(error.spreads_bohr_squared()[2], 32.869684);
    EXPECT_EQ(error.spreads_text(), "32.869717, 32.869701, 32.869684");
}

/**
 * What the utility prints after the prose, which is all the entry point knows
 * of this failure: the code on the `diagnostic:` line, and the spreads on a
 * line of their own only when the failure has any.
 */
TEST(WannierDiagnosticError, ItsDiagnosticLinesAreTheCodeAndThenTheSpreads) {
    static_assert(std::is_base_of_v<DiagnosticError, WannierDiagnosticError>,
                  "the entry point catches the base class and names no method");
    const WannierDiagnosticError refused(WannierDiagnostic::two_point_axis, "the k-mesh is 2x2x2");
    EXPECT_EQ(refused.diagnostic_lines(), (std::vector<std::string>{"diagnostic: two_point_axis"}));

    const WannierDiagnosticError unconverged(WannierDiagnostic::disentanglement_unconverged,
                                             "disentanglement did not converge in 2 iterations",
                                             {32.869717, 32.869701});
    EXPECT_EQ(unconverged.diagnostic_lines(),
              (std::vector<std::string>{"diagnostic: disentanglement_unconverged",
                                        "diagnostic_spreads_bohr_squared: 32.869717, 32.869701"}));
}
