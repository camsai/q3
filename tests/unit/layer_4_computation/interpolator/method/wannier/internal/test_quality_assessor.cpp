#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/quality_assessor.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/quality_report.hpp"

using namespace q3;

namespace {

Matrix3x3 cubic_cell(Real edge) {
    Matrix3x3 lattice;
    lattice[0] = {edge, 0.0, 0.0};
    lattice[1] = {0.0, edge, 0.0};
    lattice[2] = {0.0, 0.0, edge};
    return lattice;
}

/// A construction whose final gauge has these spreads and centres.
WannierConstructionReport construction_with(std::vector<Real> spreads_bohr_squared,
                                            std::vector<Vector3D> centers_bohr = {}) {
    WannierConstructionReport construction;
    construction.spread.spreads_bohr_squared = std::move(spreads_bohr_squared);
    construction.spread.centers_bohr = std::move(centers_bohr);
    return construction;
}

} // namespace

TEST(WannierQualityAssessor, TheSpreadStatisticsFlagTheOutlierAndTheLargeFunction) {
    const WannierQualityAssessor assessor(cubic_cell(10.0), {});
    const WannierQualityReport report = assessor.assess(construction_with({1.0, 1.0, 1.0, 10.0}));

    EXPECT_NEAR(report.spread_median_bohr_squared, 1.0, 1.0e-12);
    EXPECT_NEAR(report.spread_maximum_bohr_squared, 10.0, 1.0e-12);
    // mean 3.25; deviations 2.25 (three times) and 6.75: (3 * 5.0625 + 45.5625) / 4.
    EXPECT_NEAR(report.spread_variance_bohr_squared, 15.1875, 1.0e-12);
    EXPECT_EQ(report.spread_outliers, (std::vector<int>{3}));
    EXPECT_TRUE(report.spreads_above_ten_angstrom_squared.empty())
        << "10 bohr^2 is 2.8 angstrom^2, well under the large-function line";
    EXPECT_TRUE(report.nearest_atom_distance_bohr.empty()) << "no atoms, no distances";

    const WannierQualityReport large = assessor.assess(construction_with({40.0}));
    EXPECT_TRUE(large.spread_outliers.empty()) << "a single function is its own median";
    EXPECT_EQ(large.spreads_above_ten_angstrom_squared, (std::vector<int>{0}));

    const WannierQualityReport empty = assessor.assess(construction_with({}));
    EXPECT_TRUE(empty.spreads_bohr_squared.empty());
    EXPECT_EQ(empty.spread_median_bohr_squared, 0.0);
}

TEST(WannierQualityAssessor, CentresAreMeasuredAgainstAtomsAndBondMidpointsOverPeriodicImages) {
    /**
     * Two atoms on the x axis of a 10 bohr cube, 5 bohr apart -- also 5 bohr
     * apart through the boundary, so both are bonds and the midpoints sit at
     * 2.5 and 7.5. A centre at 2.6 is on the first bond; one at 9.9 is 0.1
     * from the image of the first atom, not 9.9 from the atom itself.
     */
    const WannierQualityAssessor assessor(cubic_cell(10.0), {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}});
    const WannierQualityReport report = assessor.assess(
        construction_with({1.0, 1.0, 1.0}, {{2.6, 0.0, 0.0}, {9.9, 0.0, 0.0}, {5.0, 5.0, 0.0}}));

    EXPECT_NEAR(report.shortest_bond_bohr, 5.0, 1.0e-12);
    ASSERT_EQ(report.nearest_atom_distance_bohr.size(), 3u);
    EXPECT_NEAR(report.nearest_atom_distance_bohr[0], 2.4, 1.0e-12);
    EXPECT_NEAR(report.nearest_atom_distance_bohr[1], 0.1, 1.0e-12);
    EXPECT_NEAR(report.nearest_atom_distance_bohr[2], 5.0, 1.0e-12);
    EXPECT_NEAR(report.nearest_bond_midpoint_distance_bohr[0], 0.1, 1.0e-12);
    EXPECT_NEAR(report.nearest_bond_midpoint_distance_bohr[1], 2.4, 1.0e-12);
    EXPECT_NEAR(
        report.nearest_bond_midpoint_distance_bohr[2], std::sqrt(2.5 * 2.5 + 25.0), 1.0e-12);
}

/// The decay is measured by the builder and carried through unchanged.
TEST(WannierQualityAssessor, TheDecayOfTheConstructionIsCarriedIntoTheReport) {
    WannierConstructionReport construction;
    construction.hamiltonian_decay.outermost_shell_size = 12;
    construction.hamiltonian_decay.ratio = 0.01;

    const WannierQualityReport report =
        WannierQualityAssessor(cubic_cell(10.0), {}).assess(construction);
    EXPECT_EQ(report.decay.outermost_shell_size, 12);
    EXPECT_DOUBLE_EQ(report.decay.ratio, 0.01);
}

/// One atom has no partner but its own images, and those are its bonds.
TEST(WannierQualityAssessor, ASingleAtomIsBondedToItsOwnImages) {
    const WannierQualityAssessor assessor(cubic_cell(4.0), {{0.0, 0.0, 0.0}});
    const WannierQualityReport report =
        assessor.assess(construction_with({1.0}, {{2.0, 0.0, 0.0}}));

    EXPECT_NEAR(report.shortest_bond_bohr, 4.0, 1.0e-12);
    EXPECT_NEAR(report.nearest_atom_distance_bohr[0], 2.0, 1.0e-12);
    EXPECT_NEAR(report.nearest_bond_midpoint_distance_bohr[0], 0.0, 1.0e-12)
        << "the centre sits on the midpoint of the atom and its image along x";
}
