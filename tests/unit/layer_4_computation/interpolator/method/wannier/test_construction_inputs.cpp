#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "utils/wannier_chain_model.hpp"

using namespace q3;
using q3::test_utils::inputs_of;
using q3::test_utils::make_wannier_chain;
using q3::test_utils::stencil_of;
using q3::test_utils::WannierChainModel;

namespace {

/// A one-band record on a coarse cubic mesh with nothing but what the projection gauge needs.
WannierInputs bare_record(int mesh) {
    WannierInputs record;
    record.kpoint_grid_dimensions = {mesh, mesh, mesh};
    record.number_of_bands = 1;
    record.number_of_trial_orbitals = 1;
    for (int x = 0; x < mesh; x++) {
        for (int y = 0; y < mesh; y++) {
            for (int z = 0; z < mesh; z++) {
                record.kpoint_coordinates.emplace_back(static_cast<Real>(x) / mesh,
                                                       static_cast<Real>(y) / mesh,
                                                       static_cast<Real>(z) / mesh);
                record.eigenvalues_rydberg.push_back({0.0});
                record.projection_matrices.push_back({Complex(1.0, 0.0)});
            }
        }
    }
    return record;
}

} // namespace

TEST(WannierConstructionInputs, ABareRecordHasNeitherStencilNorOverlaps) {
    const WannierInputs record = bare_record(3);
    const WannierConstructionInputs inputs(record);
    EXPECT_EQ(inputs.number_of_kpoints(), 27);
    EXPECT_EQ(inputs.number_of_wannier_functions(), 1);
    EXPECT_FALSE(inputs.has_stencil());
    EXPECT_FALSE(inputs.has_overlaps());
    EXPECT_FALSE(inputs.spread_is_defined());
    EXPECT_FALSE(inputs.two_point_axis_direction().has_value());
    EXPECT_THROW(inputs.stencil(), std::logic_error);
    EXPECT_THROW(inputs.overlaps(), std::logic_error);
}

/**
 * The chain's own stencil is a neighbour table written by hand; the record
 * names a grid and two steps instead. Both must resolve to the same
 * neighbours, weights and b-vectors, or every test that runs a strategy on
 * the chain as a record would be measuring a different model.
 */
TEST(WannierConstructionInputs, TheChainAsARecordResolvesTheChainsOwnStencil) {
    const WannierChainModel model = make_wannier_chain(6, {0.0, 2.0}, 5.5, 0.9);
    const WannierInputs record = inputs_of(model);
    const WannierConstructionInputs inputs(record);
    ASSERT_TRUE(inputs.has_stencil());
    ASSERT_TRUE(inputs.has_overlaps());
    EXPECT_TRUE(inputs.spread_is_defined());

    const KPointStencil expected = stencil_of(model);
    const KPointStencil& resolved = inputs.stencil();
    ASSERT_EQ(resolved.number_of_kpoints(), expected.number_of_kpoints());
    ASSERT_EQ(resolved.number_of_directions(), expected.number_of_directions());
    for (int kpoint = 0; kpoint < expected.number_of_kpoints(); kpoint++) {
        for (int direction = 0; direction < expected.number_of_directions(); direction++) {
            EXPECT_EQ(resolved.neighbor(kpoint, direction), expected.neighbor(kpoint, direction))
                << "k-point " << kpoint << ", direction " << direction;
        }
    }
    for (int direction = 0; direction < expected.number_of_directions(); direction++) {
        EXPECT_DOUBLE_EQ(resolved.weight(direction), expected.weight(direction));
        for (int axis = 0; axis < 3; axis++) {
            EXPECT_DOUBLE_EQ(resolved.b_vector(direction)[axis],
                             expected.b_vector(direction)[axis]);
        }
    }
    const BlochOverlaps expected_overlaps = BlochOverlaps::square(model.overlaps, 2);
    EXPECT_EQ(inputs.overlaps().at(1, 0), expected_overlaps.at(1, 0));
    EXPECT_EQ(inputs.overlaps().at(5, 1), expected_overlaps.at(5, 1));
}

TEST(WannierConstructionInputs, ATwoPointAxisLeavesTheSpreadUndefined) {
    const WannierChainModel model = make_wannier_chain(2, {0.0, 2.0}, 5.5, 0.9);
    const WannierInputs record = inputs_of(model);
    const WannierConstructionInputs inputs(record);
    EXPECT_TRUE(inputs.has_overlaps());
    EXPECT_EQ(inputs.two_point_axis_direction(), std::optional<int>(0));
    EXPECT_FALSE(inputs.spread_is_defined());
}

TEST(WannierConstructionInputs, OverlapsWithoutDirectionsAreRefused) {
    WannierInputs record = inputs_of(make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9));
    record.neighbor_directions.clear();
    EXPECT_THROW(WannierConstructionInputs{record}, std::invalid_argument);
}

TEST(WannierConstructionInputs, AMalformedRecordIsRefusedBeforeAnyStrategyRuns) {
    const WannierInputs empty;
    EXPECT_THROW(WannierConstructionInputs{empty}, std::invalid_argument);

    WannierInputs without_grid = bare_record(2);
    without_grid.kpoint_grid_dimensions = {0, 0, 0};
    EXPECT_THROW(WannierConstructionInputs{without_grid}, std::invalid_argument);

    WannierInputs short_of_eigenvalues = bare_record(2);
    short_of_eigenvalues.eigenvalues_rydberg.pop_back();
    EXPECT_THROW(WannierConstructionInputs{short_of_eigenvalues}, std::invalid_argument);

    WannierInputs more_bands_than_eigenvalues = bare_record(2);
    more_bands_than_eigenvalues.number_of_bands = 2;
    EXPECT_THROW(WannierConstructionInputs{more_bands_than_eigenvalues}, std::invalid_argument);
}

/**
 * The model is a Fourier sum over the Wigner-Seitz supercell of the grid, exact
 * only over the whole grid, so a record holding another number of k-points
 * than its grid has is refused rather than transformed.
 */
TEST(WannierConstructionInputs, ARecordThatDoesNotCoverItsGridIsRefused) {
    WannierInputs record = bare_record(2);
    record.kpoint_grid_dimensions = {2, 2, 3};
    try {
        const WannierConstructionInputs inputs(record);
        FAIL() << "eight k-points cannot cover a 2x2x3 grid";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("8 k-points"), std::string::npos) << message;
        EXPECT_NE(message.find("2x2x3 grid, which has 12"), std::string::npos) << message;
    }
}

/// A projection of the wrong size is refused here, with its k-point named, not inside a strategy.
TEST(WannierConstructionInputs, AMisshapenProjectionIsRefusedWithItsKPointNamed) {
    WannierInputs record = bare_record(2);
    record.projection_matrices[5] = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
    try {
        const WannierConstructionInputs inputs(record);
        FAIL() << "a 1 x 1 projection cannot hold two elements";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("k-point 5"), std::string::npos) << error.what();
    }
}

/// The projections come back restricted to the kept bands, in the classification's order.
TEST(WannierConstructionInputs, TheProjectionsAreRestrictedToTheKeptBands) {
    const WannierChainModel model = make_wannier_chain(4, {0.0, 2.0, 1.0}, 5.5, 0.9);
    const WannierInputs record = inputs_of(model);
    const WannierConstructionInputs inputs(record);

    BandClassification classification = BandClassification::every_band_kept(4, 3);
    classification.kpoints[1].kept_bands = {2, 0};
    const std::vector<DenseMatrix<Complex>> restricted =
        inputs.projections_restricted_to(classification);

    ASSERT_EQ(restricted.size(), 4u);
    EXPECT_EQ(restricted[0], DenseMatrix<Complex>::identity(3));
    ASSERT_EQ(restricted[1].rows(), 2);
    ASSERT_EQ(restricted[1].columns(), 3);
    EXPECT_EQ(restricted[1](0, 2), Complex(1.0, 0.0)) << "the first kept band is band 2";
    EXPECT_EQ(restricted[1](1, 0), Complex(1.0, 0.0)) << "the second kept band is band 0";
    EXPECT_EQ(restricted[1](0, 0), Complex(0.0, 0.0));
}

/// In the identity gauge over every band, the overlaps in gauge are the record's own.
TEST(WannierConstructionInputs, TheOverlapsInTheIdentityGaugeAreTheRecordsOwn) {
    const WannierChainModel model = make_wannier_chain(4, {0.0, 2.0}, 5.5, 0.9);
    const WannierInputs record = inputs_of(model);
    const WannierConstructionInputs inputs(record);

    const BandClassification classification = BandClassification::every_band_kept(4, 2);
    const std::vector<DenseMatrix<Complex>> identity(4, DenseMatrix<Complex>::identity(2));
    const BlochOverlaps in_gauge = inputs.overlaps_in_gauge(classification, identity);

    for (int kpoint = 0; kpoint < 4; kpoint++) {
        for (int direction = 0; direction < 2; direction++) {
            EXPECT_EQ(in_gauge.at(kpoint, direction), inputs.overlaps().at(kpoint, direction));
        }
    }
}
