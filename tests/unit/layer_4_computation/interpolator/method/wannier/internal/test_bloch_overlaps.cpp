#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/type_aliases.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"

using namespace q3;

namespace {

/// Two k-points, one direction each way, 3 x 3 blocks whose entries encode (k, row, column).
std::vector<std::vector<std::vector<Complex>>> two_point_blocks() {
    std::vector<std::vector<std::vector<Complex>>> blocks(2);
    for (int kpoint = 0; kpoint < 2; kpoint++) {
        for (int direction = 0; direction < 2; direction++) {
            std::vector<Complex> block(9);
            for (int column = 0; column < 3; column++) {
                for (int row = 0; row < 3; row++) {
                    block[static_cast<std::size_t>(column) * 3 + row] =
                        Complex(100.0 * kpoint + 10.0 * row + column, direction);
                }
            }
            blocks[static_cast<std::size_t>(kpoint)].push_back(block);
        }
    }
    return blocks;
}

KPointStencil ring_of_two() {
    return KPointStencil::from_table({{1, 1}, {0, 0}}, {0.5, 0.5});
}

} // namespace

TEST(BlochOverlaps, ReadsEachRecordBlockColumnMajor) {
    const BlochOverlaps overlaps = BlochOverlaps::square(two_point_blocks(), 3);
    EXPECT_EQ(overlaps.number_of_kpoints(), 2);
    EXPECT_EQ(overlaps.number_of_directions(), 2);
    EXPECT_EQ(overlaps.at(1, 0)(2, 1), Complex(121.0, 0.0));
    EXPECT_EQ(overlaps.at(0, 1)(0, 2), Complex(2.0, 1.0));
}

TEST(BlochOverlaps, RefusesABlockOfTheWrongSizeOrARaggedRecord) {
    std::vector<std::vector<std::vector<Complex>>> blocks = two_point_blocks();
    blocks[1][0].pop_back();
    EXPECT_THROW(BlochOverlaps::square(blocks, 3), std::invalid_argument);
    blocks = two_point_blocks();
    blocks[1].pop_back();
    EXPECT_THROW(BlochOverlaps::square(blocks, 3), std::invalid_argument);
    EXPECT_THROW(BlochOverlaps::square({}, 3), std::invalid_argument);
}

TEST(BlochOverlaps, RestrictsRowsToThisKPointsBandsAndColumnsToTheNeighbours) {
    const BlochOverlaps overlaps = BlochOverlaps::square(two_point_blocks(), 3);
    std::vector<KPointBandClassification> classification(2);
    classification[0].kept_bands = {1, 2};
    classification[1].kept_bands = {0, 2};
    const BlochOverlaps restricted = overlaps.restricted_to(classification, ring_of_two());

    // At k = 0 the rows are bands {1, 2}; its neighbour is k = 1, whose columns are {0, 2}.
    const DenseMatrix<Complex>& block = restricted.at(0, 0);
    ASSERT_EQ(block.rows(), 2);
    ASSERT_EQ(block.columns(), 2);
    EXPECT_EQ(block(0, 0), Complex(10.0, 0.0));
    EXPECT_EQ(block(1, 1), Complex(22.0, 0.0));
    // At k = 1 the rows are {0, 2} and the neighbour k = 0 gives columns {1, 2}.
    const DenseMatrix<Complex>& other = restricted.at(1, 1);
    EXPECT_EQ(other(0, 0), Complex(101.0, 1.0));
    EXPECT_EQ(other(1, 1), Complex(122.0, 1.0));

    classification[1].kept_bands = {0, 3};
    EXPECT_THROW(overlaps.restricted_to(classification, ring_of_two()), std::invalid_argument);
    EXPECT_THROW(overlaps.restricted_to(std::vector<KPointBandClassification>(3), ring_of_two()),
                 std::invalid_argument);
}

TEST(BlochOverlaps, TransformsEachBlockByTheGaugesAtItsTwoKPoints) {
    const BlochOverlaps overlaps = BlochOverlaps::square(two_point_blocks(), 3);
    const std::vector<DenseMatrix<Complex>> identity(2, DenseMatrix<Complex>::identity(3));
    const BlochOverlaps unchanged = overlaps.transformed_by(identity, identity, ring_of_two());
    EXPECT_TRUE(unchanged.at(1, 1) == overlaps.at(1, 1));

    /**
     * A phase on the left at k = 0 and a different one on the right at k = 1:
     * left^dagger M right scales every entry of M(0, b) by conj(a) b.
     */
    std::vector<DenseMatrix<Complex>> left(2, DenseMatrix<Complex>::identity(3));
    std::vector<DenseMatrix<Complex>> right(2, DenseMatrix<Complex>::identity(3));
    const Complex a(0.0, 1.0);
    const Complex b(-1.0, 0.0);
    left[0] = left[0].scaled_columns(std::vector<Complex>{a, a, a});
    right[1] = right[1].scaled_columns(std::vector<Complex>{b, b, b});
    const BlochOverlaps rotated = overlaps.transformed_by(left, right, ring_of_two());
    EXPECT_NEAR(std::abs(rotated.at(0, 0)(1, 2) - std::conj(a) * b * overlaps.at(0, 0)(1, 2)),
                0.0,
                1.0e-12);
    EXPECT_TRUE(rotated.at(1, 0) == overlaps.at(1, 0)) << "k = 1's own gauge is the identity";

    EXPECT_THROW(overlaps.transformed_by(identity, {identity[0]}, ring_of_two()),
                 std::invalid_argument);
}
