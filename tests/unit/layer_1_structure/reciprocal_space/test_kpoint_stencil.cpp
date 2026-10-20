#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"

using namespace q3;

namespace {

KPointNeighborShells cubic_shells(int dimension) {
    KPointNeighborShells shells;
    Matrix3x3 reciprocal;
    reciprocal[0] = Vector3D(1.0, 0.0, 0.0);
    reciprocal[1] = Vector3D(0.0, 1.0, 0.0);
    reciprocal[2] = Vector3D(0.0, 0.0, 1.0);
    shells.compute(reciprocal, dimension, dimension, dimension);
    return shells;
}

} // namespace

TEST(KPointStencil, ResolvesEveryNeighbourOfAGridOnce) {
    const KPointNeighborShells shells = cubic_shells(3);
    const std::array<int, 3> grid = {3, 3, 3};
    const KPointStencil stencil = KPointStencil::from_grid(grid, shells.neighbor_directions());

    EXPECT_EQ(stencil.number_of_kpoints(), 27);
    EXPECT_EQ(stencil.number_of_directions(), 6);
    EXPECT_TRUE(stencil.has_b_vectors());
    EXPECT_TRUE(stencil.has_grid_steps());
    for (int kpoint = 0; kpoint < 27; kpoint++) {
        for (int direction = 0; direction < 6; direction++) {
            const KPointNeighborDirection& shell =
                shells.neighbor_directions()[static_cast<std::size_t>(direction)];
            EXPECT_EQ(stencil.neighbor(kpoint, direction),
                      KPointNeighborShells::neighbor_kpoint_index(kpoint, shell.grid_step, grid));
            EXPECT_DOUBLE_EQ(stencil.weight(direction), shell.weight);
            EXPECT_EQ(stencil.b_vector(direction).x, shell.reciprocal_lattice_translation.x);
            EXPECT_EQ(stencil.grid_step(direction).x, shell.grid_step.x);
        }
    }
    EXPECT_NEAR(stencil.total_weight(), 6.0 * shells.neighbor_directions()[0].weight, 1.0e-12);
    EXPECT_FALSE(stencil.two_point_axis_direction().has_value()) << "three points per axis";
}

TEST(KPointStencil, NamesTheDirectionWhoseOppositeLandsOnTheSamePoint) {
    const KPointNeighborShells shells = cubic_shells(2);
    const KPointStencil stencil = KPointStencil::from_grid({2, 2, 2}, shells.neighbor_directions());
    const std::optional<int> direction = stencil.two_point_axis_direction();
    ASSERT_TRUE(direction.has_value());
    // With two points per axis, +b and -b from k are the same k-point.
    const IntegerVector3D& step = stencil.grid_step(*direction);
    const IntegerVector3D opposite = {-step.x, -step.y, -step.z};
    EXPECT_EQ(KPointNeighborShells::neighbor_kpoint_index(0, step, {2, 2, 2}),
              KPointNeighborShells::neighbor_kpoint_index(0, opposite, {2, 2, 2}));
}

TEST(KPointStencil, BuildsFromATableWithoutBVectors) {
    const KPointStencil stencil =
        KPointStencil::from_table({{1, 3}, {2, 0}, {3, 1}, {0, 2}}, {0.5, 0.5});
    EXPECT_EQ(stencil.number_of_kpoints(), 4);
    EXPECT_EQ(stencil.number_of_directions(), 2);
    EXPECT_EQ(stencil.neighbor(2, 0), 3);
    EXPECT_EQ(stencil.neighbor(2, 1), 1);
    EXPECT_DOUBLE_EQ(stencil.total_weight(), 1.0);
    EXPECT_FALSE(stencil.has_b_vectors());
    EXPECT_FALSE(stencil.has_grid_steps());
    EXPECT_FALSE(stencil.two_point_axis_direction().has_value())
        << "a table carries no steps to compare";
    EXPECT_THROW(stencil.b_vector(0), std::logic_error);
    EXPECT_THROW(stencil.grid_step(0), std::logic_error);

    const KPointStencil with_b = KPointStencil::from_table(
        {{1, 0}, {0, 1}}, {2.0, 2.0}, {Vector3D(0.5, 0.0, 0.0), Vector3D(-0.5, 0.0, 0.0)});
    EXPECT_TRUE(with_b.has_b_vectors());
    EXPECT_DOUBLE_EQ(with_b.b_vector(1).x, -0.5);
}

TEST(KPointStencil, RefusesATableThatIsNotAStencil) {
    EXPECT_THROW((void)KPointStencil::from_table({{1, 4}, {2, 0}, {3, 1}, {0, 2}}, {0.5, 0.5}),
                 std::invalid_argument)
        << "k-point 4 does not exist";
    EXPECT_THROW((void)KPointStencil::from_table({{1}, {2, 0}, {3, 1}, {0, 2}}, {0.5, 0.5}),
                 std::invalid_argument)
        << "one row is short a direction";
    EXPECT_THROW(
        (void)KPointStencil::from_table({{1, 3}, {0, 2}}, {0.5, 0.5}, {Vector3D(1.0, 0.0, 0.0)}),
        std::invalid_argument)
        << "one b-vector for two directions";
    /**
     * Bare braces, which is why the two ways of building a stencil are named:
     * as overloaded constructors an empty brace fitted the grid's array as
     * well as the table's vector, which one compiler resolved and another
     * called ambiguous.
     */
    EXPECT_THROW((void)KPointStencil::from_table({{}, {}}, {}), std::invalid_argument)
        << "no directions";
    EXPECT_THROW((void)KPointStencil::from_table({}, {0.5}), std::invalid_argument)
        << "no k-points";
    EXPECT_THROW((void)KPointStencil::from_grid({0, 2, 2}, cubic_shells(2).neighbor_directions()),
                 std::invalid_argument)
        << "a grid dimension of zero";
}
