#include <gtest/gtest.h>

#include "io/volumetric/field.hpp"

using namespace q3;

namespace {

/**
 * A 2x3x4 field in a cell whose volume is easy to divide by, with each sample
 * carrying its own grid index so a transposition cannot hide.
 */
VolumetricField make_indexed_field() {
    VolumetricField field;
    field.grid_dimensions = {2, 3, 4};
    field.lattice_vectors_bohr[0] = {2.0, 0.0, 0.0};
    field.lattice_vectors_bohr[1] = {0.0, 3.0, 0.0};
    field.lattice_vectors_bohr[2] = {0.0, 0.0, 4.0};
    field.species_symbols = {"Si"};
    field.species_valence_charges = {4.0};
    field.atom_species_indices = {0};
    field.atom_positions_crystal = {{0.25, 0.5, 0.75}};
    field.values.resize(2 * 3 * 4);
    for (int index_1 = 0; index_1 < 2; index_1++)
        for (int index_2 = 0; index_2 < 3; index_2++)
            for (int index_3 = 0; index_3 < 4; index_3++)
                field.values[(index_1 * 3 + index_2) * 4 + index_3] =
                    100.0 * index_1 + 10.0 * index_2 + index_3;
    return field;
}

} // namespace

TEST(VolumetricField, ThirdIndexVariesFastest) {
    const VolumetricField field = make_indexed_field();
    EXPECT_EQ(field.values[0], 0.0);
    EXPECT_EQ(field.values[1], field.value(0, 0, 1));
    EXPECT_EQ(field.value(0, 0, 1), 1.0);
    EXPECT_EQ(field.value(0, 1, 0), 10.0);
    EXPECT_EQ(field.value(1, 0, 0), 100.0);
    EXPECT_EQ(field.value(1, 2, 3), 123.0);
}

TEST(VolumetricField, IntegralIsTheMeanTimesTheVolume) {
    VolumetricField field = make_indexed_field();
    EXPECT_DOUBLE_EQ(field.cell_volume_bohr3(), 24.0);

    /// A constant field integrates to value * volume, exactly.
    for (Real& value : field.values)
        value = 0.5;
    EXPECT_DOUBLE_EQ(field.integral(), 12.0);
}

TEST(VolumetricField, CartesianPositionFollowsTheLattice) {
    const VolumetricField field = make_indexed_field();
    const Vector3D position = field.cartesian_position_bohr(0);
    EXPECT_DOUBLE_EQ(position.x, 0.5);
    EXPECT_DOUBLE_EQ(position.y, 1.5);
    EXPECT_DOUBLE_EQ(position.z, 3.0);
}

TEST(VolumetricField, AtomicNumberAndSpeciesCountsComeFromTheSymbols) {
    VolumetricField field = make_indexed_field();
    field.species_symbols = {"Si", "O"};
    field.species_valence_charges = {4.0, 6.0};
    field.atom_species_indices = {1, 0, 1};
    field.atom_positions_crystal = {{0, 0, 0}, {0.5, 0.5, 0.5}, {0.25, 0.25, 0.25}};

    EXPECT_EQ(field.atomic_number(0), 8);
    EXPECT_EQ(field.atomic_number(1), 14);
    EXPECT_EQ(field.number_of_atoms(), 3);
    const std::vector<int> counts = field.atoms_per_species();
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts[0], 1);
    EXPECT_EQ(counts[1], 2);
}
