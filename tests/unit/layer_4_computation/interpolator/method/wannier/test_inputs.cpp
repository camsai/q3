#include <gtest/gtest.h>

#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"

using namespace q3;

TEST(WannierInputs, DefaultConstructedRecordIsEmpty) {
    WannierInputs record;

    EXPECT_EQ(record.number_of_kpoints(), 0);
    EXPECT_EQ(record.number_of_bands, 0);
    EXPECT_EQ(record.number_of_trial_orbitals, 0);
    EXPECT_TRUE(record.neighbor_directions.empty());
    EXPECT_TRUE(record.overlap_matrices.empty());
    EXPECT_TRUE(record.projection_matrices.empty());
}

TEST(WannierInputs, NumberOfKpointsMatchesCoordinateListSize) {
    WannierInputs record;
    record.kpoint_coordinates = {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.5, 0.0, 0.0}};

    EXPECT_EQ(record.number_of_kpoints(), 3);
}
