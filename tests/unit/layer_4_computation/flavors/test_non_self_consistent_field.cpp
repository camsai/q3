#include <gtest/gtest.h>

#include "layer_0_core/types/constants.hpp"
#include "layer_4_computation/flavors/non_self_consistent_field.hpp"

using namespace q3;

/**
 * The working basis a non-self-consistent pass runs with: a default below the
 * floor is raised, and a factor the deck named is kept, even below it.
 */
TEST(NonSelfConsistentFieldDavidsonBasis, TheDefaultIsRaisedToTheFloor) {
    const int floor = constants::non_self_consistent_field_minimum_davidson_basis_expansion_factor;
    EXPECT_EQ(non_self_consistent_field_davidson_basis_expansion_factor(2, false), floor);
}

TEST(NonSelfConsistentFieldDavidsonBasis, AFactorTheDeckNamedIsKeptEvenBelowTheFloor) {
    EXPECT_EQ(non_self_consistent_field_davidson_basis_expansion_factor(2, true), 2);
}

TEST(NonSelfConsistentFieldDavidsonBasis, ALargerFactorIsNeverLowered) {
    EXPECT_EQ(non_self_consistent_field_davidson_basis_expansion_factor(6, false), 6);
    EXPECT_EQ(non_self_consistent_field_davidson_basis_expansion_factor(6, true), 6);
}
