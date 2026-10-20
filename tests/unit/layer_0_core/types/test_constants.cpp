#include <cmath>
#include <gtest/gtest.h>

#include "layer_0_core/types/constants.hpp"

using namespace q3;

TEST(Constants, PiValue) {
    EXPECT_NEAR(constants::pi, 3.14159265358979323846, 1e-15);
}

TEST(Constants, TwoPiConsistency) {
    EXPECT_DOUBLE_EQ(constants::two_pi, 2.0 * constants::pi);
}

TEST(Constants, FourPiConsistency) {
    EXPECT_DOUBLE_EQ(constants::four_pi, 4.0 * constants::pi);
}

TEST(Constants, RydbergToElectronVolt) {
    EXPECT_NEAR(constants::RY_TO_EV, 13.6058, 0.001);
}

TEST(Constants, BohrRadiusAngstrom) {
    EXPECT_NEAR(constants::BOHR_RADIUS_ANGSTROM, 0.529177, 1e-4);
}

TEST(Constants, HartreeToEvIsDoubleRydbergToEv) {
    EXPECT_DOUBLE_EQ(constants::HARTREE_TO_EV, 2.0 * constants::RY_TO_EV);
}

TEST(Constants, ElectronChargeSquaredRydberg) {
    EXPECT_DOUBLE_EQ(constants::electron_charge_squared_rydberg, 2.0);
}

TEST(Constants, SpinDegeneracy) {
    EXPECT_DOUBLE_EQ(constants::spin_degeneracy, 2.0);
}

TEST(Constants, SqrtPiValue) {
    EXPECT_NEAR(constants::sqrt_pi, std::sqrt(constants::pi), 1e-14);
}

TEST(Constants, Sqrt2Value) {
    EXPECT_NEAR(constants::sqrt2, std::sqrt(2.0), 1e-14);
}

TEST(Constants, BohrRadiusSiConsistency) {
    EXPECT_NEAR(constants::BOHR_RADIUS_ANGSTROM, constants::BOHR_RADIUS_SI * 1e10, 1e-15);
}

TEST(Constants, AngstromToBohrInverse) {
    EXPECT_NEAR(constants::ANGSTROM_TO_BOHR, 1.0 / constants::BOHR_RADIUS_ANGSTROM, 1e-14);
}
