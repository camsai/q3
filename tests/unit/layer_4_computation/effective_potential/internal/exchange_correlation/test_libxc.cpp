#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/factory.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"

using namespace q3;

/**
 * ── PZ-LDA: exchange energy at known density ────────────────────────
 * At rs = 1 bohr (rho = 3/(4π)):
 *   ex = -(3/4)(3/π)^{1/3} rho^{1/3} * (2 Ry factor)
 *   ex = -0.9163/rs  (in Ry)
 */
TEST(XCFunctional, PZ_ExchangeEnergy) {
    auto xc = XCFunctionalFactory::create("PZ");

    // rs = 1.0 → rho = 3/(4π·1³) = 0.23873
    Real rs = 1.0;
    Real rho_val = 3.0 / (4.0 * M_PI * rs * rs * rs);

    std::vector<Real> rho = {rho_val};
    std::vector<Real> vxc(1), exc(1);
    xc->compute(rho, vxc, exc);

    /**
     * Exchange energy per particle in Ry: -2 * 0.4582/rs (Ry = 2*Hartree)
     * Correlation at rs=1: ec ≈ -0.0480 Ry (PZ parameterization)
     * Total: ≈ -0.9164 - 0.096 ≈ -1.036 Ry  (approximate)
     * Actual computed value from our PZ implementation:
     */
    EXPECT_NEAR(exc[0], -1.036, 0.01);
}

// ── Zero density gives zero potential ───────────────────────────────
TEST(XCFunctional, ZeroDensity) {
    auto xc = XCFunctionalFactory::create("PZ");
    std::vector<Real> rho = {0.0, 0.0};
    std::vector<Real> vxc(2), exc(2);
    xc->compute(rho, vxc, exc);

    EXPECT_NEAR(vxc[0], 0.0, 1e-15);
    EXPECT_NEAR(exc[0], 0.0, 1e-15);
    EXPECT_NEAR(vxc[1], 0.0, 1e-15);
    EXPECT_NEAR(exc[1], 0.0, 1e-15);
}

// ── Potential is more negative than energy (|vxc| > |exc|) ──────────
TEST(XCFunctional, PotentialMoreNegative) {
    auto xc = XCFunctionalFactory::create("PZ");

    // Typical metallic density: rs ≈ 2
    Real rs = 2.0;
    Real rho_val = 3.0 / (4.0 * M_PI * rs * rs * rs);

    std::vector<Real> rho = {rho_val};
    std::vector<Real> vxc(1), exc(1);
    xc->compute(rho, vxc, exc);

    // vxc should be more negative than exc for LDA exchange
    EXPECT_LT(vxc[0], exc[0]);
    EXPECT_LT(vxc[0], 0.0);
    EXPECT_LT(exc[0], 0.0);
}

// ── Factory creates PZ ──────────────────────────────────────────────
TEST(XCFunctional, Factory) {
    auto xc = XCFunctionalFactory::create("PZ");
    EXPECT_EQ(xc->name(), "Slater exchange+Perdew & Zunger");
}

// ── Factory throws on unknown functional ─────────────────────────────
TEST(XCFunctional, Factory_Unknown) {
    EXPECT_THROW(XCFunctionalFactory::create("B3LYP"), std::runtime_error);
}

TEST(XCFunctional, GenericLibXCHybridFailsBeforeEvaluation) {
    const std::string functional_identifier =
        "libxc:" + std::to_string(XC_HYB_GGA_XC_B3LYP) + ":" + std::to_string(XC_GGA_C_LYP);

    try {
        (void)XCFunctionalFactory::create(functional_identifier);
        FAIL() << "Hybrid LibXC functional was accepted";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(functional_identifier), std::string::npos);
        EXPECT_NE(message.find("exchange component id=402"), std::string::npos);
        EXPECT_NE(message.find("HYB_GGA"), std::string::npos);
        EXPECT_NE(message.find("nonlocal exact-exchange operator"), std::string::npos);
    }
}

TEST(XCFunctional, QELibXCMetaGGAFailsBeforeEvaluation) {
    const std::string functional_identifier = "XC-001I-000I-000L-267L";

    try {
        (void)XCFunctionalFactory::create(functional_identifier);
        FAIL() << "Meta-GGA LibXC functional was accepted";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(functional_identifier), std::string::npos);
        EXPECT_NE(message.find("correlation component id=267"), std::string::npos);
        EXPECT_NE(message.find("MGGA"), std::string::npos);
        EXPECT_NE(message.find("kinetic-energy-density"), std::string::npos);
    }
}

/**
 * ── Spin-polarized PBE where the total density gradient vanishes ─────
 *
 * Between the opposite moments of an antiferromagnet the two spin densities are
 * equal and their gradients opposite, so sigma_up_up = sigma_down_down =
 * -sigma_up_down: the total gradient is zero while neither spin gradient is.
 * PBE correlation is smooth there, and its gradient derivatives have a finite
 * limit. LibXC 7.1.0, 7.1.1 and 7.1.2 return NaN for them at such a point, and a
 * build that picked one up ran the chromium AFM example to NaN from its first SCF
 * step. The input is the grid point captured from that run; the expected values
 * are LibXC 7.0.0's, in Rydberg. Against an affected LibXC -- which the build now
 * uses only under Q3_ALLOW_UNTESTED_SYSTEM_LIBXC -- the evaluation refuses the NaN
 * and this test fails.
 */
TEST(XCFunctional, SpinPolarizedPBEIsFiniteWhereTheTotalGradientVanishes) {
    auto xc = XCFunctionalFactory::create("PBE", 2);

    std::vector<Real> density_up = {0.014436788092055116};
    std::vector<Real> density_down = {0.014436788092055095};
    std::vector<Real> sigma_up_up = {1.0441942580908765e-05};
    std::vector<Real> sigma_up_down = {-1.0441942580906029e-05};
    std::vector<Real> sigma_down_down = {1.0441942580903294e-05};
    std::vector<Real> potential_up, potential_down;
    std::vector<Real> gradient_up_up, gradient_up_down, gradient_down_down;
    std::vector<Real> energy_density;

    ASSERT_NO_THROW(xc->compute_gradient_corrected_spin(density_up,
                                                        density_down,
                                                        sigma_up_up,
                                                        sigma_up_down,
                                                        sigma_down_down,
                                                        potential_up,
                                                        potential_down,
                                                        gradient_up_up,
                                                        gradient_up_down,
                                                        gradient_down_down,
                                                        energy_density));

    constexpr Real relative_tolerance = 1e-9;
    auto expect_close = [relative_tolerance](Real actual, Real expected, const char* name) {
        EXPECT_TRUE(std::isfinite(actual)) << name << " is not finite";
        EXPECT_NEAR(actual, expected, relative_tolerance * std::abs(expected)) << name;
    };
    expect_close(energy_density[0], -0.54361029545523687, "energy density");
    expect_close(potential_up[0], -0.70489556898315142, "potential up");
    expect_close(potential_down[0], -0.7048955689831522, "potential down");
    expect_close(gradient_up_up[0], -0.94173795856161457, "sigma_up_up derivative");
    expect_close(gradient_up_down[0], 1.912351711349038, "sigma_up_down derivative");
    expect_close(gradient_down_down[0], -0.94173795856162523, "sigma_down_down derivative");
}
