#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/atom.hpp"
#include "layer_1_structure/crystal/atom_species.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/transform/charge_density_transformer.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

static execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Helper to build a simple cubic crystal for transformer tests.
 */
static Crystal build_simple_cubic_crystal() {
    Crystal crystal;
    crystal.set_from_lattice_type(1, 10.0);
    AtomSpecies species{"Si", 28.08, 4.0, ""};
    crystal.add_species(species);
    crystal.add_atom({0, {0.0, 0.0, 0.0}});
    crystal.compute_derived_quantities();
    return crystal;
}

/**
 * Test that a uniform (constant) charge density in G-space (only G=0 component)
 * transforms to a uniform real-space charge density, and back to G-space
 * correctly via round-trip.
 */
TEST(ChargeDensityTransformer, UniformChargeRoundTrip) {
    Crystal crystal = build_simple_cubic_crystal();
    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal, 4.0, 16.0);

    const auto& g_vectors_grid = planewave_basis.g_vectors_grid();
    const auto& fft_rho = planewave_basis.fft_rho();
    int number_of_g_vectors = g_vectors_grid.number_of_g_vectors();
    int number_of_grid_points = fft_rho.total_grid_points();

    // Set up G-space charge: only G=0 component is nonzero
    std::vector<Complex> charge_density_g_space(number_of_g_vectors, Complex(0.0, 0.0));
    Real uniform_value = 1.5;
    charge_density_g_space[0] = Complex(uniform_value, 0.0);

    // G → R
    std::vector<Real> charge_density_real_space;
    ChargeDensityTransformer::rho_g_to_r(
        g_vectors_grid, fft_rho, charge_density_g_space, charge_density_real_space);

    ASSERT_EQ(static_cast<int>(charge_density_real_space.size()), number_of_grid_points);

    /**
     * Uniform charge in G-space → uniform in R-space.
     * FFTGrid.inverse() includes 1/N normalization, and rho_g_to_r multiplies
     * by N to compensate, so the net result is the original G=0 value.
     */
    for (int grid_index = 0; grid_index < number_of_grid_points; grid_index++) {
        EXPECT_NEAR(charge_density_real_space[grid_index], uniform_value, 1e-10)
            << "Mismatch at grid index " << grid_index;
    }

    // R → G round-trip
    std::vector<Complex> charge_density_g_roundtrip;
    ChargeDensityTransformer::rho_r_to_g(
        g_vectors_grid, fft_rho, charge_density_real_space, charge_density_g_roundtrip);

    ASSERT_EQ(static_cast<int>(charge_density_g_roundtrip.size()), number_of_g_vectors);

    // G=0 should recover the original value
    EXPECT_NEAR(charge_density_g_roundtrip[0].real(), uniform_value, 1e-10);
    EXPECT_NEAR(charge_density_g_roundtrip[0].imag(), 0.0, 1e-10);

    // All other G components should be near zero
    for (int g_index = 1; g_index < number_of_g_vectors; g_index++) {
        EXPECT_NEAR(charge_density_g_roundtrip[g_index].real(), 0.0, 1e-10)
            << "Non-zero real at G-vector index " << g_index;
        EXPECT_NEAR(charge_density_g_roundtrip[g_index].imag(), 0.0, 1e-10)
            << "Non-zero imag at G-vector index " << g_index;
    }
}

/**
 * Test that rho_r_to_g correctly normalizes by 1/N_grid.
 * A uniform real-space density of 1.0 should give G=0 component = 1.0.
 */
TEST(ChargeDensityTransformer, RealToGNormalization) {
    Crystal crystal = build_simple_cubic_crystal();
    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal, 4.0, 16.0);

    const auto& g_vectors_grid = planewave_basis.g_vectors_grid();
    const auto& fft_rho = planewave_basis.fft_rho();
    int number_of_grid_points = fft_rho.total_grid_points();
    int number_of_g_vectors = g_vectors_grid.number_of_g_vectors();

    // Uniform real-space density = 1.0 everywhere
    std::vector<Real> charge_density_real_space(number_of_grid_points, 1.0);

    std::vector<Complex> charge_density_g_space;
    ChargeDensityTransformer::rho_r_to_g(
        g_vectors_grid, fft_rho, charge_density_real_space, charge_density_g_space);

    ASSERT_EQ(static_cast<int>(charge_density_g_space.size()), number_of_g_vectors);

    // G=0 component should be 1.0 (sum of 1.0 * N / N = 1.0)
    EXPECT_NEAR(charge_density_g_space[0].real(), 1.0, 1e-10);
    EXPECT_NEAR(charge_density_g_space[0].imag(), 0.0, 1e-10);

    // All other G components should be zero
    for (int g_index = 1; g_index < number_of_g_vectors; g_index++) {
        EXPECT_NEAR(std::abs(charge_density_g_space[g_index]), 0.0, 1e-10);
    }
}
