#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/atom.hpp"
#include "layer_1_structure/crystal/atom_species.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/transform/wavefunction_transformer.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

static execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Helper to build a simple cubic crystal for wavefunction transformer tests.
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
 * Test round-trip G→R→G for a single-coefficient wavefunction (G=0 only).
 * The recovered G-space coefficients should match the original.
 */
TEST(WavefunctionTransformer, SingleCoefficientRoundTrip) {
    Crystal crystal = build_simple_cubic_crystal();
    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal, 4.0, 16.0);

    const auto& g_vectors_grid = planewave_basis.g_vectors_grid();
    const auto& fft_wave = planewave_basis.fft_wave();

    // Use just the first G-vector (G=0)
    int number_of_plane_waves = 1;
    std::vector<int> g_vector_indices = {0};

    // Set coefficient for G=0
    std::vector<Complex> coefficients_g_space = {Complex(2.5, -1.3)};

    // G → R
    std::vector<Complex> coefficients_real_space;
    WavefunctionTransformer::g_to_r(g_vectors_grid,
                                    fft_wave,
                                    coefficients_g_space,
                                    g_vector_indices,
                                    number_of_plane_waves,
                                    coefficients_real_space);

    ASSERT_EQ(static_cast<int>(coefficients_real_space.size()), fft_wave.total_grid_points());

    // R → G round-trip
    std::vector<Complex> coefficients_g_roundtrip;
    WavefunctionTransformer::r_to_g(g_vectors_grid,
                                    fft_wave,
                                    coefficients_real_space,
                                    g_vector_indices,
                                    number_of_plane_waves,
                                    coefficients_g_roundtrip);

    ASSERT_EQ(static_cast<int>(coefficients_g_roundtrip.size()), number_of_plane_waves);

    // Should recover original coefficient
    EXPECT_NEAR(coefficients_g_roundtrip[0].real(), coefficients_g_space[0].real(), 1e-10);
    EXPECT_NEAR(coefficients_g_roundtrip[0].imag(), coefficients_g_space[0].imag(), 1e-10);
}

/**
 * Test round-trip G→R→G for multiple G-vector coefficients.
 * Uses several G-vectors from the basis to verify correct mapping.
 */
TEST(WavefunctionTransformer, MultipleCoefficientRoundTrip) {
    Crystal crystal = build_simple_cubic_crystal();
    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal, 4.0, 16.0);

    const auto& g_vectors_grid = planewave_basis.g_vectors_grid();
    const auto& fft_wave = planewave_basis.fft_wave();

    int total_g_vectors = g_vectors_grid.number_of_g_vectors();
    // Use a handful of G-vectors (or all if there are few)
    int number_of_plane_waves = std::min(5, total_g_vectors);
    std::vector<int> g_vector_indices(number_of_plane_waves);
    for (int index = 0; index < number_of_plane_waves; index++) {
        g_vector_indices[index] = index;
    }

    // Set known coefficients
    std::vector<Complex> coefficients_g_space(number_of_plane_waves);
    for (int index = 0; index < number_of_plane_waves; index++) {
        coefficients_g_space[index] =
            Complex(static_cast<Real>(index + 1) * 0.5, static_cast<Real>(index) * -0.3);
    }

    // G → R
    std::vector<Complex> coefficients_real_space;
    WavefunctionTransformer::g_to_r(g_vectors_grid,
                                    fft_wave,
                                    coefficients_g_space,
                                    g_vector_indices,
                                    number_of_plane_waves,
                                    coefficients_real_space);

    // R → G round-trip
    std::vector<Complex> coefficients_g_roundtrip;
    WavefunctionTransformer::r_to_g(g_vectors_grid,
                                    fft_wave,
                                    coefficients_real_space,
                                    g_vector_indices,
                                    number_of_plane_waves,
                                    coefficients_g_roundtrip);

    ASSERT_EQ(static_cast<int>(coefficients_g_roundtrip.size()), number_of_plane_waves);

    for (int index = 0; index < number_of_plane_waves; index++) {
        EXPECT_NEAR(
            coefficients_g_roundtrip[index].real(), coefficients_g_space[index].real(), 1e-10)
            << "Real part mismatch at plane wave index " << index;
        EXPECT_NEAR(
            coefficients_g_roundtrip[index].imag(), coefficients_g_space[index].imag(), 1e-10)
            << "Imag part mismatch at plane wave index " << index;
    }
}

/**
 * Test that g_to_r produces the correct output size.
 */
TEST(WavefunctionTransformer, OutputSizeMatchesGrid) {
    Crystal crystal = build_simple_cubic_crystal();
    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal, 4.0, 16.0);

    const auto& g_vectors_grid = planewave_basis.g_vectors_grid();
    const auto& fft_wave = planewave_basis.fft_wave();

    int number_of_plane_waves = 1;
    std::vector<int> g_vector_indices = {0};
    std::vector<Complex> coefficients_g_space = {Complex(1.0, 0.0)};

    std::vector<Complex> coefficients_real_space;
    WavefunctionTransformer::g_to_r(g_vectors_grid,
                                    fft_wave,
                                    coefficients_g_space,
                                    g_vector_indices,
                                    number_of_plane_waves,
                                    coefficients_real_space);

    EXPECT_EQ(static_cast<int>(coefficients_real_space.size()), fft_wave.total_grid_points());
}
