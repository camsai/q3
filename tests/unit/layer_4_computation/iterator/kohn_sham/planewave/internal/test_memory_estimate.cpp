#include "layer_4_computation/iterator/kohn_sham/planewave/internal/memory_estimate.hpp"
#include <gtest/gtest.h>

using namespace q3;

namespace {

/**
 * The 432-atom gamma-only silicon deck from camsai/q3#16: 1040 bands over the
 * half G-sphere, 18 Kleinman-Bylander projectors per atom, at the expansion
 * factor of 4 it was measured with (the default was 4 then, and is 2 now).
 */
PlanewaveMemoryEstimateInputs large_silicon_inputs() {
    PlanewaveMemoryEstimateInputs inputs;
    inputs.davidson_basis_expansion_factor = 4;
    inputs.number_of_bands = 1040;
    inputs.max_plane_waves_per_kpoint = 167155;
    inputs.number_of_g_vectors = 1338865;
    inputs.number_of_projectors = 7776;
    inputs.number_of_atomic_orbitals = 1728;
    inputs.number_of_atoms = 432;
    inputs.number_of_spin_components = 1;
    inputs.local_number_of_kpoint_states = 1;
    inputs.mixing_history_depth = 8;
    inputs.fft_rho_grid_points = 2560000;
    inputs.fft_wave_grid_points = 2560000;
    inputs.overlap_is_identity = true;
    return inputs;
}

constexpr double bytes_per_gibibyte = 1073741824.0;

} // namespace

// ─── Basis sizing ───────────────────────────────────────────────────────────

TEST(PlanewaveMemoryEstimate, BasisIsFourTimesTheGlobalBandCount) {
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(large_silicon_inputs());
    EXPECT_EQ(estimate.maximum_basis_size, 4160);
}

/**
 * At the default factor the basis is twice the band count, and the solver's
 * plane-wave-proportional terms come to six per band and plane wave: two basis
 * arrays at twice the bands, the rotation scratch and the wavefunctions --
 * one more than a solver whose scratch is its wavefunction array.
 */
TEST(PlanewaveMemoryEstimate, DefaultFactorHoldsSixComplexPerBandAndPlaneWave) {
    PlanewaveMemoryEstimateInputs inputs = large_silicon_inputs();
    inputs.davidson_basis_expansion_factor =
        PlanewaveMemoryEstimateInputs{}.davidson_basis_expansion_factor;
    EXPECT_EQ(inputs.davidson_basis_expansion_factor, 2);
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(inputs);
    EXPECT_EQ(estimate.maximum_basis_size, 2080);
    std::uint64_t unit = static_cast<std::uint64_t>(inputs.number_of_bands) *
                         static_cast<std::uint64_t>(inputs.max_plane_waves_per_kpoint) * 16;
    std::uint64_t per_plane_wave = 0;
    for (const auto& term : estimate.steady_state) {
        if (term.formula.find("npw") != std::string::npos &&
            term.formula.find("nkb") == std::string::npos) {
            per_plane_wave += term.bytes;
        }
    }
    EXPECT_NEAR(static_cast<double>(per_plane_wave) / static_cast<double>(unit), 6.0, 0.05);
}

/**
 * A cell with fewer plane waves than the expanded basis cannot hold that many
 * independent vectors, so the basis is clamped -- but never below the number of
 * bands the caller asked for.
 */
TEST(PlanewaveMemoryEstimate, BasisIsClampedByThePlaneWaveCount) {
    PlanewaveMemoryEstimateInputs inputs = large_silicon_inputs();
    inputs.number_of_bands = 100;
    inputs.max_plane_waves_per_kpoint = 250;
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(inputs);
    EXPECT_EQ(estimate.maximum_basis_size, 250);

    inputs.max_plane_waves_per_kpoint = 50;
    EXPECT_EQ(PlanewaveMemoryEstimate::compute(inputs).maximum_basis_size, 100);
}

// ─── Totals ─────────────────────────────────────────────────────────────────

/**
 * The 432-atom deck of camsai/q3#16, which measured about 100 GiB of named
 * arrays before any of this work. Dropping the collapse scratch, the extraction
 * workspace and the identity-overlap copies takes it to roughly half that.
 */
TEST(PlanewaveMemoryEstimate, LargeSiliconDeckLandsNearTheMeasuredFootprint) {
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(large_silicon_inputs());
    double steady_gibibytes = static_cast<double>(estimate.steady_state_bytes) / bytes_per_gibibyte;
    EXPECT_GT(steady_gibibytes, 35.0);
    EXPECT_LT(steady_gibibytes, 50.0);
}

/**
 * What the solver now holds per band and plane wave, which is the number to
 * compare against another code: two basis arrays at the expansion factor, the
 * rotation scratch and the wavefunctions. A solver that reuses its wavefunction
 * array as the rotation scratch holds five of these at an expansion factor of
 * two.
 */
TEST(PlanewaveMemoryEstimate, SolverHoldsTenComplexPerBandAndPlaneWave) {
    PlanewaveMemoryEstimateInputs inputs = large_silicon_inputs();
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(inputs);
    std::uint64_t unit = static_cast<std::uint64_t>(inputs.number_of_bands) *
                         static_cast<std::uint64_t>(inputs.max_plane_waves_per_kpoint) * 16;
    /**
     * Only the terms that scale with the plane-wave count: the subspace
     * matrices go as the basis squared and belong to a different comparison.
     */
    std::uint64_t per_plane_wave = 0;
    for (const auto& term : estimate.steady_state) {
        if (term.formula.find("npw") != std::string::npos &&
            term.formula.find("nkb") == std::string::npos) {
            per_plane_wave += term.bytes;
        }
    }
    // 2 x M (= 8 nb) + rotation scratch + wavefunctions = 10 nb.
    EXPECT_NEAR(static_cast<double>(per_plane_wave) / static_cast<double>(unit), 10.0, 0.05);
}

/**
 * The collapse no longer has scratch of its own: it rotates through the Ritz
 * buffers, which are already accounted for and are idle at that point in the
 * Davidson loop.
 */
TEST(PlanewaveMemoryEstimate, CollapseHasNoScratchOfItsOwn) {
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(large_silicon_inputs());
    for (const auto& term : estimate.steady_state) {
        EXPECT_EQ(term.label.find("collapse"), std::string::npos);
    }
}

TEST(PlanewaveMemoryEstimate, SolverShareIsMostOfTheSteadyState) {
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(large_silicon_inputs());
    EXPECT_GT(estimate.solver_bytes * 2, estimate.steady_state_bytes);
}

// ─── Phases ─────────────────────────────────────────────────────────────────

/**
 * The setup spike and the solver workspace never coexist, so the peak is the
 * taller phase rather than their sum.
 */
TEST(PlanewaveMemoryEstimate, PeakIsTheTallerPhaseNotTheSum) {
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(large_silicon_inputs());
    EXPECT_EQ(estimate.peak_bytes,
              std::max(estimate.steady_state_bytes, estimate.setup_peak_bytes));
    EXPECT_LT(estimate.peak_bytes, estimate.steady_state_bytes + estimate.setup_peak_bytes);
}

/**
 * With no Davidson workspace the setup spike is what a run has to fit, so the
 * peak has to exceed the steady state rather than equal it.
 */
TEST(PlanewaveMemoryEstimate, DirectSolverPeaksDuringSetup) {
    PlanewaveMemoryEstimateInputs inputs = large_silicon_inputs();
    inputs.diagonalizer_is_davidson = false;
    PlanewaveMemoryEstimate estimate = PlanewaveMemoryEstimate::compute(inputs);
    EXPECT_EQ(estimate.solver_bytes, 0u);
    EXPECT_GT(estimate.peak_bytes, estimate.steady_state_bytes);
}

// ─── Labels ─────────────────────────────────────────────────────────────────

/**
 * A norm-conserving run does not store the overlap arrays at all: S is the
 * identity, so they would duplicate the basis and the Ritz vectors. Ultrasoft
 * needs them, and pays exactly the basis plus one band block more.
 */
TEST(PlanewaveMemoryEstimate, IdentityOverlapDropsTheOverlapArrays) {
    PlanewaveMemoryEstimate identity = PlanewaveMemoryEstimate::compute(large_silicon_inputs());
    for (const auto& term : identity.steady_state) {
        EXPECT_EQ(term.label.find("SV"), std::string::npos);
        EXPECT_EQ(term.label.find("Ritz overlap"), std::string::npos);
    }

    PlanewaveMemoryEstimateInputs ultrasoft_inputs = large_silicon_inputs();
    ultrasoft_inputs.overlap_is_identity = false;
    PlanewaveMemoryEstimate ultrasoft = PlanewaveMemoryEstimate::compute(ultrasoft_inputs);
    EXPECT_GT(ultrasoft.steady_state_bytes, identity.steady_state_bytes);

    const std::uint64_t complex_size = 16;
    std::uint64_t expected_extra =
        (static_cast<std::uint64_t>(identity.maximum_basis_size) +
         static_cast<std::uint64_t>(ultrasoft_inputs.number_of_bands)) *
        static_cast<std::uint64_t>(ultrasoft_inputs.max_plane_waves_per_kpoint) * complex_size;
    EXPECT_EQ(ultrasoft.steady_state_bytes - identity.steady_state_bytes, expected_extra);
}
