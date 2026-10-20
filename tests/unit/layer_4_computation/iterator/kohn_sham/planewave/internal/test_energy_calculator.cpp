#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "layer_4_computation/iterator/kohn_sham/planewave/internal/energy_calculator.hpp"

using namespace q3;

/**
 * Minimal SmearingScheme subclass that returns fixed smearing behavior.
 */
class TestSmearingScheme : public SmearingScheme {
  public:
    explicit TestSmearingScheme(Real width) : SmearingScheme(width) {}

    [[nodiscard]] Real occupation(Real) const override {
        return 0.5;
    }

    [[nodiscard]] Real entropy_weight(Real) const override {
        return 0.1;
    }

    [[nodiscard]] std::string name() const override {
        return "test_smearing";
    }
};

// ── Helper to set up a minimal quantum system ──────────────────────

/**
 * Creates a KPointsGrid with the given number of kpoints (1x1xN, no shift),
 * and a PlanewaveKohnShamQuantumSystem with the given parameters.
 * The quantum system's quantum states are allocated with 1 PW per kpoint.
 */
struct TestFixture {
    KPointsGrid kpoint_grid;
    std::unique_ptr<PlanewaveBasis> basis;
    PlanewaveKohnShamQuantumSystem quantum_system;

    TestFixture(int number_of_kpoints,
                int number_of_bands,
                int number_of_spin_components,
                Real number_of_electrons) {
        kpoint_grid.generate(1, 1, number_of_kpoints, 0, 0, 0);

        // Create a basis placeholder — the calculator doesn't use the basis directly.
        basis = std::make_unique<PlanewaveBasis>();

        quantum_system = PlanewaveKohnShamQuantumSystem(
            kpoint_grid, *basis, number_of_bands, number_of_spin_components, number_of_electrons);

        // Allocate states: 1 PW per kpoint for simplicity
        std::vector<int> plane_waves_per_kpoint(number_of_kpoints, 1);
        quantum_system.allocate_electronic_states(plane_waves_per_kpoint);
    }
};

// ── Null smearing scheme ────────────────────────────────────────────

TEST(PlanewaveKohnShamEnergyCalculator, SmearingEntropyIsZeroWithNullScheme) {
    TestFixture fixture(1, 2, 1, 2.0);
    PlanewaveKohnShamEnergyCalculator calculator(
        fixture.kpoint_grid, fixture.quantum_system, nullptr);
    EXPECT_DOUBLE_EQ(calculator.compute_smearing_entropy(2, 1), 0.0);
}

// ── Band energy with zero eigenvalues ───────────────────────────────

TEST(PlanewaveKohnShamEnergyCalculator, BandEnergyIsZeroForZeroEigenvalues) {
    TestFixture fixture(1, 2, 1, 2.0);
    // Eigenvalues are initialized to 0.0 by allocate_electronic_states
    PlanewaveKohnShamEnergyCalculator calculator(
        fixture.kpoint_grid, fixture.quantum_system, nullptr);
    EXPECT_DOUBLE_EQ(calculator.compute_band_energy(2, 1), 0.0);
}

// ── Band energy with known values ───────────────────────────────────

TEST(PlanewaveKohnShamEnergyCalculator, BandEnergyComputesWeightedSum) {
    // 1 kpoint, 2 bands, 1 spin, 2 electrons (1 occupied band)
    TestFixture fixture(1, 2, 1, 2.0);

    // Set eigenvalues: band 0 = 1.0, band 1 = 3.0
    fixture.quantum_system.set_eigenvalue(0, 0, 1.0);
    fixture.quantum_system.set_eigenvalue(0, 1, 3.0);

    PlanewaveKohnShamEnergyCalculator calculator(
        fixture.kpoint_grid, fixture.quantum_system, nullptr);

    /**
     * With fixed occupation: band 0 is occupied (occupation = 1.0), band 1 is empty.
     * Weight = 2.0 / 1 = 2.0 (1 kpoint).
     * band_energy = weight * (1.0 * 1.0 + 0.0 * 3.0) = 2.0 * 1.0 = 2.0
     */
    Real band_energy = calculator.compute_band_energy(2, 1);
    EXPECT_NEAR(band_energy, 2.0, 1e-12);
}

// ── Smearing entropy with active scheme ─────────────────────────────

TEST(PlanewaveKohnShamEnergyCalculator, SmearingEntropyWithActiveScheme) {
    TestFixture fixture(1, 2, 1, 2.0);
    fixture.quantum_system.set_eigenvalue(0, 0, 0.5);
    fixture.quantum_system.set_eigenvalue(0, 1, 1.5);
    fixture.quantum_system.set_fermi_energy(1.0);

    TestSmearingScheme scheme(0.01);
    PlanewaveKohnShamEnergyCalculator calculator(
        fixture.kpoint_grid, fixture.quantum_system, &scheme);

    Real entropy = calculator.compute_smearing_entropy(2, 1);
    // Should be non-zero since we have an active smearing scheme
    EXPECT_NE(entropy, 0.0);
}

// ── Smearing entropy with non-smearing scheme ───────────────────────

TEST(PlanewaveKohnShamEnergyCalculator, SmearingEntropyZeroForFixedOccupation) {
    TestFixture fixture(1, 2, 1, 2.0);

    // Width = 0.0 means uses_smearing() returns false
    TestSmearingScheme fixed_scheme(0.0);
    PlanewaveKohnShamEnergyCalculator calculator(
        fixture.kpoint_grid, fixture.quantum_system, &fixed_scheme);

    EXPECT_DOUBLE_EQ(calculator.compute_smearing_entropy(2, 1), 0.0);
}
