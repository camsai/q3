#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "layer_4_computation/iterator/kohn_sham/planewave/internal/reporter.hpp"

using namespace q3;

// ── SetupSummary struct defaults ────────────────────────────────────

TEST(PlanewaveKohnShamSCFSetupSummary, DefaultFieldsAreZeroOrEmpty) {
    PlanewaveKohnShamSCFSetupSummary summary{};
    EXPECT_DOUBLE_EQ(summary.lattice_parameter, 0.0);
    EXPECT_DOUBLE_EQ(summary.cell_volume, 0.0);
    EXPECT_EQ(summary.number_of_atoms, 0);
    EXPECT_DOUBLE_EQ(summary.wavefunction_energy_cutoff, 0.0);
    EXPECT_DOUBLE_EQ(summary.charge_density_energy_cutoff, 0.0);
    EXPECT_EQ(summary.number_of_g_vectors, 0);
    EXPECT_EQ(summary.number_of_kpoints, 0);
    EXPECT_EQ(summary.number_of_bands, 0);
    EXPECT_EQ(summary.number_of_spin_components, 0);
    EXPECT_TRUE(summary.xc_functional_name.empty());
    EXPECT_TRUE(summary.diagonalizer_name.empty());
    EXPECT_TRUE(summary.mixer_name.empty());
}

// ── ConvergenceSummary struct defaults ───────────────────────────────

TEST(PlanewaveKohnShamSCFConvergenceSummary, DefaultFieldsAreZeroOrNull) {
    PlanewaveKohnShamSCFConvergenceSummary summary{};
    EXPECT_EQ(summary.number_of_spin_components, 0);
    EXPECT_EQ(summary.number_of_kpoints, 0);
    EXPECT_EQ(summary.number_of_bands, 0);
    EXPECT_EQ(summary.energies, nullptr);
    EXPECT_TRUE(summary.eigenvalues.empty());
    EXPECT_TRUE(summary.kpoints.empty());
    EXPECT_DOUBLE_EQ(summary.total_magnetization, 0.0);
    EXPECT_DOUBLE_EQ(summary.absolute_magnetization, 0.0);
}

// ── SetupSummary population ─────────────────────────────────────────

TEST(PlanewaveKohnShamSCFSetupSummary, CanPopulateAllFields) {
    PlanewaveKohnShamSCFSetupSummary summary{};
    summary.lattice_parameter = 10.26;
    summary.cell_volume = 270.0;
    summary.number_of_atoms = 2;
    summary.wavefunction_energy_cutoff = 30.0;
    summary.charge_density_energy_cutoff = 240.0;
    summary.fft_rho_grid_dimension_1 = 36;
    summary.fft_rho_grid_dimension_2 = 36;
    summary.fft_rho_grid_dimension_3 = 36;
    summary.fft_wave_grid_dimension_1 = 20;
    summary.fft_wave_grid_dimension_2 = 20;
    summary.fft_wave_grid_dimension_3 = 20;
    summary.number_of_g_vectors = 3000;
    summary.number_of_kpoints = 4;
    summary.max_plane_waves_per_kpoint = 500;
    summary.number_of_spin_components = 1;
    summary.number_of_electrons = 8.0;
    summary.number_of_bands = 10;
    summary.xc_functional_name = "PBE";
    summary.diagonalizer_name = "davidson";
    summary.mixer_name = "broyden";
    summary.mixing_beta = 0.7;
    summary.ewald_energy = -8.12345;
    summary.density_symmetrization_active = true;
    summary.number_of_shells = 100;
    summary.number_of_symmetry_operations = 48;

    EXPECT_DOUBLE_EQ(summary.lattice_parameter, 10.26);
    EXPECT_EQ(summary.xc_functional_name, "PBE");
    EXPECT_EQ(summary.number_of_symmetry_operations, 48);
}

// ── Reporter static methods run without crashing ────────────────────

TEST(PlanewaveKohnShamSCFReporter, ReportSetupSummaryRunsWithoutCrash) {
    PlanewaveKohnShamSCFSetupSummary summary{};
    summary.lattice_parameter = 10.26;
    summary.cell_volume = 270.0;
    summary.number_of_atoms = 2;
    summary.wavefunction_energy_cutoff = 30.0;
    summary.charge_density_energy_cutoff = 240.0;
    summary.fft_rho_grid_dimension_1 = 36;
    summary.fft_rho_grid_dimension_2 = 36;
    summary.fft_rho_grid_dimension_3 = 36;
    summary.fft_wave_grid_dimension_1 = 20;
    summary.fft_wave_grid_dimension_2 = 20;
    summary.fft_wave_grid_dimension_3 = 20;
    summary.number_of_g_vectors = 3000;
    summary.number_of_kpoints = 4;
    summary.max_plane_waves_per_kpoint = 500;
    summary.number_of_spin_components = 1;
    summary.number_of_electrons = 8.0;
    summary.number_of_bands = 10;
    summary.xc_functional_name = "PBE";
    summary.diagonalizer_name = "davidson";
    summary.mixer_name = "broyden";
    summary.mixing_beta = 0.7;
    summary.ewald_energy = -8.12345;
    summary.density_symmetrization_active = false;
    summary.number_of_shells = 0;
    summary.number_of_symmetry_operations = 0;

    // Should not throw — reporter logs to spdlog which doesn't crash in tests
    EXPECT_NO_THROW(PlanewaveKohnShamSCFReporter::report_setup_summary(summary));
}

TEST(PlanewaveKohnShamSCFReporter, ReportFirstIterationRunsWithoutCrash) {
    PlanewaveKohnShamEnergies energies;
    energies.band_energy = -1.0;
    energies.double_counting_correction = 0.5;
    energies.hartree_energy = 2.0;
    energies.exchange_correlation_energy = -0.5;
    energies.ewald_energy = -3.0;

    EXPECT_NO_THROW(PlanewaveKohnShamSCFReporter::report_first_iteration(-2.0, energies));
}

TEST(PlanewaveKohnShamSCFReporter, ReportConvergenceSummaryRunsWithoutCrash) {
    PlanewaveKohnShamEnergies energies;
    energies.band_energy = -1.0;
    energies.double_counting_correction = 0.5;
    energies.hartree_energy = 2.0;
    energies.exchange_correlation_energy = -0.5;
    energies.ewald_energy = -3.0;

    PlanewaveKohnShamSCFConvergenceSummary summary{};
    summary.number_of_spin_components = 1;
    summary.number_of_kpoints = 1;
    summary.number_of_bands = 2;
    summary.energies = &energies;

    // Set up eigenvalues and kpoints
    summary.eigenvalues = {{-0.1, 0.5}};
    KPoint kpoint;
    kpoint.coordinates = Vector3D(0.0, 0.0, 0.0);
    kpoint.weight = 2.0;
    summary.kpoints = {kpoint};

    EXPECT_NO_THROW(PlanewaveKohnShamSCFReporter::report_convergence_summary(summary));
}

// ── PlanewaveKohnShamEnergies total ─────────────────────────────────

TEST(PlanewaveKohnShamEnergies, TotalIncludesAllComponents) {
    PlanewaveKohnShamEnergies energies;
    energies.band_energy = 1.0;
    energies.double_counting_correction = 2.0;
    energies.hartree_energy = 3.0;
    energies.exchange_correlation_energy = 4.0;
    energies.smearing_entropy_contribution = 0.1;
    energies.ewald_energy = -5.0;

    // total = band + dcc + hartree + xc + smearing + ewald
    EXPECT_NEAR(energies.total(), 1.0 + 2.0 + 3.0 + 4.0 + 0.1 - 5.0, 1e-12);
}

TEST(PlanewaveKohnShamEnergies, DefaultTotalIsZero) {
    PlanewaveKohnShamEnergies energies;
    EXPECT_DOUBLE_EQ(energies.total(), 0.0);
}
