#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_4_computation/charge_density/planewave/internal/accumulator.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"

using namespace q3;

/**
 * Helper: load the norm-conserving Si pseudopotential.
 */
static PseudopotentialSet load_norm_conserving_si_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));
    return pseudopotential_set;
}

/**
 * Build a KohnShamSCFIteratorConfig for norm-conserving Si with defaults.
 */
static KohnShamSCFIteratorConfig build_scf_config(Crystal& crystal_structure,
                                                  KPointsGrid& kpoint_grid,
                                                  PseudopotentialSet& pseudopotential_set) {
    KohnShamSCFIteratorConfig config;
    config.crystal_structure = &crystal_structure;
    config.number_of_spin_components = 1;
    config.kpoint_grid = &kpoint_grid;
    config.pseudopotential_set = &pseudopotential_set;
    return config;
}

// ── Test: total charge integrates to number of electrons ────────────

TEST(ChargeDensityAccumulator, TotalChargeEqualsNumberOfElectrons) {
    auto pseudopotential_set = load_norm_conserving_si_pseudopotential();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si_species{"Si", 28.08, pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si_species);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 16.0;
    config.maximum_iterations = 1;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    const auto& basis = planewave_iterator->planewave_basis();
    const auto& quantum_system = planewave_iterator->quantum_system();

    // Create accumulator and compute charge density
    PlanewaveChargeDensityAccumulator accumulator(basis, kpoint_grid, quantum_system);

    int number_of_bands = planewave_iterator->number_of_bands();
    int number_of_spin_components = 1;
    Real inverse_cell_volume = 1.0 / crystal_structure.cell_volume();

    std::vector<std::vector<Real>> charge_density;
    accumulator.accumulate(
        number_of_bands, number_of_spin_components, inverse_cell_volume, charge_density);

    ASSERT_EQ(charge_density.size(), 1u) << "Should have 1 spin component";

    int grid_points = basis.fft_rho().total_grid_points();
    ASSERT_EQ(static_cast<int>(charge_density[0].size()), grid_points);

    // Integrate charge density: total_charge = Σ ρ(r) * (Ω / N_grid)
    Real total_charge = 0.0;
    Real volume_per_grid_point = crystal_structure.cell_volume() / grid_points;
    for (int grid_index = 0; grid_index < grid_points; ++grid_index) {
        total_charge += charge_density[0][grid_index] * volume_per_grid_point;
    }

    // Si has 4 valence electrons per atom × 2 atoms = 8 electrons
    Real expected_number_of_electrons = 8.0;
    EXPECT_NEAR(total_charge, expected_number_of_electrons, 0.01)
        << "Total charge should equal number of valence electrons (8 for two Si atoms)";
}

// ── Test: charge density is non-negative everywhere ──────────────────

TEST(ChargeDensityAccumulator, ChargeDensityIsNonNegative) {
    auto pseudopotential_set = load_norm_conserving_si_pseudopotential();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si_species{"Si", 28.08, pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si_species);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 16.0;
    config.maximum_iterations = 1;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    const auto& basis = planewave_iterator->planewave_basis();
    const auto& quantum_system = planewave_iterator->quantum_system();

    PlanewaveChargeDensityAccumulator accumulator(basis, kpoint_grid, quantum_system);

    int number_of_bands = planewave_iterator->number_of_bands();
    Real inverse_cell_volume = 1.0 / crystal_structure.cell_volume();

    std::vector<std::vector<Real>> charge_density;
    accumulator.accumulate(number_of_bands, 1, inverse_cell_volume, charge_density);

    int grid_points = basis.fft_rho().total_grid_points();
    int negative_count = 0;
    Real most_negative_value = 0.0;

    for (int grid_index = 0; grid_index < grid_points; ++grid_index) {
        if (charge_density[0][grid_index] < -1e-10) {
            negative_count++;
            most_negative_value = std::min(most_negative_value, charge_density[0][grid_index]);
        }
    }

    /**
     * Allow a very small tolerance for numerical noise from FFTs
     * The charge density is |ψ|² which is non-negative by definition,
     * but Fourier truncation can introduce small Gibbs oscillations.
     */
    EXPECT_EQ(negative_count, 0) << "Found " << negative_count << " negative charge density points "
                                 << "(most negative: " << most_negative_value << ")";
}

// ── Test: charge density has correct shape ──────────────────────────

TEST(ChargeDensityAccumulator, OutputHasCorrectShape) {
    auto pseudopotential_set = load_norm_conserving_si_pseudopotential();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si_species{"Si", 28.08, pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si_species);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 16.0;
    config.maximum_iterations = 1;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    const auto& basis = planewave_iterator->planewave_basis();
    const auto& quantum_system = planewave_iterator->quantum_system();

    PlanewaveChargeDensityAccumulator accumulator(basis, kpoint_grid, quantum_system);

    int number_of_bands = planewave_iterator->number_of_bands();
    Real inverse_cell_volume = 1.0 / crystal_structure.cell_volume();

    std::vector<std::vector<Real>> charge_density;
    accumulator.accumulate(number_of_bands, 1, inverse_cell_volume, charge_density);

    // The output should have exactly 1 spin component
    EXPECT_EQ(charge_density.size(), 1u);

    // The grid size should match the rho FFT grid
    int expected_grid_points = basis.fft_rho().total_grid_points();
    EXPECT_EQ(static_cast<int>(charge_density[0].size()), expected_grid_points);

    // The charge density should not be all zeros
    Real maximum_density = 0.0;
    for (int grid_index = 0; grid_index < expected_grid_points; ++grid_index) {
        maximum_density = std::max(maximum_density, charge_density[0][grid_index]);
    }
    EXPECT_GT(maximum_density, 0.0) << "Charge density should have non-zero maximum";
}
