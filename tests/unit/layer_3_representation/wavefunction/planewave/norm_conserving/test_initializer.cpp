#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"

using namespace q3;

/**
 * Helper: build a Si FCC crystal with norm-conserving pseudopotential.
 *
 * Uses ibrav=2 (FCC), alat=10.263048 Bohr, and two Si atoms at
 * (0,0,0) and (1/4,1/4,1/4).
 */
static PseudopotentialSet load_norm_conserving_si_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));
    return pseudopotential_set;
}

/**
 * Build a Si FCC crystal with the given pseudopotential set.
 */
static Crystal build_si_fcc_crystal(const PseudopotentialSet& pseudopotential_set) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si_species{"Si", 28.08, pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si_species);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
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

/**
 * Compute the gamma-only inner product ⟨ψ_a|ψ_b⟩ for two wavefunction bands.
 *
 * For gamma-only storage (half-sphere), the inner product is:
 *   ⟨a|b⟩ = conj(a[0])*b[0] + 2 * Re( Σ_{G>0} conj(a[G])*b[G] )
 *
 * This arises because each stored G>0 component represents both G and -G.
 */
static Real compute_gamma_inner_product(const Complex* band_a,
                                        const Complex* band_b,
                                        int number_of_plane_waves) {
    // G=0 term
    Real inner_product = (std::conj(band_a[0]) * band_b[0]).real();
    // G>0 terms: factor of 2 for gamma-only half-sphere
    for (int g_index = 1; g_index < number_of_plane_waves; ++g_index) {
        inner_product += 2.0 * (std::conj(band_a[g_index]) * band_b[g_index]).real();
    }
    return inner_product;
}

// ── Test: wavefunctions are orthonormal after initialization ─────────

TEST(WavefunctionInitializer, OrthonormalityAfterInitialization) {
    auto pseudopotential_set = load_norm_conserving_si_pseudopotential();
    auto crystal_structure = build_si_fcc_crystal(pseudopotential_set);

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 16.0;
    config.maximum_iterations = 1;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    const auto& quantum_system = planewave_iterator->quantum_system();
    int number_of_bands = planewave_iterator->number_of_bands();
    int number_of_plane_waves = planewave_iterator->number_of_plane_waves(0);

    ASSERT_GE(number_of_bands, 4) << "Should have at least 4 bands for Si (4 valence electrons)";
    ASSERT_GT(number_of_plane_waves, 0) << "Should have positive number of plane waves";

    /**
     * Check orthonormality: ⟨ψ_i|ψ_j⟩ = δ_ij
     * Use gamma-only inner product since this is a Gamma-only calculation
     */
    Real tolerance = 1e-6;

    for (int band_i = 0; band_i < number_of_bands; ++band_i) {
        const Complex* wavefunction_i = quantum_system.wavefunction_band_data(0, band_i);

        for (int band_j = band_i; band_j < number_of_bands; ++band_j) {
            const Complex* wavefunction_j = quantum_system.wavefunction_band_data(0, band_j);

            Real inner_product =
                compute_gamma_inner_product(wavefunction_i, wavefunction_j, number_of_plane_waves);

            Real expected_value = (band_i == band_j) ? 1.0 : 0.0;
            EXPECT_NEAR(inner_product, expected_value, tolerance)
                << "Inner product <psi_" << band_i << "|psi_" << band_j << "> should be "
                << expected_value;
        }
    }
}

// ── Test: each wavefunction is individually normalized ───────────────

TEST(WavefunctionInitializer, IndividualNormalization) {
    auto pseudopotential_set = load_norm_conserving_si_pseudopotential();
    auto crystal_structure = build_si_fcc_crystal(pseudopotential_set);

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 16.0;
    config.maximum_iterations = 1;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    const auto& quantum_system = planewave_iterator->quantum_system();
    int number_of_bands = planewave_iterator->number_of_bands();
    int number_of_plane_waves = planewave_iterator->number_of_plane_waves(0);

    Real tolerance = 1e-6;

    for (int band_index = 0; band_index < number_of_bands; ++band_index) {
        const Complex* wavefunction_data = quantum_system.wavefunction_band_data(0, band_index);

        Real norm_squared = compute_gamma_inner_product(
            wavefunction_data, wavefunction_data, number_of_plane_waves);

        EXPECT_NEAR(norm_squared, 1.0, tolerance)
            << "Wavefunction band " << band_index << " should be normalized (||psi||^2 = 1)";
    }
}

// ── Test: eigenvalues are ordered ────────────────────────────────────

TEST(WavefunctionInitializer, EigenvaluesAreOrdered) {
    auto pseudopotential_set = load_norm_conserving_si_pseudopotential();
    auto crystal_structure = build_si_fcc_crystal(pseudopotential_set);

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 16.0;
    config.maximum_iterations = 1;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    int number_of_bands = planewave_iterator->number_of_bands();
    ASSERT_GE(number_of_bands, 2);

    for (int band_index = 1; band_index < number_of_bands; ++band_index) {
        Real eigenvalue_current = planewave_iterator->eigenvalue(0, band_index);
        Real eigenvalue_previous = planewave_iterator->eigenvalue(0, band_index - 1);
        EXPECT_GE(eigenvalue_current, eigenvalue_previous - 1e-10)
            << "Eigenvalue for band " << band_index << " should be >= eigenvalue for band "
            << (band_index - 1);
    }
}

// ── Test: the random padding follows the plane wave, not its position ─

/**
 * Where the atomic orbitals do not reach the band count the starting set is
 * padded with random vectors, and ranks sharing a pool each hold a different
 * part of the plane-wave list, in their own order. Every coefficient has to
 * follow its plane wave for all of them to start from the same guess: listing
 * the plane waves in reverse must list every starting vector in reverse. Drawn
 * from a generator stepped along the list, the padding came out different for
 * every order.
 */
TEST(WavefunctionInitializer, RandomPaddingFollowsThePlaneWaveNotItsPosition) {
    auto pseudopotential_set = load_norm_conserving_si_pseudopotential();
    auto crystal_structure = build_si_fcc_crystal(pseudopotential_set);

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 16.0;
    config.maximum_iterations = 1;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();
    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    const PlanewaveBasis& planewave_basis = planewave_iterator->planewave_basis();
    const std::vector<int>& indices = planewave_basis.g_vector_indices_by_kpoint_index(0);
    std::vector<int> reversed_indices(indices.rbegin(), indices.rend());
    int number_of_plane_waves = static_cast<int>(indices.size());
    constexpr int number_of_bands = 16; // twice the eight atomic orbitals of two Si atoms

    NormConservingPlanewaveWavefunctionInitializer initializer;
    initializer.initialize_interpolation_table(pseudopotential_set,
                                               crystal_structure.cell_volume());
    std::vector<std::vector<Complex>> in_order;
    std::vector<std::vector<Complex>> in_reverse;
    initializer.compute_atomic_wavefunctions(0,
                                             number_of_plane_waves,
                                             number_of_bands,
                                             indices,
                                             crystal_structure,
                                             kpoint_grid,
                                             planewave_basis,
                                             pseudopotential_set,
                                             in_order);
    initializer.compute_atomic_wavefunctions(0,
                                             number_of_plane_waves,
                                             number_of_bands,
                                             reversed_indices,
                                             crystal_structure,
                                             kpoint_grid,
                                             planewave_basis,
                                             pseudopotential_set,
                                             in_reverse);

    ASSERT_EQ(static_cast<int>(in_order.size()), number_of_bands);
    ASSERT_EQ(in_reverse.size(), in_order.size());
    Real worst = 0.0;
    for (std::size_t vector_index = 0; vector_index < in_order.size(); ++vector_index) {
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            worst = std::max(worst,
                             std::abs(in_order[vector_index][static_cast<std::size_t>(g_index)] -
                                      in_reverse[vector_index][static_cast<std::size_t>(
                                          number_of_plane_waves - 1 - g_index)]));
        }
    }
    EXPECT_LT(worst, 1e-14);
}
