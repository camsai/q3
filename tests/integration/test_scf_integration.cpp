#include <gtest/gtest.h>

#include <cmath>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/davidson.hpp"
#include "layer_4_computation/diagonalizer/planewave/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/internal/hubbard.hpp"

using namespace q3;

// ── Helper: set up Si crystal with PP ───────────────────────────

static PseudopotentialSet load_si_pp() {
    PseudopotentialSet set;
    set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));
    return set;
}

/**
 * Build a KohnShamSCFIteratorConfig for norm-conserving Si with defaults.
 *
 * Populates the crystal, pseudopotential set, and k-point grid pointers.
 */
static KohnShamSCFIteratorConfig build_scf_config(Crystal& crystal,
                                                  KPointsGrid& kpoint_grid,
                                                  PseudopotentialSet& pseudopotential_set,
                                                  int number_of_spin_components = 1) {
    KohnShamSCFIteratorConfig config;
    config.crystal_structure = &crystal;
    config.number_of_spin_components = number_of_spin_components;
    config.kpoint_grid = &kpoint_grid;
    config.pseudopotential_set = &pseudopotential_set;
    return config;
}

// Reference energies: Q3 regression anchors.
static constexpr double REF_GAMMA = -14.573853923; // ecutwfc=16, gamma-only (Q3 converged)
static constexpr double REF_EWALD = -16.79593991;  // Ewald (independent of cutoff)
static constexpr double TOL_ENERGY = 0.01;         // 0.01 Ry — regression guard, not QE parity

// ── Ewald energy (independent of SCF) ───────────────────────────────
TEST(SCFIntegration, EwaldEnergy) {
    auto pps = load_si_pp();

    Crystal cryst2;
    cryst2.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    cryst2.add_species(si);
    cryst2.add_atom({0, {0.0, 0.0, 0.0}});
    cryst2.add_atom({0, {0.25, 0.25, 0.25}});
    cryst2.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(cryst2, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.maximum_iterations = 1;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);

    EXPECT_NEAR(pseudopotential_iterator->energies().ewald_energy, REF_EWALD, 1e-6)
        << "Ewald energy should match QE to high precision";
}

// ── Full SCF: gamma-only ────────────────────────────────────────────
TEST(SCFIntegration, SiFCC_GammaOnly) {
    auto pps = load_si_pp();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.convergence_threshold = 1e-6;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);

    EXPECT_NEAR(pseudopotential_iterator->energies().total(), REF_GAMMA, TOL_ENERGY)
        << "Gamma-only total energy should match QE reference";
}

// ── total_charge: charged-cell electron count ───────────────────────
TEST(SCFIntegration, SiFCC_TotalChargeAdjustsElectronCount) {
    /**
     * QE tot_charge convention: positive = electrons removed (cationic).
     * Si FCC (2 atoms x 4 valence electrons) is neutral at 8 electrons.
     */
    auto pps = load_si_pp();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.convergence_threshold = 1e-6;
    config.total_charge = 1.0; // cationic: one electron removed
    /**
     * An odd electron count cannot be represented by fixed (integer)
     * band occupations at a single k-point -- smearing is required, exactly
     * as a real charged-defect/open-shell calculation would need.
     */
    config.smearing = "gaussian";
    config.smearing_width = 0.01;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);

    EXPECT_TRUE(scf_iterator->is_converged());
    EXPECT_NEAR(pseudopotential_iterator->quantum_system().number_of_electrons(), 7.0, 1e-9);
    // A charged cell must not silently reproduce the neutral-cell energy.
    EXPECT_GT(std::abs(pseudopotential_iterator->energies().total() - REF_GAMMA), 0.1);
}

TEST(SCFIntegration, SiFCC_NegativeTotalChargeAddsElectrons) {
    auto pps = load_si_pp();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.convergence_threshold = 1e-6;
    config.number_of_bands = 8; // extra empty bands for the additional electron
    config.total_charge = -1.0; // anionic: one electron added
    config.smearing = "gaussian";
    config.smearing_width = 0.01;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);

    EXPECT_TRUE(scf_iterator->is_converged());
    EXPECT_NEAR(pseudopotential_iterator->quantum_system().number_of_electrons(), 9.0, 1e-9);
}

// ── Full SCF: 2×2×2 k-mesh ─────────────────────────────────────────
TEST(SCFIntegration, SiFCC_Kpoints2x2x2) {
    auto pps = load_si_pp();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.convergence_threshold = 1e-6;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);

    EXPECT_TRUE(scf_iterator->is_converged()) << "2×2×2 SCF should converge";
    EXPECT_LT(pseudopotential_iterator->energies().total(), -14.0)
        << "Total energy should be below -14 Ry";
}

// ── Eigenvalue degeneracies at Gamma ────────────────────────────────
TEST(SCFIntegration, SiFCC_GammaEigenvalues) {
    auto pps = load_si_pp();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.convergence_threshold = 1e-6;

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);

    // Access eigenvalues via the PW layer
    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    const auto& et = planewave_iterator->eigenvalues();
    ASSERT_GE(et.size(), 1u);
    ASSERT_GE(et.cols(), 4u);

    Real e2 = et[0][1], e3 = et[0][2], e4 = et[0][3];
    /**
     * NOTE: gamma-only half-sphere G-vector storage partially breaks cubic
     * symmetry, so degeneracies are approximate. Use relaxed tolerance.
     */
    EXPECT_NEAR(e2, e3, 0.07) << "Bands 2-3 should be approximately degenerate at Gamma";
    EXPECT_NEAR(e3, e4, 0.07) << "Bands 3-4 should be approximately degenerate at Gamma";
    EXPECT_LT(et[0][0], e2 - 0.3) << "Band gap at Gamma should be > 0.3 Ry";
}

// ── US-PP SCF run ───────────────────────────────────────────────────
TEST(SCFIntegration, SiFCC_USPP_SCF) {
    auto pp =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");
    ASSERT_EQ(pp->pseudopotential_type_enum(), PseudopotentialType::ultrasoft);

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.334401769190510);
    AtomSpecies si{"Si", 28.0855, pp->valence_charge, "Si_PBE_USPP.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.875, 0.875, 0.875}});
    crystal_structure.add_atom({0, {0.125, 0.125, 0.125}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);
    kpoint_grid = SymmetryAnalyzer::reduce_kpoints(crystal_structure, kpoint_grid);

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(std::move(pp));

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 40.0;
    config.charge_density_energy_cutoff = 200.0;
    config.convergence_threshold = 1e-6;
    config.mixing_mode = "anderson";
    config.mixing_beta = 0.3;
    config.maximum_iterations = 200;
    config.exchange_correlation_functional = "PBE";

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    EXPECT_TRUE(scf_iterator->is_converged()) << "Si US-PP SCF should converge";

    Real total_energy = scf_iterator->total_energy();
    EXPECT_LT(total_energy, -15.0) << "Total energy should be below -15 Ry";
    EXPECT_GT(total_energy, -16.5) << "Total energy should be reasonable";
}

// ── Davidson gives same eigenvalues as direct on same potential ──
TEST(SCFIntegration, DavidsonSingleShot) {
    auto pps = load_si_pp();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    // Run direct SCF to get converged potential and eigenvalues
    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.convergence_threshold = 1e-6;
    config.diagonalization_method = "direct";

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();
    ASSERT_TRUE(scf_iterator->is_converged()) << "Direct SCF should converge";

    // Downcast to access PW-level internals for the Davidson single-shot test
    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);

    // Get the converged potential and k-point data
    const auto& real_space_potential = planewave_iterator->real_space_potential();
    const auto& g_vector_indices = planewave_iterator->g_vector_indices();
    const auto& kinetic_energy_squared = planewave_iterator->kinetic_energy_squared();
    const auto& planewave_basis = planewave_iterator->planewave_basis();
    int number_of_plane_waves = (int)g_vector_indices[0].size();

    // Direct eigenvalues (reference)
    auto eigenvalue_matrix = planewave_iterator->eigenvalues();
    int number_of_bands = static_cast<int>(eigenvalue_matrix.cols());
    std::vector<Real> direct_eigenvalues(number_of_bands);
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        direct_eigenvalues[band_index] = eigenvalue_matrix[0][band_index];
    }

    // Now call Davidson on the same potential
    auto davidson_diagonalizer = create_planewave_diagonalizer("davidson");
    if (auto* davidson_diagonalizer_ptr =
            dynamic_cast<PlanewaveDavidsonDiagonalizer*>(davidson_diagonalizer.get())) {
        davidson_diagonalizer_ptr->set_tolerance(1e-8);
    }

    // Build a standalone quantum system for the davidson solve
    PlanewaveKohnShamQuantumSystem davidson_quantum_system(
        kpoint_grid,
        planewave_basis,
        number_of_bands,
        1,
        planewave_iterator->quantum_system().number_of_electrons());
    davidson_quantum_system.allocate_electronic_states(
        planewave_basis.kpoint_number_of_plane_waves());
    // Copy wavefunctions from converged direct result
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        const Complex* src =
            planewave_iterator->quantum_system().wavefunction_band_data(0, band_index);
        Complex* dst = davidson_quantum_system.wavefunction_band_data(0, band_index);
        std::copy(src, src + number_of_plane_waves, dst);
    }

    NormConservingPlanewaveKohnShamQuantumSystem pseudopotential_quantum_system;
    pseudopotential_quantum_system.init(crystal_structure, planewave_basis, pps);

    davidson_diagonalizer->solve(0,
                                 0,
                                 number_of_plane_waves,
                                 number_of_bands,
                                 g_vector_indices[0],
                                 kinetic_energy_squared[0],
                                 real_space_potential,
                                 crystal_structure,
                                 kpoint_grid,
                                 planewave_basis,
                                 pseudopotential_quantum_system,
                                 davidson_quantum_system);

    /**
     * Occupied eigenvalues should match.
     * NOTE: with gamma-only half-sphere, the direct diagonalizer (LAPACK)
     * and Davidson can converge to slightly different eigenvalues because
     * the broken symmetry creates near-degenerate pairs that can mix
     * differently. A tolerance of 0.30 Ry (~4 eV) accounts for the
     * worst-case near-degenerate mixing at this tiny basis (ecutwfc=16 Ry).
     */
    int number_of_occupied_bands = 4;
    for (int band_index = 0; band_index < number_of_occupied_bands; band_index++) {
        EXPECT_NEAR(
            direct_eigenvalues[band_index], davidson_quantum_system.eigenvalue(0, band_index), 0.30)
            << "Occupied band " << band_index << " eigenvalue should match";
    }
}

/**
 * ── Davidson: a capped solve counts the bands it returns unconverged ──
 *
 * A non-self-consistent pass diagonalizes once and keeps what the solver
 * returns, so a solve that runs out of iterations has to say how many bands
 * it left short of the tolerance instead of handing them back as converged.
 * Started from plane waves -- a quantum system with no wavefunctions yet --
 * one iteration cannot converge every band to the tightest tolerance; the cap
 * and working basis a non-self-consistent pass runs with must, and must land on
 * a direct diagonalization of the same potential.
 *
 * Occupied bands are held to the tolerance and agree to 1.1e-7 Ry here. Empty
 * bands stop at the solver's relaxed empty-band test and agree to 1.7e-5 Ry, so
 * their bound is looser -- but still ten times tighter than the 9.4e-4 Ry the
 * top band was off by at a working basis of twice the band count, which is what
 * a non-self-consistent pass ran with before it got a floor.
 */
TEST(SCFIntegration, DavidsonCountsTheBandsACappedSolveLeavesUnconverged) {
    auto pps = load_si_pp();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    /**
     * Shifted, so no k-point is Gamma and the direct reference is exact rather
     * than a half-sphere approximation.
     */
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.convergence_threshold = 1e-10;
    config.diagonalization_method = "direct";
    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();
    ASSERT_TRUE(scf_iterator->is_converged()) << "Direct SCF should converge";

    auto* planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(planewave_iterator, nullptr);
    const auto& g_vector_indices = planewave_iterator->g_vector_indices();
    const auto& kinetic_energy_squared = planewave_iterator->kinetic_energy_squared();
    const auto& planewave_basis = planewave_iterator->planewave_basis();
    int number_of_plane_waves = static_cast<int>(g_vector_indices[0].size());
    int number_of_bands = static_cast<int>(planewave_iterator->eigenvalues().cols());

    NormConservingPlanewaveKohnShamQuantumSystem pseudopotential_quantum_system;
    pseudopotential_quantum_system.init(crystal_structure, planewave_basis, pps);

    struct Solve {
        int unconverged;
        std::vector<Real> eigenvalues;
    };
    auto solve_from_plane_waves = [&](const std::string& method, int maximum_iterations) {
        auto diagonalizer = create_planewave_diagonalizer(method);
        diagonalizer->set_tolerance(math::tolerances::minimum_eigenvalue_tolerance);
        diagonalizer->set_basis_expansion_factor(
            constants::non_self_consistent_field_minimum_davidson_basis_expansion_factor);
        diagonalizer->set_maximum_iterations(maximum_iterations);
        PlanewaveKohnShamQuantumSystem quantum_system(
            kpoint_grid,
            planewave_basis,
            number_of_bands,
            1,
            planewave_iterator->quantum_system().number_of_electrons());
        quantum_system.allocate_electronic_states(planewave_basis.kpoint_number_of_plane_waves());
        diagonalizer->solve(0,
                            0,
                            number_of_plane_waves,
                            number_of_bands,
                            g_vector_indices[0],
                            kinetic_energy_squared[0],
                            planewave_iterator->real_space_potential(),
                            crystal_structure,
                            kpoint_grid,
                            planewave_basis,
                            pseudopotential_quantum_system,
                            quantum_system);
        Solve result{diagonalizer->number_of_unconverged_bands(), {}};
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            result.eigenvalues.push_back(quantum_system.eigenvalue(0, band_index));
        }
        return result;
    };

    Solve capped = solve_from_plane_waves("davidson", 1);
    EXPECT_GT(capped.unconverged, 0)
        << "one iteration from plane waves reported every band converged to 1e-13";
    EXPECT_LE(capped.unconverged, number_of_bands);

    Solve full = solve_from_plane_waves(
        "davidson", constants::non_self_consistent_field_maximum_davidson_iterations);
    EXPECT_EQ(full.unconverged, 0)
        << "the non-self-consistent cap left bands unconverged at a basis twice the band count";

    Solve direct = solve_from_plane_waves("direct", 1);
    EXPECT_EQ(direct.unconverged, 0) << "a direct diagonalization has nothing to converge";
    int number_of_occupied_bands =
        planewave_iterator->quantum_system().number_of_occupied_bands_per_spin();
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        Real bound = band_index < number_of_occupied_bands ? 1e-6 : 1e-4;
        EXPECT_NEAR(full.eigenvalues[band_index], direct.eigenvalues[band_index], bound)
            << "band " << band_index << " did not land on the direct eigenvalue";
    }
}

// ── Davidson: full SCF convergence ───────────────────────────────────
TEST(SCFIntegration, DavidsonSCF) {
    auto pps = load_si_pp();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, pps[0].valence_charge, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 16.0;
    config.convergence_threshold = 1e-6;
    config.diagonalization_method = "davidson";

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();
    EXPECT_TRUE(scf_iterator->is_converged()) << "Davidson SCF should converge";

    Real total_energy = scf_iterator->total_energy();
    EXPECT_NEAR(total_energy, REF_GAMMA, TOL_ENERGY)
        << "Davidson total energy should match direct SCF";
}

// ── DFT+U (Hubbard) ──────────────────────────────────────────────────

static PseudopotentialSet load_nickel_pseudopotentials() {
    PseudopotentialSet set;
    set.push_back(UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Ni.upf"));
    return set;
}

/**
 * FCC Ni, spin-polarized, U=3 eV on the 3D shell. Converges and reports a
 * correctly-signed, physically-reasonable Hubbard energy and the same
 * converged magnetization (0.50 Bohr mag/cell) as an identical run without
 * Hubbard.
 *
 * PBE is set explicitly to match Ni.upf's own functional (the test would
 * otherwise silently fall back to KohnShamSCFIteratorConfig's PZ default,
 * an inconsistent functional/pseudopotential pairing). The k-point grid is
 * IBZ-reduced and spin-weight-halved by hand because this test builds a
 * KPointsGrid directly instead of going through
 * build_global_kpoint_grid_from_input(), which normally does both -- see
 * KPointsGrid::scale_weights()'s docstring. Skipping that step was a real,
 * previously-undetected bug in this exact test: every k-point-summed energy
 * term was computed on an un-symmetry-reduced, un-halved (2x too heavy)
 * weight set, which alone accounted for most of a ~0.9 Ry Q3-vs-QE
 * discrepancy once blamed on the Hubbard implementation (TODOs.md).
 *
 * With that fixed, a from-scratch QE comparison (HUBBARD {atomic}, same
 * system, same PBE pseudopotential) narrows the gap to total energy
 * -323.87 Ry (Q3) vs -323.25 Ry (QE, ~0.6 Ry) and Hubbard energy 0.144 Ry
 * (Q3) vs 0.164 Ry (QE, ~12%) -- both substantially improved from the
 * stale ~0.9 Ry / ~30% figures. Directly comparing occupation-matrix
 * traces (Tr[n], the basis-invariant total d-electron count Q3's Hubbard
 * energy formula integrates over) shows Q3 8.061 vs QE 8.077 -- agreement
 * to ~0.2%, which rules out projector normalization/construction as the
 * residual gap's cause. The remaining difference is consistent with the
 * well-documented DFT+U sensitivity to *how* a fixed d-electron count is
 * distributed across the near-degenerate cubic eg/t2g manifold (same Tr[n],
 * different Tr[n^2], i.e. a different but each self-consistent occupation
 * pattern) rather than an implementation bug -- see TODOs.md for the full
 * writeup.
 */
TEST(SCFIntegration, NiFCC_HubbardUConverges) {
    auto pps = load_nickel_pseudopotentials();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.625);
    AtomSpecies nickel{"Ni", 58.6934, pps[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);
    kpoint_grid = SymmetryAnalyzer::reduce_kpoints_internal(crystal_structure, kpoint_grid);
    kpoint_grid.scale_weights(0.5); // nspin=2: see KPointsGrid::scale_weights()'s docstring

    auto config =
        build_scf_config(crystal_structure, kpoint_grid, pps, /*number_of_spin_components=*/2);
    config.exchange_correlation_functional = "PBE"; // matches Ni.upf's own functional
    config.wavefunction_energy_cutoff = 30.0;
    config.convergence_threshold = 1e-6;
    config.maximum_iterations = 200;
    config.smearing = "gaussian";
    config.smearing_width = 0.02;
    config.mixing_mode = "broyden";
    config.mixing_beta = 0.1;
    config.starting_magnetization = {0.3};
    config.hubbard = {{"Ni", "3D", 3.0 * 2.0 / constants::HARTREE_TO_EV}};

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);

    EXPECT_TRUE(scf_iterator->is_converged()) << "DFT+U SCF should converge";
    Real hubbard_energy = pseudopotential_iterator->energies().hubbard_energy_contribution();
    EXPECT_GT(hubbard_energy, 0.0) << "E_U = (U/2)Tr[n(1-n)] is never negative";
    /**
     * Regression anchors for this exact config.
     *
     * The total moved by **+0.600 Ry on 2026-08-08** when the DFT+U
     * double-counting term was added: the Hubbard potential enters the band
     * energy through the Hamiltonian, so Tr[V_U n] has to come back out before
     * E_U is added. It never did, and because Tr[V_U n] = U(½Tr n − Tr n²) is
     * negative for a partially filled shell, every DFT+U total was too low.
     *
     * That is almost certainly the "~0.6 Ry unexplained residual" PR #38
     * recorded after fixing this test's k-point weighting — the magnitudes
     * match. The pre-2026-08-08 value here was described as QE-cross-checked;
     * it was not reproducing QE, it was reproducing the bug. E_U itself was
     * always right, which is why only the total anchor moved.
     *
     * Moved again the same day when the occupation matrix started being
     * **symmetrized** over the point group (`symmetrize_hubbard_occupation_
     * matrices`). An IBZ sum does not reproduce a full-zone sum for a
     * quantity carrying angular indices; the trace was right and the structure
     * was not, which showed up in E_U (0.1444 -> 0.1637 here) and not in Tr[n].
     *
     * The earlier weighting bug moved the total by ~47 Ry and hubbard_energy
     * by ~4x, so these tolerances still catch a regression of that kind.
     */
    EXPECT_NEAR(pseudopotential_iterator->energies().total(), -323.2454, 0.01)
        << "Total energy drifted from the reference value";
    EXPECT_NEAR(hubbard_energy, 0.1637, 0.01) << "Hubbard energy drifted from the reference value";
}

/**
 * Same fixture as NiFCC_HubbardUConverges, but with hubbard_projectors:
 * ortho-atomic. Naively, one might expect atomic and ortho-atomic to coincide
 * on this single-atom, high-symmetry system (3D vs 3S/3P should be exactly
 * angular-momentum-orthogonal in the continuum). Empirically they do NOT: the
 * actual G-space atomic-orbital overlap matrix (build_atomic_orbitals_g_space)
 * has non-negligible 3S-3D / 4S-3D cross terms (up to ~0.18, and 3S-4S ~-0.37)
 * -- see TODOs.md's DFT+U ortho-atomic write-up for the measured matrix and the
 * two reasons this happens (3S/4S come from different pseudo-atomic reference
 * configurations with no reason to be radially orthogonal even though they
 * share l=0; the finite plane-wave cutoff samples a discrete, only-cubically-
 * symmetric G-vector set, which does not exactly preserve continuum spherical-
 * harmonic orthogonality between different l). So ortho-atomic is a genuinely
 * different, real regression anchor here, not a "should match atomic" smoke
 * test -- these values were measured directly from this exact configuration,
 * not derived analytically.
 */
TEST(SCFIntegration, NiFCC_HubbardOrthoAtomicConvergesToDifferentEnergyThanAtomic) {
    auto pps = load_nickel_pseudopotentials();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.625);
    AtomSpecies nickel{"Ni", 58.6934, pps[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);
    kpoint_grid = SymmetryAnalyzer::reduce_kpoints_internal(crystal_structure, kpoint_grid);
    kpoint_grid.scale_weights(0.5);

    auto config =
        build_scf_config(crystal_structure, kpoint_grid, pps, /*number_of_spin_components=*/2);
    config.exchange_correlation_functional = "PBE";
    config.wavefunction_energy_cutoff = 30.0;
    config.convergence_threshold = 1e-6;
    config.maximum_iterations = 200;
    config.smearing = "gaussian";
    config.smearing_width = 0.02;
    config.mixing_mode = "broyden";
    config.mixing_beta = 0.1;
    config.starting_magnetization = {0.3};
    config.hubbard = {{"Ni", "3D", 3.0 * 2.0 / constants::HARTREE_TO_EV}};
    config.hubbard_projector_type = "ortho-atomic";

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);

    EXPECT_TRUE(scf_iterator->is_converged()) << "DFT+U (ortho-atomic) SCF should converge";
    Real hubbard_energy = pseudopotential_iterator->energies().hubbard_energy_contribution();
    /**
     * Regression anchors measured directly from this exact configuration (see
     * comment above for why these genuinely differ from atomic mode's -323.866 /
     * 0.1444 Ry regression anchors, rather than coinciding with them).
     */
    EXPECT_NEAR(pseudopotential_iterator->energies().total(), -323.2761, 0.01)
        << "ortho-atomic total energy drifted from its measured regression anchor";
    EXPECT_NEAR(hubbard_energy, 0.1313, 0.01)
        << "ortho-atomic Hubbard energy drifted from its measured regression anchor";
}

/**
 * Ortho-atomic DFT+U forces used to be refused outright; they are now derived
 * (HubbardForceCalculator::compute_with_orthogonal_projectors), including the
 * term from every atomic orbital's own position-dependence through the joint
 * Lowdin S^-1/2 transform.
 *
 * This cell has one atom on a site with full cubic symmetry, so the only
 * force vector its point group admits is zero -- which is exactly why the
 * quantitative gate lives in tests/validation/test_paw_forces_and_stress.py
 * on two-atom cells with an off-symmetry displacement. What this asserts is
 * that the path runs and respects the symmetry it is handed; a term with the
 * wrong index order would generally break it.
 */
TEST(SCFIntegration, HubbardOrthoAtomicForcesAreDerivedAndVanishBySymmetry) {
    auto pps = load_nickel_pseudopotentials();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.625);
    AtomSpecies nickel{"Ni", 58.6934, pps[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);
    kpoint_grid = SymmetryAnalyzer::reduce_kpoints_internal(crystal_structure, kpoint_grid);
    kpoint_grid.scale_weights(0.5);

    auto config =
        build_scf_config(crystal_structure, kpoint_grid, pps, /*number_of_spin_components=*/2);
    config.exchange_correlation_functional = "PBE";
    config.wavefunction_energy_cutoff = 30.0;
    config.convergence_threshold = 1e-6;
    config.maximum_iterations = 200;
    config.smearing = "gaussian";
    config.smearing_width = 0.02;
    config.mixing_mode = "broyden";
    config.mixing_beta = 0.1;
    config.starting_magnetization = {0.3};
    config.hubbard = {{"Ni", "3D", 3.0 * 2.0 / constants::HARTREE_TO_EV}};
    config.hubbard_projector_type = "ortho-atomic";

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    scf_iterator->run();

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);
    ASSERT_TRUE(scf_iterator->is_converged());

    auto scf_state = pseudopotential_iterator->result();
    HubbardForceCalculator hubbard_force_calculator(*scf_state.crystal_structure,
                                                    *scf_state.planewave_basis,
                                                    *scf_state.kpoint_grid,
                                                    scf_state);
    std::vector<Vector3D> hubbard_force;
    ASSERT_NO_THROW(hubbard_force = hubbard_force_calculator.compute());
    ASSERT_EQ(hubbard_force.size(), 1u);
    for (int direction = 0; direction < 3; direction++) {
        EXPECT_NEAR(hubbard_force[0][direction], 0.0, 1e-8)
            << "the ortho-atomic Hubbard force on a cubic site must vanish, direction "
            << direction;
    }
}

TEST(SCFIntegration, HubbardUnrecognizedProjectorTypeThrows) {
    auto pps = load_nickel_pseudopotentials();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.625);
    AtomSpecies nickel{"Ni", 58.6934, pps[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 20.0;
    config.hubbard = {{"Ni", "3D", 0.2}};
    config.hubbard_projector_type = "wannier"; // not a recognized value

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    EXPECT_THROW(scf_iterator->run(), std::runtime_error);
}

TEST(SCFIntegration, HubbardUWithUltrasoftRuns) {
    /**
     * DFT+U with an ultrasoft dataset used to be refused outright. It now runs:
     * the Hubbard projectors are premultiplied by the overlap operator, so the
     * occupation matrix is built from <phi|S|psi> — the all-electron inner
     * product — rather than from the bare pseudo overlap, which for a
     * transition metal sees under a third of the shell.
     *
     * Silicon's 3S manifold is not a physically interesting Hubbard target;
     * this is here because it is the ultrasoft dataset the repository ships
     * that carries labelled atomic orbitals, and what is being exercised is
     * that the ultrasoft path completes and produces a finite energy.
     * `tests/validation/test_paw_forces_and_stress.py` gates the occupations
     * themselves, on nickel, where the number means something.
     */
    auto pp =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");
    ASSERT_EQ(pp->pseudopotential_type_enum(), PseudopotentialType::ultrasoft);

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.334401769190510);
    AtomSpecies si{"Si", 28.0855, pp->valence_charge, "Si_PBE_USPP.UPF"};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.875, 0.875, 0.875}});
    crystal_structure.add_atom({0, {0.125, 0.125, 0.125}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(std::move(pp));

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 20.0;
    config.hubbard = {{"Si", "3S", 0.2}};

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    ASSERT_NO_THROW(scf_iterator->run());
}

TEST(SCFIntegration, HubbardUWithGammaOnlyRuns) {
    /**
     * This asserted the opposite -- that DFT+U refused a Gamma-only basis --
     * until 2026-08-09, and it is rewritten rather than deleted because the
     * refusal is exactly what a regression here would restore.
     *
     * A single unshifted Gamma point makes Q3 store half the G sphere, so every
     * DFT+U overlap needs completing to its full-sphere value. That the run now
     * reaches a positive E_U is the shallowest thing to check and the cheapest
     * place to catch a throw coming back; the numbers are gated end to end in
     * `tests/validation/test_gamma_only_hubbard.py`, against Q3's own full
     * sphere and its own energy gradient.
     */
    auto pps = load_nickel_pseudopotentials();

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.625);
    AtomSpecies nickel{"Ni", 58.6934, pps[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0); // Gamma-only: no shifts, 1x1x1

    auto config = build_scf_config(crystal_structure, kpoint_grid, pps);
    config.wavefunction_energy_cutoff = 20.0;
    config.hubbard = {{"Ni", "3D", 0.2}};

    auto scf_iterator = create_kohn_sham_iterator_from_config(config);
    ASSERT_NO_THROW(scf_iterator->run());

    auto* pseudopotential_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    ASSERT_NE(pseudopotential_iterator, nullptr);
    EXPECT_GT(pseudopotential_iterator->energies().hubbard_energy_contribution(), 0.0)
        << "E_U = (U/2)Tr[n(1-n)] is positive semi-definite, so a Gamma-only run that "
           "reports zero or a negative value has not projected onto the manifold at all";
}

TEST(SCFIntegration, HubbardURaisesTheTotalEnergy) {
    /**
     * E_U = (U/2)Tr[n(1−n)] is positive semi-definite, so switching U on must
     * *raise* the total energy. Until 2026-08-08 Q3 lowered it — by ~1 Ry on
     * silicon and ~0.6 Ry here — because the Hubbard potential entered the band
     * energy through the Hamiltonian and Tr[V_U n] was never taken back out.
     *
     * This asserts the *direction* rather than a value, so it survives any
     * legitimate change to the reference numbers while still failing the moment
     * the double-counting term goes missing again. The magnitude is anchored by
     * `NiFCC_HubbardUConverges` above.
     */
    auto run_with = [](Real u_electron_volts) {
        auto pseudopotentials = load_nickel_pseudopotentials();
        Crystal crystal_structure;
        crystal_structure.set_from_lattice_type(2, 6.625);
        AtomSpecies nickel{"Ni", 58.6934, pseudopotentials[0].valence_charge, "Ni.upf"};
        crystal_structure.add_species(nickel);
        crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
        crystal_structure.compute_derived_quantities();

        KPointsGrid kpoint_grid;
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);
        kpoint_grid = SymmetryAnalyzer::reduce_kpoints_internal(crystal_structure, kpoint_grid);
        kpoint_grid.scale_weights(0.5); // nspin=2

        auto config = build_scf_config(
            crystal_structure, kpoint_grid, pseudopotentials, /*number_of_spin_components=*/2);
        config.exchange_correlation_functional = "PBE";
        config.wavefunction_energy_cutoff = 30.0;
        config.convergence_threshold = 1e-6;
        config.maximum_iterations = 200;
        config.smearing = "gaussian";
        config.smearing_width = 0.02;
        config.mixing_beta = 0.1;
        config.starting_magnetization = {0.3};
        if (u_electron_volts > 0.0)
            config.hubbard = {{"Ni", "3D", u_electron_volts * 2.0 / constants::HARTREE_TO_EV}};

        auto iterator = create_kohn_sham_iterator_from_config(config);
        iterator->run();
        auto* pseudopotential_iterator =
            dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(iterator.get());
        EXPECT_NE(pseudopotential_iterator, nullptr);
        return pseudopotential_iterator == nullptr ? 0.0
                                                   : pseudopotential_iterator->energies().total();
    };

    Real without = run_with(0.0);
    Real with = run_with(3.0);
    EXPECT_GT(with, without) << "switching U on must raise the total energy: " << without << " -> "
                             << with << " Ry";
}

TEST(SCFIntegration, HubbardOccupationsAreSymmetryAdapted) {
    /**
     * The IBZ and the full zone must give the same DFT+U energy.
     *
     * The occupation matrix carries angular indices, so a weighted
     * irreducible-wedge sum does not reproduce a full-zone sum the way an
     * invariant scalar does — it has to be averaged over the point group
     * afterwards, as Quantum ESPRESSO does when it builds the occupation
     * matrix. Until 2026-08-08 Q3 did not, and the two disagreed: on silicon's
     * 3p manifold, which symmetry forces to be isotropic, E_U came out
     * 0.1204 Ry from the wedge against 0.1918 from the full zone (QE: 0.1918).
     *
     * Gated as an invariance rather than against a reference value: it needs no
     * external code, and it is the property whose violation *was* the bug.
     * Nickel rather than silicon because `load_nickel_pseudopotentials` is the fixture already
     * here and its 3d manifold makes the anisotropy larger.
     */
    auto run_with_symmetry = [](bool reduce_to_irreducible_wedge) {
        auto pseudopotentials = load_nickel_pseudopotentials();
        Crystal crystal_structure;
        crystal_structure.set_from_lattice_type(2, 6.625);
        AtomSpecies nickel{"Ni", 58.6934, pseudopotentials[0].valence_charge, "Ni.upf"};
        crystal_structure.add_species(nickel);
        crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
        crystal_structure.compute_derived_quantities();

        KPointsGrid kpoint_grid;
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);
        if (reduce_to_irreducible_wedge)
            kpoint_grid = SymmetryAnalyzer::reduce_kpoints_internal(crystal_structure, kpoint_grid);
        kpoint_grid.scale_weights(0.5); // nspin=2

        auto config = build_scf_config(
            crystal_structure, kpoint_grid, pseudopotentials, /*number_of_spin_components=*/2);
        config.exchange_correlation_functional = "PBE";
        config.wavefunction_energy_cutoff = 30.0;
        config.convergence_threshold = 1e-6;
        config.maximum_iterations = 200;
        config.smearing = "gaussian";
        config.smearing_width = 0.02;
        config.mixing_beta = 0.1;
        config.starting_magnetization = {0.3};
        config.hubbard = {{"Ni", "3D", 3.0 * 2.0 / constants::HARTREE_TO_EV}};

        auto iterator = create_kohn_sham_iterator_from_config(config);
        iterator->run();
        auto* pseudopotential_iterator =
            dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(iterator.get());
        EXPECT_NE(pseudopotential_iterator, nullptr);
        return pseudopotential_iterator == nullptr
                   ? 0.0
                   : pseudopotential_iterator->energies().hubbard_energy_contribution();
    };

    Real wedge = run_with_symmetry(true);
    Real full_zone = run_with_symmetry(false);
    EXPECT_NEAR(wedge, full_zone, 5e-3) << "DFT+U energy differs between the irreducible wedge ("
                                        << wedge << ") and the full zone (" << full_zone
                                        << "), so the occupation matrix is not being symmetrized";
}
