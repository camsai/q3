#include <gtest/gtest.h>

#include <memory>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/ultrasoft/init.hpp"

using namespace q3;

/**
 * Helper: build a KohnShamSCFIteratorConfig for integration-test-style usage.
 */
static KohnShamSCFIteratorConfig build_scf_config(Crystal& crystal_structure,
                                                  KPointsGrid& kpoint_grid,
                                                  PseudopotentialSet& pseudopotential_set,
                                                  int number_of_spin_components = 1) {
    KohnShamSCFIteratorConfig config;
    config.crystal_structure = &crystal_structure;
    config.number_of_spin_components = number_of_spin_components;
    config.kpoint_grid = &kpoint_grid;
    config.pseudopotential_set = &pseudopotential_set;
    return config;
}

// ── Test: NC pseudopotential produces non-null iterator ─────────────

TEST(KohnShamFactory, NormConservingProducesNonNullIterator) {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));

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

    auto iterator = create_kohn_sham_iterator_from_config(config);
    ASSERT_NE(iterator, nullptr) << "Factory should return a non-null iterator for NC PP";
}

// ── Test: NC pseudopotential dispatches to NC concrete type ─────────

TEST(KohnShamFactory, NormConservingDispatchesToCorrectType) {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));

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

    auto iterator = create_kohn_sham_iterator_from_config(config);
    ASSERT_NE(iterator, nullptr);

    // Should be a PlanewavePseudopotentialKohnShamSCFIterator (the NC type)
    auto* norm_conserving_iterator =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(iterator.get());
    EXPECT_NE(norm_conserving_iterator, nullptr)
        << "NC pseudopotential should dispatch to PlanewavePseudopotentialKohnShamSCFIterator";

    // Should NOT be the ultrasoft subclass
    auto* ultrasoft_iterator =
        dynamic_cast<UltrasoftPlanewavePseudopotentialKohnShamSCFIterator*>(iterator.get());
    EXPECT_EQ(ultrasoft_iterator, nullptr)
        << "NC pseudopotential should NOT dispatch to "
           "UltrasoftPlanewavePseudopotentialKohnShamSCFIterator";
}

// ── Test: USPP dispatches to ultrasoft concrete type ────────────────

TEST(KohnShamFactory, UltrasoftDispatchesToCorrectType) {
    auto ultrasoft_pseudopotential =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");
    ASSERT_EQ(ultrasoft_pseudopotential->pseudopotential_type_enum(),
              PseudopotentialType::ultrasoft)
        << "Loaded PP should be ultrasoft type";

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.334401769190510);
    AtomSpecies si_species{
        "Si", 28.0855, ultrasoft_pseudopotential->valence_charge, "Si_PBE_USPP.UPF"};
    crystal_structure.add_species(si_species);
    crystal_structure.add_atom({0, {0.875, 0.875, 0.875}});
    crystal_structure.add_atom({0, {0.125, 0.125, 0.125}});
    crystal_structure.compute_derived_quantities();

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(std::move(ultrasoft_pseudopotential));

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto config = build_scf_config(crystal_structure, kpoint_grid, pseudopotential_set);
    config.wavefunction_energy_cutoff = 40.0;
    config.charge_density_energy_cutoff = 200.0;
    config.maximum_iterations = 1;
    config.exchange_correlation_functional = "PBE";

    auto iterator = create_kohn_sham_iterator_from_config(config);
    ASSERT_NE(iterator, nullptr) << "Factory should return a non-null iterator for USPP";

    // Should be the ultrasoft subclass
    auto* ultrasoft_iterator =
        dynamic_cast<UltrasoftPlanewavePseudopotentialKohnShamSCFIterator*>(iterator.get());
    EXPECT_NE(ultrasoft_iterator, nullptr)
        << "Ultrasoft pseudopotential should dispatch to "
           "UltrasoftPlanewavePseudopotentialKohnShamSCFIterator";

    /**
     * WHICH of the iterator's two quantum systems is the polymorphic one.
     *
     * The iterator holds both: `quantum_system()` returns a by-value member
     * declared as the base PlanewaveKohnShamQuantumSystem, so its dynamic type
     * is always exactly that whatever the dataset; the augmentation-aware
     * object the factory built is the snapshot behind
     * `norm_conserving_quantum_system()`.
     *
     * This is asserted because the Wannier capture downcast the wrong one, and
     * a downcast that can never succeed is invisible: build_ultrasoft had four
     * green unit tests of its own and was unreachable for every ultrasoft and
     * PAW run. A caller choosing between these two accessors is choosing
     * between "works" and "refuses everything", so the difference is pinned
     * here rather than left to be rediscovered.
     *
     * The snapshot is built during initialization rather than construction, so
     * the iterator has to have run once -- config.maximum_iterations is 1
     * above. A caller reaching for it before that gets a null, which is itself
     * worth knowing.
     */
    ultrasoft_iterator->run();

    auto* polymorphic = dynamic_cast<const UltrasoftPlanewaveKohnShamQuantumSystem*>(
        &ultrasoft_iterator->norm_conserving_quantum_system());
    EXPECT_NE(polymorphic, nullptr)
        << "the snapshot is the ultrasoft quantum system for an ultrasoft dataset";
    auto* sliced = dynamic_cast<const UltrasoftPlanewaveKohnShamQuantumSystem*>(
        &ultrasoft_iterator->quantum_system());
    EXPECT_EQ(sliced, nullptr) << "quantum_system() is the base-typed member and never the "
                                  "ultrasoft one -- downcasting it is the defect this pins";
}

// ── Test: NC and USPP dispatch to different types ───────────────────

TEST(KohnShamFactory, NormConservingAndUltrasoftDispatchToDifferentTypes) {
    // Load NC pseudopotential
    PseudopotentialSet norm_conserving_pseudopotential_set;
    norm_conserving_pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));

    Crystal norm_conserving_crystal;
    norm_conserving_crystal.set_from_lattice_type(2, 10.263048);
    AtomSpecies norm_conserving_si_species{
        "Si", 28.08, norm_conserving_pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"};
    norm_conserving_crystal.add_species(norm_conserving_si_species);
    norm_conserving_crystal.add_atom({0, {0.0, 0.0, 0.0}});
    norm_conserving_crystal.add_atom({0, {0.25, 0.25, 0.25}});
    norm_conserving_crystal.compute_derived_quantities();

    KPointsGrid norm_conserving_kpoint_grid;
    norm_conserving_kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto norm_conserving_config = build_scf_config(
        norm_conserving_crystal, norm_conserving_kpoint_grid, norm_conserving_pseudopotential_set);
    norm_conserving_config.wavefunction_energy_cutoff = 16.0;
    norm_conserving_config.maximum_iterations = 1;

    auto norm_conserving_iterator = create_kohn_sham_iterator_from_config(norm_conserving_config);

    // Load USPP
    auto ultrasoft_pseudopotential =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");

    Crystal ultrasoft_crystal;
    ultrasoft_crystal.set_from_lattice_type(2, 10.334401769190510);
    AtomSpecies ultrasoft_si_species{
        "Si", 28.0855, ultrasoft_pseudopotential->valence_charge, "Si_PBE_USPP.UPF"};
    ultrasoft_crystal.add_species(ultrasoft_si_species);
    ultrasoft_crystal.add_atom({0, {0.875, 0.875, 0.875}});
    ultrasoft_crystal.add_atom({0, {0.125, 0.125, 0.125}});
    ultrasoft_crystal.compute_derived_quantities();

    PseudopotentialSet ultrasoft_pseudopotential_set;
    ultrasoft_pseudopotential_set.push_back(std::move(ultrasoft_pseudopotential));

    KPointsGrid ultrasoft_kpoint_grid;
    ultrasoft_kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    auto ultrasoft_config =
        build_scf_config(ultrasoft_crystal, ultrasoft_kpoint_grid, ultrasoft_pseudopotential_set);
    ultrasoft_config.wavefunction_energy_cutoff = 40.0;
    ultrasoft_config.charge_density_energy_cutoff = 200.0;
    ultrasoft_config.maximum_iterations = 1;
    ultrasoft_config.exchange_correlation_functional = "PBE";

    auto ultrasoft_iterator = create_kohn_sham_iterator_from_config(ultrasoft_config);

    // The NC iterator should NOT be ultrasoft
    auto* norm_conserving_as_ultrasoft =
        dynamic_cast<UltrasoftPlanewavePseudopotentialKohnShamSCFIterator*>(
            norm_conserving_iterator.get());
    EXPECT_EQ(norm_conserving_as_ultrasoft, nullptr) << "NC iterator should not be ultrasoft type";

    // The USPP iterator SHOULD be ultrasoft
    auto* ultrasoft_as_ultrasoft =
        dynamic_cast<UltrasoftPlanewavePseudopotentialKohnShamSCFIterator*>(
            ultrasoft_iterator.get());
    EXPECT_NE(ultrasoft_as_ultrasoft, nullptr) << "USPP iterator should be ultrasoft type";

    // Both should be PlanewavePseudopotentialKohnShamSCFIterator (the base)
    auto* norm_conserving_as_base =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(norm_conserving_iterator.get());
    auto* ultrasoft_as_base =
        dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(ultrasoft_iterator.get());
    EXPECT_NE(norm_conserving_as_base, nullptr);
    EXPECT_NE(ultrasoft_as_base, nullptr);
}
