/**
 * The pseudopotential iterator as the owner of the Kleinman-Bylander projector
 * sets: who reads the same set, and what is kept when.
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set_store.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

using namespace q3;

namespace {

/// Two-atom silicon after two SCF iterations on a shifted 2x2x2 mesh.
struct IteratedSilicon {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    KPointsGrid kpoint_grid;
    std::unique_ptr<KohnShamSCFIterator> scf_iterator;
    PlanewavePseudopotentialKohnShamSCFIterator* iterator = nullptr;

    IteratedSilicon() {
        pseudopotential_set.push_back(
            UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));
        crystal_structure.set_from_lattice_type(2, 10.263048);
        AtomSpecies si_species{"Si", 28.08, pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"};
        crystal_structure.add_species(si_species);
        crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
        crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
        crystal_structure.compute_derived_quantities();
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);

        KohnShamSCFIteratorConfig config;
        config.crystal_structure = &crystal_structure;
        config.number_of_spin_components = 1;
        config.kpoint_grid = &kpoint_grid;
        config.pseudopotential_set = &pseudopotential_set;
        config.wavefunction_energy_cutoff = 16.0;
        config.maximum_iterations = 2;
        scf_iterator = create_kohn_sham_iterator_from_config(config);
        scf_iterator->run();
        iterator = dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator.get());
    }

    [[nodiscard]] int number_of_kpoints() const {
        return kpoint_grid.number_of_kpoints();
    }
};

} // namespace

TEST(PlanewavePseudopotentialKohnShamSCFIterator, TheSCFLeavesEveryKpointsProjectorSetBuilt) {
    IteratedSilicon silicon;
    ASSERT_NE(silicon.iterator, nullptr);
    KleinmanBylanderProjectorSetStore& store = silicon.iterator->projector_set_store();

    ASSERT_EQ(store.number_of_kpoints(), silicon.number_of_kpoints());
    for (int kpoint_index = 0; kpoint_index < silicon.number_of_kpoints(); ++kpoint_index) {
        EXPECT_TRUE(store.is_built(kpoint_index));
        const KleinmanBylanderProjectorSet& projector_set = store.at(kpoint_index);
        EXPECT_EQ(projector_set.number_of_projectors(), 8);
        EXPECT_EQ(projector_set.number_of_plane_waves(),
                  silicon.iterator->number_of_plane_waves(kpoint_index));
        EXPECT_TRUE(
            projector_set.is_built_from(silicon.kpoint_grid[kpoint_index].coordinates,
                                        silicon.crystal_structure,
                                        silicon.iterator->number_of_plane_waves(kpoint_index)));
    }
}

TEST(PlanewavePseudopotentialKohnShamSCFIterator, TheSCFStateReadsTheIteratorsOwnSets) {
    IteratedSilicon silicon;
    ASSERT_NE(silicon.iterator, nullptr);

    NormConservingPlanewaveKohnShamSCFState scf_state = silicon.iterator->result();

    for (int kpoint_index = 0; kpoint_index < silicon.number_of_kpoints(); ++kpoint_index) {
        EXPECT_EQ(&scf_state.projector_set(kpoint_index),
                  &silicon.iterator->projector_set(kpoint_index));
    }
}

TEST(PlanewavePseudopotentialKohnShamSCFIterator, ReleasingTheWorkspaceFreesTheProjectorSets) {
    IteratedSilicon silicon;
    ASSERT_NE(silicon.iterator, nullptr);
    KleinmanBylanderProjectorSetStore& store = silicon.iterator->projector_set_store();

    silicon.iterator->release_scf_workspace();

    for (int kpoint_index = 0; kpoint_index < silicon.number_of_kpoints(); ++kpoint_index) {
        EXPECT_FALSE(store.is_built(kpoint_index));
    }
}

TEST(PlanewavePseudopotentialKohnShamSCFIterator, AfterTheReleaseTheStateGetsOneSetAtATime) {
    IteratedSilicon silicon;
    ASSERT_NE(silicon.iterator, nullptr);
    ASSERT_GE(silicon.number_of_kpoints(), 2);
    NormConservingPlanewaveKohnShamSCFState scf_state = silicon.iterator->result();
    KleinmanBylanderProjectorSetStore& store = silicon.iterator->projector_set_store();
    silicon.iterator->release_scf_workspace();

    // What forces and stress do: walk the k-points, reading each set while at that k-point.
    const KleinmanBylanderProjectorSet& first = scf_state.projector_set(0);
    EXPECT_TRUE(first.is_built());
    EXPECT_FALSE(first.empty());
    EXPECT_FALSE(store.is_built(1));

    const KleinmanBylanderProjectorSet& second = scf_state.projector_set(1);
    EXPECT_TRUE(second.is_built());
    EXPECT_FALSE(store.is_built(0));
}

TEST(PlanewavePseudopotentialKohnShamSCFIterator, ASetRebuiltAfterTheReleaseHoldsTheSameValues) {
    IteratedSilicon silicon;
    ASSERT_NE(silicon.iterator, nullptr);
    const KleinmanBylanderProjectorSet& projector_set = silicon.iterator->projector_set(1);
    const int number_of_elements =
        projector_set.number_of_projectors() * projector_set.number_of_plane_waves();
    const std::vector<Complex> before(projector_set.host(),
                                      projector_set.host() + number_of_elements);

    silicon.iterator->release_scf_workspace();
    const KleinmanBylanderProjectorSet& rebuilt = silicon.iterator->projector_set(1);

    ASSERT_EQ(rebuilt.number_of_projectors() * rebuilt.number_of_plane_waves(), number_of_elements);
    for (int element = 0; element < number_of_elements; ++element) {
        EXPECT_EQ(rebuilt.host()[element], before[static_cast<std::size_t>(element)]);
    }
}

TEST(PlanewavePseudopotentialKohnShamSCFIterator, AStateNoIteratorProducedMakesItsOwnStore) {
    IteratedSilicon silicon;
    ASSERT_NE(silicon.iterator, nullptr);
    NormConservingPlanewaveKohnShamSCFState scf_state = silicon.iterator->result();

    // As a hand-built state has it: no store to share.
    scf_state.projector_sets.reset();
    const KleinmanBylanderProjectorSet& own_set = scf_state.projector_set(0);

    EXPECT_NE(&own_set, &silicon.iterator->projector_set(0));
    EXPECT_EQ(own_set.number_of_projectors(), 8);
    // One set at a time, as the state's consumers read them.
    (void)scf_state.projector_set(1);
    EXPECT_FALSE(scf_state.projector_sets->is_built(0));
    EXPECT_TRUE(scf_state.projector_sets->is_built(1));
}
