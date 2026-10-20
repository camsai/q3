#include <gtest/gtest.h>

#include <complex>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/pseudo_atomic_orbital_set.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Silicon's PAW dataset: an occupied s and p shell, each followed by the
 * unoccupied second partial wave of the same l -- eight orbitals per atom, the
 * case trial_shells exists for.
 */
struct Fixture {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::vector<int> g_vector_indices;
    int number_of_plane_waves = 0;

    Fixture() {
        pseudopotential_set.push_back(
            UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/paw/Si.upf"));
        crystal_structure.set_from_lattice_type(2, 10.2);
        AtomSpecies silicon{"Si", 28.09, pseudopotential_set[0].valence_charge, "Si.upf"};
        crystal_structure.add_species(silicon);
        crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
        crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
        crystal_structure.compute_derived_quantities();
        basis.generate(serial_policy, crystal_structure, 24.0, 96.0);
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);
        std::vector<Real> kinetic_energy_squared;
        basis.compute_gk(kpoint_grid[0].coordinates,
                         crystal_structure,
                         g_vector_indices,
                         kinetic_energy_squared);
        number_of_plane_waves = static_cast<int>(g_vector_indices.size());
    }

    std::vector<std::vector<Complex>> render(const PlanewavePseudoAtomicOrbitalSet& set) const {
        std::vector<std::vector<Complex>> orbitals;
        set.build(0,
                  number_of_plane_waves,
                  g_vector_indices,
                  crystal_structure,
                  kpoint_grid,
                  basis,
                  orbitals);
        return orbitals;
    }
};

} // namespace

TEST(PlanewavePseudoAtomicOrbitalSet, NamedShellsAreExactlyTheirRowsOfTheWholeSet) {
    Fixture fixture;
    ASSERT_EQ(fixture.pseudopotential_set[0].wavefunctions_atomic.size(), 4u)
        << "the fixture's premise: s, s', p, p'";

    const PlanewavePseudoAtomicOrbitalSet every_shell(fixture.crystal_structure,
                                                      fixture.pseudopotential_set);
    ASSERT_EQ(every_shell.number_of_orbitals(), 16);

    const PlanewavePseudoAtomicOrbitalSet occupied_shells(
        fixture.crystal_structure, fixture.pseudopotential_set, {{"Si", {0, 2}}});
    ASSERT_EQ(occupied_shells.number_of_orbitals(), 8);
    EXPECT_NE(occupied_shells.description().find("8 left out"), std::string::npos)
        << occupied_shells.description();

    const auto whole = fixture.render(every_shell);
    const auto kept = fixture.render(occupied_shells);
    ASSERT_EQ(kept.size(), 8u);
    // Per atom: shell 0 is row 0, shell 1 row 1, shell 2 rows 2-4, shell 3 rows 5-7.
    const std::vector<int> expected_rows = {0, 2, 3, 4, 8, 10, 11, 12};
    for (std::size_t orbital = 0; orbital < kept.size(); orbital++) {
        ASSERT_EQ(kept[orbital].size(), whole[expected_rows[orbital]].size());
        for (std::size_t g = 0; g < kept[orbital].size(); g++) {
            EXPECT_EQ(kept[orbital][g], whole[expected_rows[orbital]][g])
                << "orbital " << orbital << ", plane wave " << g;
        }
        EXPECT_EQ(occupied_shells.orbital_labels()[orbital].angular_momentum,
                  every_shell.orbital_labels()[expected_rows[orbital]].angular_momentum);
    }
}

TEST(PlanewavePseudoAtomicOrbitalSet, ASelectionThatCannotMeanWhatItSaysIsRefused) {
    Fixture fixture;
    EXPECT_THROW(PlanewavePseudoAtomicOrbitalSet(
                     fixture.crystal_structure, fixture.pseudopotential_set, {{"C", {0}}}),
                 std::invalid_argument);
    EXPECT_THROW(PlanewavePseudoAtomicOrbitalSet(
                     fixture.crystal_structure, fixture.pseudopotential_set, {{"Si", {4}}}),
                 std::invalid_argument);
}
