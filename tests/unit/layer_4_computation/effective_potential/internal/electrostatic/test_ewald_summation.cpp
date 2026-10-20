#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/electrostatic/ewald_summation.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "utils/boron_phosphate.hpp"
#include <gtest/gtest.h>

#include <utility>
#include <vector>

using namespace q3;

static execution::PlanewaveSerialExecutionPolicy serial_policy;

TEST(EwaldTest, SiliconEwaldEnergy) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048); // FCC silicon
    AtomSpecies silicon{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 16.0, 64.0);

    Real ewald_energy = compute_ewald(crystal_structure, basis);
    // Reference Ewald energy for FCC Silicon: -16.79593991 Ry
    EXPECT_NEAR(ewald_energy, -16.79593991, 1e-6);
}

TEST(EwaldTest, EnergyDoesNotDependOnWhichCellEachAtomIsWrittenIn) {
    /**
     * As written, one O-O pair's nearest image lies outside the real-space image loop around
     * the raw separation, and its term was dropped: -79.866912831 Ry instead of
     * -79.866771929 Ry.
     */
    const std::vector<std::pair<const char*, std::vector<Atom>>> layouts = {
        {"as written", test_support::boron_phosphate_atoms_as_written()},
        {"wrapped into [0, 1)",
         test_support::wrapped_into_home_cell(test_support::boron_phosphate_atoms_as_written())},
        {"each shifted by a lattice vector",
         test_support::shifted_by_lattice_vectors(
             test_support::boron_phosphate_atoms_as_written())},
    };
    for (const auto& [layout, atoms] : layouts) {
        Crystal crystal = test_support::boron_phosphate_crystal(atoms);
        PlanewaveBasis basis;
        test_support::generate_boron_phosphate_basis(basis, crystal);
        EXPECT_NEAR(
            compute_ewald(crystal, basis), test_support::boron_phosphate_ewald_energy_rydberg, 1e-8)
            << layout;
    }
}
