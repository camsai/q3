#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_5_properties/forces/planewave/internal/ewald.hpp"
#include "utils/boron_phosphate.hpp"

using namespace q3;

namespace {

std::vector<Vector3D> ewald_forces(const std::vector<Atom>& atoms) {
    Crystal crystal = test_support::boron_phosphate_crystal(atoms);
    PlanewaveBasis basis;
    test_support::generate_boron_phosphate_basis(basis, crystal);
    return EwaldForceCalculator(crystal, basis).compute();
}

} // namespace

TEST(EwaldForceCalculator, ForcesDoNotDependOnWhichCellEachAtomIsWrittenIn) {
    /**
     * An atom moved by a lattice vector leaves the crystal unchanged, so no force may change.
     * As written, the real-space image loop around the raw separation of an O-O pair 1.08 Å
     * apart missed its nearest image, and both atoms lost that pair's force.
     */
    const std::vector<Vector3D> reference = ewald_forces(
        test_support::wrapped_into_home_cell(test_support::boron_phosphate_atoms_as_written()));

    const std::vector<std::pair<const char*, std::vector<Atom>>> layouts = {
        {"as written", test_support::boron_phosphate_atoms_as_written()},
        {"each shifted by a lattice vector",
         test_support::shifted_by_lattice_vectors(
             test_support::boron_phosphate_atoms_as_written())},
    };
    for (const auto& [layout, atoms] : layouts) {
        const std::vector<Vector3D> forces = ewald_forces(atoms);
        ASSERT_EQ(forces.size(), reference.size()) << layout;
        for (size_t atom = 0; atom < forces.size(); atom++) {
            for (int direction = 0; direction < 3; direction++) {
                EXPECT_NEAR(forces[atom][direction], reference[atom][direction], 1e-9)
                    << layout << ", atom " << atom << ", direction " << direction;
            }
        }
    }
}
