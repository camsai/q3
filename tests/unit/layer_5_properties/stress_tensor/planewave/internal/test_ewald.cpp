#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/ewald.hpp"
#include "utils/boron_phosphate.hpp"

using namespace q3;

namespace {

StressTensorContributions ewald_stress(const std::vector<Atom>& atoms) {
    Crystal crystal = test_support::boron_phosphate_crystal(atoms);
    PlanewaveBasis basis;
    test_support::generate_boron_phosphate_basis(basis, crystal);
    StressTensorContributions contributions;
    EwaldStressTensorCalculator(crystal, basis, contributions).compute();
    return contributions;
}

} // namespace

TEST(EwaldStressTensorCalculator, StressDoesNotDependOnWhichCellEachAtomIsWrittenIn) {
    /**
     * An atom moved by a lattice vector leaves the crystal unchanged, so the stress may not
     * change. As written, the real-space image loop around the raw separation of an O-O pair
     * 1.08 Å apart missed its nearest image, and the pair's strain derivative with it.
     */
    const StressTensorContributions reference = ewald_stress(
        test_support::wrapped_into_home_cell(test_support::boron_phosphate_atoms_as_written()));

    const std::vector<std::pair<const char*, std::vector<Atom>>> layouts = {
        {"as written", test_support::boron_phosphate_atoms_as_written()},
        {"each shifted by a lattice vector",
         test_support::shifted_by_lattice_vectors(
             test_support::boron_phosphate_atoms_as_written())},
    };
    for (const auto& [layout, atoms] : layouts) {
        const StressTensorContributions stress = ewald_stress(atoms);
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                EXPECT_NEAR(stress.ewald[row][column], reference.ewald[row][column], 1e-11)
                    << layout << ", component [" << row << "][" << column << "]";
            }
        }
    }
}
